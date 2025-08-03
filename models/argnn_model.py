import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, scatter
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

class MetricNetwork(nn.Module):
    """Efficient network for learning diagonal metric coefficients."""
    
    def __init__(self, input_dim, output_dim, hidden_dim, activation='softplus', min_metric_value=0.001, max_metric_value=10.0):
        super().__init__()
        
        self.min_metric_value = min_metric_value
        self.max_metric_value = max_metric_value
        
        # Lightweight network for metric learning
        self.net = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Activation to ensure positive metrics
        if activation == 'softplus':
            self.activation = nn.Softplus(beta=2)
        elif activation == 'exp':
            self.activation = lambda x: torch.exp(x.clamp(-5, 5))
        else:
            self.activation = lambda x: F.relu(x) + self.min_metric_value
            
    def forward(self, x, x_neighbors):
        """
        Learn diagonal metric coefficients.
        
        Args:
            x: Node features (N, input_dim)
            x_neighbors: Aggregated neighbor features (N, input_dim)
        Returns:
            g: Diagonal metric coefficients (N, output_dim)
        """
        combined = torch.cat([x, x_neighbors], dim=-1)
        raw_metrics = self.net(combined)
        
        # Ensure positive and bounded metrics
        g = self.activation(raw_metrics)
        g = g.clamp(min=self.min_metric_value, max=self.max_metric_value)
        
        return g

class ARGNNLayer(MessagePassing):
    """Efficient geometric message passing with diagonal metric tensors."""
    
    def __init__(self, in_channels, out_channels, metric_hidden_dim=64):
        super().__init__(aggr='add', flow='source_to_target')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Diagonal metric learning network
        self.metric_net = MetricNetwork(
            in_channels, in_channels, metric_hidden_dim
        )
        
        # Message and update transformations
        self.message_lin = nn.Linear(in_channels, out_channels, bias=False)
        self.self_lin = nn.Linear(in_channels, out_channels, bias=True)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(out_channels)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.message_lin.weight)
        nn.init.xavier_uniform_(self.self_lin.weight)
        nn.init.zeros_(self.self_lin.bias)
        
    def forward(self, x, edge_index):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute neighbor aggregation for metric learning
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        
        # Fast neighbor feature aggregation
        x_neighbors = scatter(x[row], col, dim=0, dim_size=x.size(0), reduce='mean')
        
        # Learn diagonal metric coefficients
        self.metric_coeffs = self.metric_net(x, x_neighbors)  # (N, d)
        
        # Message passing
        out = self.propagate(edge_index, x=x, metric_coeffs=self.metric_coeffs)
        
        # Self-connection
        out = out + self.self_lin(x)
        
        # Normalization
        out = self.norm(out)
        
        return out, self.metric_coeffs
    
    def message(self, x_i, x_j, metric_coeffs_i, index, ptr, size_i):
        """Compute geometric messages with diagonal metrics."""
        # Compute direction vector efficiently
        diff = x_j - x_i
        diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
        d_ij = diff / diff_norm  # Normalized direction
        
        # Compute directional propagation coefficient
        # For diagonal metric: τ_ij = Σ_k d²_ijk * tanh(-log(g_ik))
        d_squared = d_ij * d_ij  # Element-wise square
        log_metrics = torch.log(metric_coeffs_i.clamp(min=1e-8))
        tau_components = d_squared * torch.tanh(-log_metrics)
        tau = tau_components.sum(dim=-1, keepdim=True)  # (E, 1)
        
        # Compute attention using diagonal Mahalanobis distance
        # α_ij = (x_i^T G_i x_j) / (||x_i||_G_i ||x_j||_G_j)
        # For diagonal G: x^T G y = Σ_k g_k * x_k * y_k
        weighted_inner = (metric_coeffs_i * x_i * x_j).sum(dim=-1, keepdim=True)
        norm_i_sq = (metric_coeffs_i * x_i * x_i).sum(dim=-1, keepdim=True)
        norm_j_sq = (metric_coeffs_i * x_j * x_j).sum(dim=-1, keepdim=True)
        
        alpha = torch.sigmoid(weighted_inner / (torch.sqrt(norm_i_sq * norm_j_sq) + 1e-8))
        
        # Final message
        message = tau * alpha * self.message_lin(x_j)
        
        return message
    
    def aggregate(self, inputs, index, ptr, dim_size):
        """Efficient aggregation using scatter operations."""
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='add')

class ARGNNModel(nn.Module):
    """Adaptive Riemannian Graph Neural Networks (using learnable Layer-Adaptive Diagonal Metric)"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.5, 
                 metric_hidden_dim=64, metric_reg=0.01, smoothness_reg=0.01, ricci_reg=0):
        super().__init__()
        
        self.num_layers = num_layers
        self.metric_reg = metric_reg
        self.smoothness_reg = smoothness_reg
        self.ricci_reg = ricci_reg
        # Initial feature transformation
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Geometric convolution layers
        self.conv_layers = nn.ModuleList()
        dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = dims[i]
            self.conv_layers.append(ARGNNLayer(in_dim, out_dim, metric_hidden_dim))
        
        # Dropout for intermediate layers
        self.dropout = nn.Dropout(dropout)
        
        # Store metric coefficients for regularization
        self.metric_coeffs_list = []
        
    def forward(self, data):
        x = data.x if hasattr(data, 'x') else data['x']
        edge_index = data.edge_index if hasattr(data, 'edge_index') else data['edge_index']
        
        # Initial encoding
        x = self.input_encoder(x)
        
        # Reset metric storage
        self.metric_coeffs_list = []
        
        # Layer-wise propagation
        for i, conv in enumerate(self.conv_layers):
            x, metric_coeffs = conv(x, edge_index)
            self.metric_coeffs_list.append(metric_coeffs)
            
            # Apply activation and dropout for intermediate layers
            if i < self.num_layers - 1:
                x = F.relu(x, inplace=True)
                x = self.dropout(x)
        
        return F.log_softmax(x, dim=1)
    
    def encode(self, x, edge_index, kappa=None):
        """
        Generates node embeddings for link prediction.
        
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            kappa (Tensor, optional): Curvature values (not used in ARGNN).
        Returns:
            Tensor: Node embeddings.
        """
        # Create data object
        data = {'x': x, 'edge_index': edge_index}
        
        # Initial encoding
        x = self.input_encoder(x)
        
        # Reset metric storage
        self.metric_coeffs_list = []
        
        # Layer-wise propagation (excluding final classification layer)
        for i, conv in enumerate(self.conv_layers[:-1]):
            x, metric_coeffs = conv(x, edge_index)
            self.metric_coeffs_list.append(metric_coeffs)
            x = F.relu(x, inplace=True)
            x = self.dropout(x)
        
        # Normalize embeddings to unit norm
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def decode(self, z, edge_index):
        """
        Inner product decoder for Link Prediction.
        
        Args:
            z (Tensor): Node embeddings.
            edge_index (Tensor): Edge indices.
        
        Returns:
            Tensor: Scores for each edge.
        """
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)
    
    def test(self, z, pos_edge_index, neg_edge_index):
        """
        Evaluates the model on Link Prediction metrics.
        
        Args:
            z (Tensor): Node embeddings.
            pos_edge_index (Tensor): Positive edge indices.
            neg_edge_index (Tensor): Negative edge indices.
        
        Returns:
            Tuple[float, float, float, float]: AUC, AP, F1, and Accuracy scores.
        """
        pos_scores = self.decode(z, pos_edge_index).detach().cpu().numpy()
        neg_scores = self.decode(z, neg_edge_index).detach().cpu().numpy()
        
        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones(pos_scores.shape[0]), np.zeros(neg_scores.shape[0])])
        
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        
        # Calculate F1 and Accuracy using threshold 0.5
        predictions = (scores > 0.5).astype(int)
        f1 = f1_score(labels, predictions)
        acc = accuracy_score(labels, predictions)
        
        return auc, ap, f1, acc
    
 
    def compute_regularization_loss(self, edge_index):
        if not self.metric_coeffs_list:
            return 0.0

        reg_loss = 0.0
        row, col = edge_index

        for metric_coeffs in self.metric_coeffs_list:
            # (i) Metric magnitude (L2)
            reg_loss += self.metric_reg * metric_coeffs.pow(2).mean()

            # (ii) Smoothness
            diff = metric_coeffs[row] - metric_coeffs[col]
            reg_loss += self.smoothness_reg * diff.pow(2).mean()

            # (iii) Ricci regularization
            if self.ricci_reg > 0:
                agg_diff = scatter(diff, row, dim=0,
                                dim_size=metric_coeffs.size(0), reduce='mean')
                ricci_node = 0.5 * agg_diff     # -1/2 factor
                reg_loss += self.ricci_reg * ricci_node.pow(2).mean()

        return reg_loss

    
    def get_metrics(self):

        metrics = {}
        if not self.metric_coeffs_list:
            return metrics

        with torch.no_grad():  
            for layer_idx, g in enumerate(self.metric_coeffs_list):
                if g is None:          
                    continue
                
                if g.device.type == 'cuda':
                    metrics[f"G{layer_idx}"] = g.detach().cpu().numpy()
                else:
                    metrics[f"G{layer_idx}"] = g.detach().numpy()
        return metrics

