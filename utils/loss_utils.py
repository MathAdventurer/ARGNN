"""
Loss Utilities for Training

This module provides loss functions and helper functions for training,
including link prediction loss and negative edge sampling.
"""

import torch
import torch.nn.functional as F


def link_prediction_loss(model, z, pos_edge_index, neg_edge_index):
    """
    Compute link prediction loss for both positive and negative edges using the inner product decoder.

    Args:
        model (nn.Module): Model with decode method
        z (Tensor): Node embeddings
        pos_edge_index (Tensor): Positive edge indices
        neg_edge_index (Tensor): Negative edge indices

    Returns:
        Tensor: Binary cross-entropy loss
    """
    # Positive edge loss
    pos_logits = model.decode(z, pos_edge_index)
    pos_labels = torch.ones(pos_logits.size(0), device=pos_logits.device)

    # Negative edge loss
    neg_logits = model.decode(z, neg_edge_index)
    neg_labels = torch.zeros(neg_logits.size(0), device=neg_logits.device)

    # Concatenate positive and negative logits and labels
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([pos_labels, neg_labels])

    # Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss


def sample_neg_edges_from_mask(neg_adj_mask, num_neg_edges):
    """
    Sample negative edges from the negative adjacency mask.

    Args:
        neg_adj_mask (Tensor): Negative adjacency mask of shape [num_nodes, num_nodes]
        num_neg_edges (int): Number of negative edges to sample

    Returns:
        Tensor: Negative edge indices of shape [2, num_neg_edges]

    Raises:
        ValueError: If not enough negative edges are available to sample
    """
    # Get all possible negative edge indices
    neg_edge_indices = torch.nonzero(neg_adj_mask, as_tuple=False).t()  # Shape: [2, num_neg_edges_available]

    num_neg_available = neg_edge_indices.size(1)
    if num_neg_available < num_neg_edges:
        raise ValueError(f"Not enough negative edges to sample: requested {num_neg_edges}, available {num_neg_available}")

    # Randomly permute and select the required number of negative edges
    perm = torch.randperm(num_neg_available)
    neg_edge_index = neg_edge_indices[:, perm[:num_neg_edges]]

    return neg_edge_index
