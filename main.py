"""
Main Training Script for ARGNN (Adaptive Riemannian Graph Neural Networks)

This script provides the main entry point for training and evaluating ARGNN models
on both node classification and link prediction tasks.
"""

import argparse
import time
import torch
import torch.nn.functional as F
from torch_geometric.utils import train_test_split_edges, to_undirected
from models.argnn_model import ARGNNModel
import networkx as nx
import numpy as np
import geoopt
from datetime import datetime

# Import utility functions
from utils.seed_utils import set_seed
from utils.logging_utils import (
    create_log_directory,
    save_epoch_records,
    save_best_model,
    save_global_best_info
)
from utils.data_utils import load_dataset, load_datasets, apply_custom_splits
from utils.train_utils import train_node_classification, train_link_prediction


def main():
    """
    Main function for training ARGNN models.

    Handles argument parsing, dataset loading, model initialization,
    training loop execution, and results logging.
    """
    parser = argparse.ArgumentParser(description="CUSP Model Training with Node Classification and Link Prediction")

    # Basic training arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of experiment runs')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'Citeseer', 'PubMed', 'Chameleon', 'Actor', 'Squirrel', 'Texas', 'Cornell', 'Wisconsin'],
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='argnn', choices=['argnn'], help='Model to use (ARGNN or baselines)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'radam'], help='Optimizer to use')
    parser.add_argument('--task', type=str, default='node_classification',
                        choices=['node_classification', 'link_prediction'], help='Task to perform')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension size')

    # ARGNN-specific arguments
    parser.add_argument('--argnn_hidden_dim', type=int, default=128, help='Hidden dimension for ARGNN')
    parser.add_argument('--argnn_num_layers', type=int, default=3, help='Number of layers for ARGNN')
    parser.add_argument('--argnn_metric_hidden_dim', type=int, default=64, help='Hidden dimension for metric network in ARGNN')
    parser.add_argument('--argnn_metric_reg', type=float, default=0.01, help='L2 regularization on metric coefficients')
    parser.add_argument('--argnn_smoothness_reg', type=float, default=0.01, help='Smoothness regularization for ARGNN')
    parser.add_argument('--argnn_ricci_reg', type=float, default=0, help='Ricci regularization for ARGNN')

    # Data and logging arguments
    parser.add_argument('--use_splits', type=str, default="./splits",
                        help='Path to Official GeomGCN splits directory (e.g., ./splits) Please download from Github and load it')
    parser.add_argument('--logs', type=str, default='./logs', help='Base directory for logging (default: ./logs)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Create log directory and save configuration
    log_dir = create_log_directory(args.logs, args.dataset, args)
    print(f"\n=== Training Session Started ===")
    print(f"Log directory: {log_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize all records structure
    all_records = {
        'runs': {},
        'summary': {
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_time': 0,
            'avg_epoch_time': 0
        }
    }

    # Collect metrics over runs
    metrics = []
    global_best_test_f1 = -1
    global_best_run_idx = -1
    global_best_metrics = None
    global_best_model_state = None

    # Record start time
    total_start_time = time.time()

    for run in range(args.num_runs):
        print(f"\nRun {run + 1}/{args.num_runs}")
        run_start_time = time.time()

        # Set random seed for this run
        current_seed = args.seed + run
        set_seed(current_seed)

        # Initialize run records
        run_records = {
            'epochs': [],
            'best_epoch': -1,
            'best_metrics': None,
            'run_time': 0,
            'seed': current_seed
        }

        # # Load datasets (reload for each run to ensure randomness in splits)
        # datasets = load_datasets()
        # dataset = datasets.get(args.dataset)
        
        # Load only the specified dataset (on-demand loading to avoid unnecessary downloads)
        dataset = load_dataset(args.dataset)
        if dataset is None:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        data = dataset[0]

        # Apply custom splits if use_splits is provided
        data = apply_custom_splits(data, args, run, args.num_runs)

        num_nodes = data.x.shape[0]

        # Set model input and output dimensions
        input_dim = data.num_features
        output_dim = dataset.num_classes

        # Convert edge_index to NetworkX graph
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        edge_list = edge_index.t().tolist()  # Shape: (E, 2)

        G = nx.Graph()
        G.add_edges_from(edge_list)

        # Assign edge weights as ones to recover standard graph Laplacian
        num_edges = data.edge_index.size(1)
        data.edge_weight = torch.ones(num_edges, dtype=torch.float, device=data.edge_index.device)
        data.kappa = torch.zeros(data.num_nodes, dtype=torch.float, device=data.edge_index.device)  # All curvatures are 0 in Euclidean space

        # If task is link prediction, split edges
        if args.task == 'link_prediction':
            # Preserve node features before splitting edges
            x = data.x.clone()

            # Ensure the graph is undirected
            data.edge_index = to_undirected(data.edge_index)

            # Split edges into train/val/test sets
            data = train_test_split_edges(data)

            # Restore node features
            data.x = x

        # Initialize model
        if args.model == 'argnn':
            model = ARGNNModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=args.argnn_hidden_dim,
                num_layers=args.argnn_num_layers,
                dropout=args.dropout,
                metric_hidden_dim=args.argnn_metric_hidden_dim,
                metric_reg=args.argnn_metric_reg,
                smoothness_reg=args.argnn_smoothness_reg,
                ricci_reg=args.argnn_ricci_reg
            )
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        model = model.to(device)
        data = data.to(device)

        # Define optimizer (Choose between Riemannian Adam and the traditional Adam operator)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'radam':
            optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr, stabilize=10)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        # Training loop
        if args.task == 'node_classification':
            criterion = F.nll_loss
            # Training loop
            best_test_f1, best_test_acc, best_test_auc, best_test_prc, run_records = train_node_classification(
                model, data, optimizer, scheduler, args, run_records, log_dir, run)
            best_metric = (best_test_f1, best_test_acc, best_test_auc, best_test_prc)
        elif args.task == 'link_prediction':
            criterion = F.binary_cross_entropy_with_logits
            # Training loop
            best_metric, run_records = train_link_prediction(
                model, data, optimizer, scheduler, args, run_records, log_dir, run)
            best_test_f1 = best_metric[2]  # F1 is the third element
        else:
            raise ValueError(f"Unsupported task: {args.task}")

        # Record run time
        run_end_time = time.time()
        run_records['run_time'] = run_end_time - run_start_time

        # Store run records
        all_records['runs'][f'run_{run}'] = run_records

        # Check if this is the global best
        if best_test_f1 > global_best_test_f1:
            global_best_test_f1 = best_test_f1
            global_best_run_idx = run
            global_best_metrics = best_metric
            global_best_model_state = model.state_dict()

            # Save global best model
            save_best_model(model, log_dir, args.dataset, run, is_global_best=True, global_best_run=run)

        metrics.append(best_metric)
        print(f"Run {run + 1} completed in {run_records['run_time']:.2f} seconds")

    # Record total time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    all_records['summary']['total_time'] = total_time
    all_records['summary']['end_time'] = datetime.now().isoformat()

    # Calculate average epoch time across all runs
    total_epochs = 0
    total_epoch_time = 0
    for run_key, run_data in all_records['runs'].items():
        for epoch_data in run_data['epochs']:
            if 'epoch_time' in epoch_data:
                total_epochs += 1
                total_epoch_time += epoch_data['epoch_time']

    if total_epochs > 0:
        all_records['summary']['avg_epoch_time'] = total_epoch_time / total_epochs

    # Store global best information
    all_records['global_best'] = {
        'run_idx': global_best_run_idx,
        'test_f1': global_best_test_f1,
        'metrics': global_best_metrics
    }

    # Save global best information
    save_global_best_info(log_dir, args.dataset, global_best_run_idx, global_best_metrics)

    # Save all records
    save_epoch_records(log_dir, all_records, args.dataset)

    # Compute average and standard deviation of best metrics
    avg_metric = np.mean(metrics)
    std_metric = np.std(metrics)
    print(f"\n=== Final Results over {args.num_runs} runs ===")

    if args.task == 'node_classification':
        f1_scores, acc_scores, auc_scores, prc_scores = zip(*metrics)
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        avg_acc = np.mean(acc_scores)
        std_acc = np.std(acc_scores)
        avg_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        avg_prc = np.mean(prc_scores)
        std_prc = np.std(prc_scores)
        print(f'Best Test F1 Score: {avg_f1:.4f} ± {std_f1:.4f}, CI: ± {1.96 * std_f1 / np.sqrt(args.num_runs):.4f}')
        print(f'Best Test Accuracy: {avg_acc:.4f} ± {std_acc:.4f}, CI: ± {1.96 * std_acc / np.sqrt(args.num_runs):.4f}')
        print(f'Best Test AUROC: {avg_auc:.4f} ± {std_auc:.4f}, CI: ± {1.96 * std_auc / np.sqrt(args.num_runs):.4f}')
        print(f'Best Test AUPRC: {avg_prc:.4f} ± {std_prc:.4f}, CI: ± {1.96 * std_prc / np.sqrt(args.num_runs):.4f}')
    elif args.task == 'link_prediction':
        aucs, aps, f1s, accs = zip(*metrics)
        avg_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        avg_ap = np.mean(aps)
        std_ap = np.std(aps)
        avg_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f'Best Test AUROC: {avg_auc:.4f} ± {std_auc:.4f}, CI: ± {1.96 * std_auc / np.sqrt(args.num_runs):.4f}')
        print(f'Best Test AUPRC: {avg_ap:.4f} ± {std_ap:.4f}, CI: ± {1.96 * std_ap / np.sqrt(args.num_runs):.4f}')
        print(f'Best Test F1 Score: {avg_f1:.4f} ± {std_f1:.4f}, CI: ± {1.96 * std_f1 / np.sqrt(args.num_runs):.4f}')
        print(f'Best Test Accuracy: {avg_acc:.4f} ± {std_acc:.4f}, CI: ± {1.96 * std_acc / np.sqrt(args.num_runs):.4f}')

    print(f"\n=== Training Summary ===")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average epoch time: {all_records['summary']['avg_epoch_time']:.4f} seconds")
    print(f"Global best run: {global_best_run_idx + 1} (Test F1: {global_best_test_f1:.4f})")
    print(f"\n=== Saved Files ===")
    print(f"Log directory: {log_dir}")
    print(f"Configuration: run_configs.json")
    print(f"Training records: {args.dataset}_records.pkl")
    print(f"Global best model: {args.dataset}_global_best_model_{global_best_run_idx}.pth")
    print(f"Global best info: {args.dataset}_global_best_evaluation_results.pkl")
    for run in range(args.num_runs):
        print(f"Run {run} best model: {args.dataset}_best_{run}.pth")


if __name__ == '__main__':
    main()
