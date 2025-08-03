"""
Data Utilities for Dataset Loading and Splitting

This module provides functions for loading datasets, handling data splits,
and managing train/validation/test masks.
"""

import os
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB


def load_split_masks(splits_path, dataset_name, split_idx):
    """
    Load train/val/test masks from GeomGCN split files.

    Args:
        splits_path (str): Path to splits directory
        dataset_name (str): Name of the dataset (e.g., 'Cora')
        split_idx (int): Split index (0-9)

    Returns:
        tuple: (train_mask, val_mask, test_mask) as torch tensors

    Raises:
        FileNotFoundError: If the split file does not exist
    """
    dataset_name_lower = dataset_name.lower()
    split_file = f"{dataset_name_lower}_split_0.6_0.2_{split_idx}.npz"

    # Special case for Actor dataset (uses 'film' prefix)
    if dataset_name_lower == 'actor':
        split_file = f"film_split_0.6_0.2_{split_idx}.npz"

    split_path = os.path.join(splits_path, split_file)

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split_data = np.load(split_path)
    train_mask = torch.from_numpy(split_data['train_mask']).bool()
    val_mask = torch.from_numpy(split_data['val_mask']).bool()
    test_mask = torch.from_numpy(split_data['test_mask']).bool()

    return train_mask, val_mask, test_mask


def get_split_index(run, num_runs):
    """
    Get the split index for a given run.

    This function maps run indices to split indices (0-9).
    For runs beyond 10, it cycles through the available splits.

    Args:
        run (int): Current run number (0-based)
        num_runs (int): Total number of runs

    Returns:
        int: Split index (0-9)
    """
    if num_runs <= 10:
        return run
    else:
        # For runs beyond 10, cycle through splits 0-9
        return run % 10


def load_datasets():
    """
    Load all supported datasets with appropriate transformations.

    Returns:
        dict: Dictionary mapping dataset names to loaded datasets
    """
    # Define transforms
    transforms_undirected = T.ToUndirected()
    transforms_with_norm = T.Compose([
        T.ToUndirected(),
        T.NormalizeFeatures()
    ])

    datasets = {
        "Cora": Planetoid(root="data/Cora", name="Cora", transform=transforms_undirected),
        "Citeseer": Planetoid(root="data/Citeseer", name="Citeseer", transform=transforms_undirected),
        "PubMed": Planetoid(root="data/PubMed", name="PubMed", transform=transforms_undirected),
        "Chameleon": WikipediaNetwork(root="data/WikipediaNetwork", name="chameleon", transform=transforms_undirected),
        "Actor": Actor(root="data/Actor", transform=transforms_with_norm),
        "Squirrel": WikipediaNetwork(root="data/WikipediaNetwork", name="squirrel", transform=transforms_undirected),
        "Texas": WebKB(root="data/WebKB", name="Texas", transform=transforms_undirected),
        "Cornell": WebKB(root="data/WebKB", name="Cornell", transform=transforms_undirected),
        "Wisconsin": WebKB(root="data/WebKB", name="Wisconsin", transform=transforms_undirected)
    }

    return datasets

def load_dataset(dataset_name):
    """
    Load a specific dataset with appropriate transformations.

    This function loads only the requested dataset to avoid unnecessary downloads
    and save time when working with a single dataset.

    Args:
        dataset_name (str): Name of the dataset to load

    Returns:
        Dataset: Loaded PyTorch Geometric dataset

    Raises:
        ValueError: If the dataset name is not supported
    """
    # Define transforms
    transforms_undirected = T.ToUndirected()
    transforms_with_norm = T.Compose([
        T.ToUndirected(),
        T.NormalizeFeatures()
    ])

    # Load only the requested dataset
    if dataset_name == "Cora":
        return Planetoid(root="data/Cora", name="Cora", transform=transforms_undirected)
    elif dataset_name == "Citeseer":
        return Planetoid(root="data/Citeseer", name="Citeseer", transform=transforms_undirected)
    elif dataset_name == "PubMed":
        return Planetoid(root="data/PubMed", name="PubMed", transform=transforms_undirected)
    elif dataset_name == "Chameleon":
        return WikipediaNetwork(root="data/WikipediaNetwork", name="chameleon", transform=transforms_undirected)
    elif dataset_name == "Actor":
        return Actor(root="data/Actor", transform=transforms_with_norm)
    elif dataset_name == "Squirrel":
        return WikipediaNetwork(root="data/WikipediaNetwork", name="squirrel", transform=transforms_undirected)
    elif dataset_name == "Texas":
        return WebKB(root="data/WebKB", name="Texas", transform=transforms_undirected)
    elif dataset_name == "Cornell":
        return WebKB(root="data/WebKB", name="Cornell", transform=transforms_undirected)
    elif dataset_name == "Wisconsin":
        return WebKB(root="data/WebKB", name="Wisconsin", transform=transforms_undirected)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def apply_custom_splits(data, args, run, num_runs):
    """
    Apply custom splits to the data if use_splits is provided.

    Args:
        data (Data): PyTorch Geometric Data object
        args (Namespace): Arguments containing use_splits path
        run (int): Current run index
        num_runs (int): Total number of runs

    Returns:
        Data: Data object with updated masks
    """
    if args.use_splits is not None:
        split_idx = get_split_index(run, num_runs)
        print(f"Using split {split_idx} for run {run + 1}")

        try:
            train_mask, val_mask, test_mask = load_split_masks(args.use_splits, args.dataset, split_idx)
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
            print(f"Loaded custom split from {args.use_splits}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Falling back to default dataset splits")
    else:
        # Use default dataset splits with custom processing for certain datasets
        if args.dataset in ['Chameleon', 'Actor', 'Squirrel', 'Texas', 'Cornell', 'Wisconsin']:
            # These datasets have multiple masks, we use only the first one
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.test_mask = data.test_mask[:, 0]

            # Replace with 0.6/0.2/0.2 split with random seed 42
            num_nodes = data.x.shape[0]
            indices = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(42))

            train_size = int(0.6 * num_nodes)
            val_size = int(0.2 * num_nodes)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            # Create new masks
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            data.train_mask[train_indices] = True
            data.val_mask[val_indices] = True
            data.test_mask[test_indices] = True

    return data
