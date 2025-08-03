"""
Logging Utilities for Training Session Management

This module provides functions for creating log directories, saving training records,
models, and evaluation results.
"""

import os
import json
import pickle
import torch
from datetime import datetime


def create_log_directory(base_log_dir, dataset_name, args):
    """
    Create a timestamped log directory and save run configuration.

    Args:
        base_log_dir (str): Base directory for all logs
        dataset_name (str): Name of the dataset
        args (Namespace): Arguments containing run configuration

    Returns:
        str: Path to the created log directory
    """
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    log_dir = os.path.join(base_log_dir, f"{dataset_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Save run configuration
    config_path = os.path.join(log_dir, "run_configs.json")
    config_dict = vars(args).copy()

    # Convert non-serializable objects to strings
    for key, value in config_dict.items():
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            config_dict[key] = str(value)

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    return log_dir


def save_epoch_records(log_dir, all_records, dataset_name):
    """
    Save all training records to a pickle file.

    Args:
        log_dir (str): Directory to save the records
        all_records (dict): Dictionary containing all training records
        dataset_name (str): Name of the dataset
    """
    records_path = os.path.join(log_dir, f"{dataset_name}_records.pkl")
    with open(records_path, 'wb') as f:
        pickle.dump(all_records, f)


def save_best_model(model, log_dir, dataset_name, run_idx, is_global_best=False, global_best_run=None):
    """
    Save the best model for a run and optionally the global best model.

    Args:
        model (nn.Module): PyTorch model to save
        log_dir (str): Directory to save the model
        dataset_name (str): Name of the dataset
        run_idx (int): Index of the current run
        is_global_best (bool): Whether this is the global best model across all runs
        global_best_run (int): Index of the global best run (if is_global_best is True)
    """
    # Save run-specific best model
    model_path = os.path.join(log_dir, f"{dataset_name}_best_{run_idx}.pth")
    torch.save(model.state_dict(), model_path)

    # Save global best model if this is the best across all runs
    if is_global_best:
        global_model_path = os.path.join(log_dir, f"{dataset_name}_global_best_model_{global_best_run}.pth")
        torch.save(model.state_dict(), global_model_path)


def save_global_best_info(log_dir, dataset_name, best_run_idx, best_metrics):
    """
    Save global best evaluation results.

    Args:
        log_dir (str): Directory to save the information
        dataset_name (str): Name of the dataset
        best_run_idx (int): Index of the best run
        best_metrics (tuple): Tuple containing best metrics
    """
    best_info = {
        'best_run_idx': best_run_idx,
        'best_metrics': best_metrics,
        'timestamp': datetime.now().isoformat()
    }

    info_path = os.path.join(log_dir, f"{dataset_name}_global_best_evaluation_results.pkl")
    with open(info_path, 'wb') as f:
        pickle.dump(best_info, f)
