"""
Training Utilities for Model Training

This module provides training loops for both node classification
and link prediction tasks, including model training, validation, and checkpoint saving.
"""

import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from utils.eval_utils import evaluate_node_classification, evaluate_link_prediction
from utils.loss_utils import link_prediction_loss, sample_neg_edges_from_mask
from utils.logging_utils import save_best_model


def train_node_classification(model, data, optimizer, scheduler, args, run_records, log_dir, run_idx):
    """
    Training loop for node classification task.

    Args:
        model (nn.Module): Model to train
        data (Data): PyTorch Geometric Data object
        optimizer (Optimizer): PyTorch optimizer
        scheduler (Scheduler): Learning rate scheduler
        args (Namespace): Training arguments
        run_records (dict): Dictionary to store training records
        log_dir (str): Directory to save logs and models
        run_idx (int): Index of the current run

    Returns:
        tuple: (best_test_f1, best_test_acc, best_test_auc, best_test_prc, run_records)
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    best_val_f1 = 0
    best_test_f1 = 0
    best_epoch = -1
    best_model_state = None

    best_val_acc = 0
    best_test_acc = 0

    # Lists to store metrics over epochs
    train_f1_list = []
    val_f1_list = []
    test_f1_list = []
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    train_auc_list = []
    val_auc_list = []
    test_auc_list = []
    train_prc_list = []
    val_prc_list = []
    test_prc_list = []

    loss_list = []
    curvature_logs = []

    for epoch in range(1, args.epochs + 1):

        epoch_start_time = time.time()

        model.train()
        optimizer.zero_grad()
        out = model(data)  # Raw logits
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        # Add regularization loss for ARGNN
        if args.model == 'argnn':
            reg_loss = model.compute_regularization_loss(data.edge_index)
            loss = loss + reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_f1, val_f1, test_f1, train_acc, val_acc, test_acc, train_auc, val_auc, test_auc, train_prc, val_prc, test_prc = evaluate_node_classification(model, data)

        # Record epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Log curvature changes for ARGNN
        if args.model == 'argnn':
            if epoch == 1:
                metrics = model.get_metrics()
                pre_curvature = metrics
                # Initialize curvature differences (all zeros for first epoch)
                metrics = {key: 0 for key in metrics.keys()}
            else:
                cur_metrics = model.get_metrics()
                metrics = {}
                for key in cur_metrics.keys():
                    if key in pre_curvature:
                        metrics[key] = np.sqrt(np.sum((cur_metrics[key] - pre_curvature[key])**2, axis=1).mean())
                    else:
                        metrics[key] = 0
                pre_curvature = cur_metrics
        else:
            metrics = model.get_metrics()

        epoch_metrics = {
            'epoch': epoch,
            'loss': loss.item(),
            'epoch_time': epoch_time,
            'train_f1': train_f1,
            'train_acc': train_acc,
            'train_auc': train_auc,
            'train_prc': train_prc,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_prc': val_prc,
            'test_f1': test_f1,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_prc': test_prc,
            'metrics': metrics
        }

        # Append to run_records
        run_records['epochs'].append(epoch_metrics)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_val_auc = val_auc
            best_val_prc = val_prc

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_test_acc = test_acc
            best_test_auc = test_auc
            best_test_prc = test_prc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

            # Update run records with best metrics
            run_records['best_epoch'] = epoch
            run_records['best_metrics'] = {
                'test_f1': test_f1,
                'test_accuracy': test_acc,
                'test_auc': test_auc,
                'test_prc': test_prc,
                'val_f1': val_f1,
                'val_accuracy': val_acc,
                'val_auc': val_auc,
                'val_prc': val_prc
            }

        # Store metrics
        train_f1_list.append(train_f1)
        val_f1_list.append(val_f1)
        test_f1_list.append(test_f1)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        loss_list.append(loss.item())
        curvature_logs.append(metrics)

        print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, '
              f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, '
              f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}, '
              f'Train PRC: {train_prc:.4f}, Val PRC: {val_prc:.4f}, Test PRC: {test_prc:.4f}, '
              f'Time: {epoch_time:.3f}s')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        save_best_model(model, log_dir, args.dataset, run_idx)

    print(f'Best Val F1: {best_val_f1:.4f}, Best Test F1: {best_test_f1:.4f}')
    print(f'Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}')
    print(f'Best Val AUC: {best_val_auc:.4f}, Best Test AUC: {best_test_auc:.4f}')
    print(f'Best Val PRC: {best_val_prc:.4f}, Best Test PRC: {best_test_prc:.4f}')

    print('\nLearned metrics:')
    final_metrics = model.get_metrics()
    for key, value in final_metrics.items():
        print(f'{key}: {value}')
    for key, value in run_records['epochs'][1]['metrics'].items():
        print(f'Start diff G {key}: {value}')
    for key, value in run_records['epochs'][-1]['metrics'].items():
        print(f'End diff G {key}: {value}')

    return best_test_f1, best_test_acc, best_test_auc, best_test_prc, run_records


def train_link_prediction(model, data, optimizer, scheduler, args, run_records, log_dir, run_idx):
    """
    Training loop for link prediction task.

    Args:
        model (nn.Module): Model to train
        data (Data): PyTorch Geometric Data object
        optimizer (Optimizer): PyTorch optimizer
        scheduler (Scheduler): Learning rate scheduler
        args (Namespace): Training arguments
        run_records (dict): Dictionary to store training records
        log_dir (str): Directory to save logs and models
        run_idx (int): Index of the current run

    Returns:
        tuple: ((best_test_auc, best_test_ap, best_test_f1, best_test_acc), run_records)
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    best_val_auc = 0
    best_val_ap = 0
    best_val_f1 = 0
    best_val_acc = 0
    best_test_auc = 0
    best_test_ap = 0
    best_test_f1 = 0
    best_test_acc = 0

    # Lists to store metrics over epochs
    train_auc_list = []
    train_ap_list = []
    train_f1_list = []
    train_acc_list = []
    val_auc_list = []
    val_ap_list = []
    val_f1_list = []
    val_acc_list = []
    test_auc_list = []
    test_ap_list = []
    test_f1_list = []
    test_acc_list = []
    loss_list = []
    curvature_logs = []

    train_neg_edge_index = sample_neg_edges_from_mask(data.train_neg_adj_mask, num_neg_edges=data.train_pos_edge_index.size(1))

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        model.train()
        optimizer.zero_grad()

        # Generate node embeddings using training edges
        if args.model == 'argnn':
            z = model.encode(data.x, data.train_pos_edge_index, kappa=data.kappa)
        else:
            # Baseline models (GCN, GAT, SAGE) don't use kappa
            z = model.encode(data.x, data.train_pos_edge_index)

        # Compute loss using positive and negative edges
        loss = link_prediction_loss(model, z, data.train_pos_edge_index, train_neg_edge_index)

        # Add regularization loss for ARGNN
        if args.model == 'argnn':
            reg_loss = model.compute_regularization_loss(data.train_pos_edge_index)
            loss = loss + reg_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Record epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Evaluate on all sets
        train_auc, train_ap, train_f1, train_acc, val_auc, val_ap, val_f1, val_acc, test_auc, test_ap, test_f1, test_acc = evaluate_link_prediction(args, model, data)

        # Update best metrics and save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_ap = val_ap
            best_val_f1 = val_f1
            best_val_acc = val_acc
            # Save best model for this run
            save_best_model(model, log_dir, args.dataset, run_idx)

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_test_ap = test_ap
            best_test_f1 = test_f1
            best_test_acc = test_acc
            run_records['best_epoch'] = epoch
            run_records['best_metrics'] = {
                'test_auc': best_test_auc,
                'test_ap': best_test_ap,
                'test_f1': best_test_f1,
                'test_acc': best_test_acc,
                'val_auc': best_val_auc,
                'val_ap': best_val_ap,
                'val_f1': best_val_f1,
                'val_acc': best_val_acc
            }

        # Store metrics
        train_auc_list.append(train_auc)
        train_ap_list.append(train_ap)
        val_auc_list.append(val_auc)
        val_ap_list.append(val_ap)
        test_auc_list.append(test_auc)
        test_ap_list.append(test_ap)
        loss_list.append(loss.item())

        # Log curvature changes for ARGNN
        if args.model == 'argnn':
            if epoch == 1:
                metrics = model.get_metrics()
                pre_curvature = metrics
                # Initialize curvature differences (all zeros for first epoch)
                metrics = {key: 0 for key in metrics.keys()}
            else:
                cur_metrics = model.get_metrics()
                metrics = {}
                for key in cur_metrics.keys():
                    if key in pre_curvature:
                        metrics[key] = np.sqrt(np.sum((cur_metrics[key] - pre_curvature[key])**2, axis=1).mean())
                    else:
                        metrics[key] = 0
                pre_curvature = cur_metrics
        else:
            # Get metrics for other models
            metrics = model.get_metrics()

        curvature_logs.append(metrics)

        # Add epoch record to run_records
        epoch_record = {
            'epoch': epoch,
            'train_auc': train_auc,
            'train_ap': train_ap,
            'train_f1': train_f1,
            'train_acc': train_acc,
            'val_auc': val_auc,
            'val_ap': val_ap,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'test_auc': test_auc,
            'test_ap': test_ap,
            'test_f1': test_f1,
            'test_acc': test_acc,
            'loss': loss.item(),
            'epoch_time': epoch_time,
            'metrics': metrics
        }
        run_records['epochs'].append(epoch_record)

        # Print detailed metrics with AUROC first, then AUPRC, F1, ACC
        print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, '
              f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}, '
              f'Train PRC: {train_ap:.4f}, Val PRC: {val_ap:.4f}, Test PRC: {test_ap:.4f}, '
              f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    print(f'Best Val AUC: {best_val_auc:.4f}, Best Test AUC: {best_test_auc:.4f}')
    print(f'Best Val PRC: {best_val_ap:.4f}, Best Test PRC: {best_test_ap:.4f}')
    print(f'Best Val F1: {best_val_f1:.4f}, Best Test F1: {best_test_f1:.4f}')
    print(f'Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}')

    print('\nLearned metrics:')
    final_metrics = model.get_metrics()
    for key, value in final_metrics.items():
        print(f'{key}: {value}')
    for key, value in run_records['epochs'][1]['metrics'].items():
        print(f'Start diff G {key}: {value}')
    for key, value in run_records['epochs'][-1]['metrics'].items():
        print(f'End diff G {key}: {value}')

    return (best_test_auc, best_test_ap, best_test_f1, best_test_acc), run_records
