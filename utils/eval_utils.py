"""
Evaluation Utilities for Model Performance Assessment

This module provides evaluation functions for both node classification
and link prediction tasks, including metrics like F1, accuracy, AUC, and AUPRC.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from utils.loss_utils import sample_neg_edges_from_mask


def evaluate_node_classification(model, data):
    """
    Evaluate model performance on node classification task.

    Computes F1 score, accuracy, AUC-ROC, and AUPRC for train, validation, and test sets.

    Args:
        model (nn.Module): Model to evaluate
        data (Data): PyTorch Geometric Data object with node features and masks

    Returns:
        tuple: 12 metrics (train_f1, val_f1, test_f1, train_acc, val_acc, test_acc,
               train_auc, val_auc, test_auc, train_prc, val_prc, test_prc)
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=1).cpu().numpy()  # Convert predictions to numpy
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # Softmax probabilities for AUC and PRC
        labels = data.y.cpu().numpy()  # Convert true labels to numpy

        # Masks
        train_mask = data.train_mask.cpu()
        val_mask = data.val_mask.cpu()
        test_mask = data.test_mask.cpu()

        # F1 scores
        train_f1 = f1_score(labels[train_mask], preds[train_mask], average='weighted')
        val_f1 = f1_score(labels[val_mask], preds[val_mask], average='weighted')
        test_f1 = f1_score(labels[test_mask], preds[test_mask], average='weighted')

        # Accuracies
        train_acc = (preds[train_mask] == labels[train_mask]).mean()
        val_acc = (preds[val_mask] == labels[val_mask]).mean()
        test_acc = (preds[test_mask] == labels[test_mask]).mean()

        # AUC-ROC (one-vs-rest, weighted) with safe computation
        train_auc = safe_auc_score(labels[train_mask], probs[train_mask])
        val_auc = safe_auc_score(labels[val_mask], probs[val_mask])
        test_auc = safe_auc_score(labels[test_mask], probs[test_mask])

        # AUPRC (average precision score, weighted) with safe computation
        train_prc = safe_ap_score(labels[train_mask], probs[train_mask])
        val_prc = safe_ap_score(labels[val_mask], probs[val_mask])
        test_prc = safe_ap_score(labels[test_mask], probs[test_mask])

    return train_f1, val_f1, test_f1, train_acc, val_acc, test_acc, train_auc, val_auc, test_auc, train_prc, val_prc, test_prc


def safe_auc_score(y_true, y_score):
    """
    Safely compute AUC-ROC score, handling edge cases.

    Args:
        y_true (np.ndarray): True labels
        y_score (np.ndarray): Predicted probabilities

    Returns:
        float: AUC-ROC score, or 0.5 if computation fails
    """
    try:
        # Check if all classes are present in y_true
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            # If only one class present, return 0.5 (random performance)
            return 0.5
        elif len(unique_classes) == y_score.shape[1]:
            # All classes present, compute normally
            return roc_auc_score(y_true, y_score, multi_class='ovr', average='weighted')
        else:
            # Subset of classes present, use only relevant columns
            y_score_subset = y_score[:, unique_classes]
            return roc_auc_score(y_true, y_score_subset, multi_class='ovr', average='weighted')
    except ValueError:
        # Fallback to 0.5 if AUC computation fails
        return 0.5


def safe_ap_score(y_true, y_score):
    """
    Safely compute average precision score, handling edge cases.

    Args:
        y_true (np.ndarray): True labels
        y_score (np.ndarray): Predicted probabilities

    Returns:
        float: Average precision score, or 0.5 if computation fails
    """
    try:
        # Check if all classes are present in y_true
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            # If only one class present, return 0.5 (random performance)
            return 0.5
        elif len(unique_classes) == y_score.shape[1]:
            # All classes present, compute normally
            return average_precision_score(y_true, y_score, average='weighted')
        else:
            # Subset of classes present, use only relevant columns
            y_score_subset = y_score[:, unique_classes]
            return average_precision_score(y_true, y_score_subset, average='weighted')
    except ValueError:
        # Fallback to 0.5 if AP computation fails
        return 0.5


def evaluate_link_prediction(args, model, data):
    """
    Evaluate model performance on link prediction task.

    Computes AUC, average precision, F1 score, and accuracy for train, validation, and test sets.

    Args:
        args (Namespace): Arguments containing model configuration
        model (nn.Module): Model to evaluate
        data (Data): PyTorch Geometric Data object with edge information

    Returns:
        tuple: 12 metrics (train_auc, train_ap, train_f1, train_acc, val_auc, val_ap,
               val_f1, val_acc, test_auc, test_ap, test_f1, test_acc)
    """
    model.eval()
    with torch.no_grad():
        # Encode nodes using training edges
        if args.model == 'cusp':
            z = model.encode(data.x, data.train_pos_edge_index, kappa=data.kappa)
        elif args.model == 'argnn':
            z = model.encode(data.x, data.train_pos_edge_index, kappa=data.kappa)
        else:
            # Baseline models (GCN, GAT, SAGE) don't use kappa
            z = model.encode(data.x, data.train_pos_edge_index)

        # Evaluate on train set
        train_neg_edge_index = sample_neg_edges_from_mask(data.train_neg_adj_mask, num_neg_edges=data.train_pos_edge_index.size(1))
        train_auc, train_ap, train_f1, train_acc = model.test(z, data.train_pos_edge_index, train_neg_edge_index)

        # Evaluate on val set
        val_auc, val_ap, val_f1, val_acc = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)

        # Evaluate on test set
        test_auc, test_ap, test_f1, test_acc = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)

    return train_auc, train_ap, train_f1, train_acc, val_auc, val_ap, val_f1, val_acc, test_auc, test_ap, test_f1, test_acc
