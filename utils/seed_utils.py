"""
Seed Utilities

This module provides functions for setting random seeds across different libraries.
"""

import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set random seed across all libraries.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
