"""
Data scaling utilities for preprocessing.
"""

import numpy as np
from typing import Optional


def standardize(data: np.ndarray, mean: Optional[np.ndarray] = None, 
                std: Optional[np.ndarray] = None) -> tuple:
    """
    Standardize data to have zero mean and unit variance.
    
    Parameters
    ----------
    data : np.ndarray
        Input data to standardize, shape (n_samples, n_features)
    mean : Optional[np.ndarray]
        Pre-computed mean values. If None, computed from data.
    std : Optional[np.ndarray]
        Pre-computed standard deviation values. If None, computed from data.
    
    Returns
    -------
    tuple
        (standardized_data, mean, std)
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    standardized = (data - mean) / std
    return standardized, mean, std


def normalize(data: np.ndarray, min_val: Optional[np.ndarray] = None,
              max_val: Optional[np.ndarray] = None) -> tuple:
    """
    Normalize data to [0, 1] range.
    
    Parameters
    ----------
    data : np.ndarray
        Input data to normalize, shape (n_samples, n_features)
    min_val : Optional[np.ndarray]
        Pre-computed minimum values. If None, computed from data.
    max_val : Optional[np.ndarray]
        Pre-computed maximum values. If None, computed from data.
    
    Returns
    -------
    tuple
        (normalized_data, min_val, max_val)
    """
    if min_val is None:
        min_val = np.min(data, axis=0)
    if max_val is None:
        max_val = np.max(data, axis=0)
    
    # Avoid division by zero
    range_val = max_val - min_val
    range_val = np.where(range_val == 0, 1, range_val)
    
    normalized = (data - min_val) / range_val
    return normalized, min_val, max_val
