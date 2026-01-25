"""
Feature selection utilities for dimensionality reduction.
"""

import numpy as np

def select_features(data: np.ndarray, variance_threshold: float = 0.0) -> tuple:
    """
    Select features based on variance threshold.
    
    Parameters
    ----------
    data : np.ndarray
        Input data, shape (n_samples, n_features)
    variance_threshold : float
        Features with variance below this threshold will be removed.
        Default is 0.0 (remove constant features).
    
    Returns
    -------
    tuple
        (selected_data, feature_indices)
        - selected_data: Data with low-variance features removed
        - feature_indices: Indices of selected features
    """
    variances = np.var(data, axis=0)
    feature_indices = np.where(variances > variance_threshold)[0]
    selected_data = data[:, feature_indices]

    return selected_data, feature_indices
