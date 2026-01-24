"""
Preprocessing utilities for dimensionality reduction.

This module provides functions for preparing data before applying
dimensionality reduction techniques.
"""

from .scaler import standardize, normalize
from .feature_selection import select_features

__all__ = ["standardize", "normalize", "select_features"]
