"""
Preprocessing utilities for dimensionality reduction.

This module provides functions for preparing data before applying
dimensionality reduction techniques.
"""

from .scaler import Scaler
from .feature_selection import select_features

__all__ = ["Scaler", "select_features"]
