"""
Dimensionality Reduction Utilities

A Python package containing utility functions for performing dimensionality 
reduction techniques over various datasets, including preprocessing and 
visualization tools.
"""

__version__ = "0.1.0"

from . import preprocessing
from . import visualization
from .pca_model import PCAModel

__all__ = ["preprocessing", "visualization", "PCAModel", "__version__"]
