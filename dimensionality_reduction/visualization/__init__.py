"""
Visualization utilities for dimensionality reduction results.

This module provides functions for visualizing components and results
from dimensionality reduction techniques like PCA.
"""

from .pca_plots import (
    plot_loadings,
    plot_variance_explained,
    plot_eigenvalues,
    plot_scores,
    plot_loadings_2d,
    plot_contribs
)
from .pca_analysis import (
    get_last_best_eigenvalue,
    compute_sce,
    compute_t2_hotelling,
    compute_spe,
    compute_spe_jackson_mudholkar,
    get_outlier_indexes
)

__all__ = [
    # Plotting functions
    "plot_loadings",
    "plot_variance_explained",
    "plot_eigenvalues",
    "plot_scores",
    "plot_loadings_2d",
    "plot_contribs",
    # Analysis functions
    "get_last_best_eigenvalue",
    "compute_sce",
    "compute_t2_hotelling",
    "compute_spe",
    "compute_spe_jackson_mudholkar",
    "get_outlier_indexes"
]
