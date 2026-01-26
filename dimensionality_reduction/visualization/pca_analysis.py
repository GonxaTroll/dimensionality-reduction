"""
PCA analysis and diagnostics utilities.

This module provides functions for analyzing PCA results, computing diagnostics,
and identifying outliers using various statistical methods.
"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def get_last_best_eigenvalue(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Extract eigenvalues greater than 1 (Kaiser criterion).
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues from PCA
    
    Returns
    -------
    np.ndarray
        Subset of eigenvalues > 1
    """
    pos = np.where(eigenvalues > 1)[0]
    subset = eigenvalues[pos]
    print(f"First {len(pos)} components have eigenvalues > 1")
    return subset


def compute_sce(scores: np.ndarray, selected_components: int) -> np.ndarray:
    """
    Compute Squared prediction error (SCE) - variability of each observation 
    explained by selected principal components.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix, shape (n_samples, n_components)
    selected_components : int
        Number of components to consider

    Returns
    -------
    np.ndarray
        SCE values for each observation
    """
    return np.sum(scores[:, :selected_components] ** 2, axis=1)


def compute_t2_hotelling(scores: np.ndarray, eigenvalues: np.ndarray,
                         selected_components: int,
                         plot: bool = True, figsize: tuple = (10, 6)) -> Tuple:
    """
    Compute Hotelling's T² statistic for outlier detection in score space.
    
    The T² statistic measures the distance of an observation from the center
    of the model in the retained principal component space.
    
    Parameters
    ----------
    scores : np.ndarray
        Score matrix, shape (n_samples, n_components)
    eigenvalues : np.ndarray
        Array of eigenvalues
    selected_components : int
        Number of principal components to use
    plot : bool
        Whether to create a plot. Default is True.
    figsize : tuple
        Figure size if plot=True. Default is (10, 6).
    
    Returns
    -------
    tuple
        (t2_values, threshold_95, threshold_99) - T² values and confidence limits
    """
    scores_subset = scores[:, :selected_components]
    eigenvalues_subset = eigenvalues[:selected_components]

    t2 = np.sum(scores_subset ** 2 / eigenvalues_subset, axis=1)
    ndata = scores.shape[0]

    # F-distribution based thresholds
    frac = selected_components * (ndata ** 2 - 1) / (ndata * (ndata - selected_components))
    f95 = frac * stats.f.ppf(0.95, selected_components, ndata - selected_components)
    f99 = frac * stats.f.ppf(0.99, selected_components, ndata - selected_components)

    if plot:
        _, ax = plt.subplots(figsize=figsize)
        ax.plot(t2, 'o-', markersize=4)
        ax.axhline(y = f95, color = "orange",
                 linestyle = "dashed", label = "95% confidence limit")
        ax.axhline(y=f99, color = "red",
                 linestyle = "dashed", label="99% confidence limit")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Hotelling T² statistic")
        ax.set_title("Hotelling T² Plot for Outlier Detection")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return t2, f95, f99


def compute_spe(original_data: np.ndarray, scores: np.ndarray,
                loadings: np.ndarray, selected_components: int,
                plot: bool = True, figsize: tuple = (10, 6)) -> Tuple:
    """
    Compute Squared Prediction Error (SPE), also known as Q-statistic.

    SPE measures the lack of fit of the model to each observation, indicating
    how well the observation is described by the selected components.
    Uses chi-squared approximation for thresholds.

    Parameters
    ----------
    original_data : np.ndarray
        Original data matrix, shape (n_samples, n_features)
    scores : np.ndarray
        Score matrix, shape (n_samples, n_components)
    loadings : np.ndarray
        Loadings matrix, shape (n_components, n_features)
    selected_components : int
        Number of principal components used
    plot : bool
        Whether to create a plot. Default is True.
    figsize : tuple
        Figure size if plot=True. Default is (10, 6).

    Returns
    -------
    tuple
        (spe_values, threshold_95, threshold_99) - SPE values and confidence limits
    """
    # ndata = scores.shape[0]
    scores_subset = scores[:, :selected_components]
    loadings_subset = loadings[:selected_components, :]

    # Reconstruct data and compute residuals
    reconstructed_data = np.matmul(scores_subset, loadings_subset)
    residuals = original_data - reconstructed_data
    spe_values = np.sum(residuals ** 2, axis=1)

    # Chi-squared approximation
    g = np.var(spe_values) / (2 * np.mean(spe_values))
    h = 2 * np.mean(spe_values) ** 2 / np.var(spe_values)
    chi2_95 = g * stats.chi2.ppf(0.95, df=h)
    chi2_99 = g * stats.chi2.ppf(0.99, df=h)

    if plot:
        _, ax = plt.subplots(figsize=figsize)
        ax.plot(spe_values, 'o-', markersize=4)
        ax.axhline(y = chi2_95, color = "orange",
                 linestyle = "dashed", label = "95% confidence limit")
        ax.axhline(y = chi2_99, color = "red",
                 linestyle = "dashed", label = "99% confidence limit")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("SPE (Q-statistic)")
        ax.set_title("SPE Plot for Outlier Detection")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return spe_values, chi2_95, chi2_99


def compute_spe_jackson_mudholkar(original_data: np.ndarray, scores: np.ndarray,
                                  loadings: np.ndarray, eigenvalues: np.ndarray,
                                  selected_components: int,
                                  plot: bool = True, figsize: tuple = (10, 6)) -> Tuple:
    """
    Compute SPE with Jackson-Mudholkar approximation for thresholds.

    This method uses the eigenvalues of the residual space for more accurate
    threshold estimation (Jackson & Mudholkar, 1979).

    Parameters
    ----------
    original_data : np.ndarray
        Original data matrix, shape (n_samples, n_features)
    scores : np.ndarray
        Score matrix, shape (n_samples, n_components)
    loadings : np.ndarray
        Loadings matrix, shape (n_components, n_features)
    eigenvalues : np.ndarray
        Array of all eigenvalues
    selected_components : int
        Number of principal components used
    plot : bool
        Whether to create a plot. Default is True.
    figsize : tuple
        Figure size if plot=True. Default is (10, 6).

    Returns
    -------
    tuple
        (spe_values, threshold_95, threshold_99) - SPE values and confidence limits

    References
    ----------
    Jackson, J. E., & Mudholkar, G. S. (1979). Control procedures for residuals 
    associated with principal component analysis. Technometrics, 21(3), 341-349.
    """
    # ndata = scores.shape[0]
    scores_subset = scores[:, :selected_components]
    loadings_subset = loadings[:selected_components, :]

    # Reconstruct data and compute residuals
    reconstructed_data = np.matmul(scores_subset, loadings_subset)
    residuals = original_data - reconstructed_data
    spe_values = np.sum(residuals ** 2, axis=1)

    # Jackson-Mudholkar approximation using residual eigenvalues
    residual_eigenvalues = eigenvalues[selected_components:]
    theta1 = np.sum(residual_eigenvalues)
    theta2 = np.sum(residual_eigenvalues ** 2)
    theta3 = np.sum(residual_eigenvalues ** 3)

    h0 = 1 - 2 * theta1 * theta3 / (3 * theta2 ** 2)

    z095 = stats.norm.ppf(0.95)
    z099 = stats.norm.ppf(0.99)

    ca_95 = (z095 * np.sqrt(2 * theta2 * h0 ** 2) / theta1 +
             theta2 * h0 * (h0 - 1) / theta1 ** 2 + 1)
    ca_99 = (z099 * np.sqrt(2 * theta2 * h0 ** 2) / theta1 +
             theta2 * h0 * (h0 - 1) / theta1 ** 2 + 1)

    q095 = theta1 * (ca_95 ** (1 / h0))
    q099 = theta1 * (ca_99 ** (1 / h0))

    if plot:
        _, ax = plt.subplots(figsize=figsize)
        ax.plot(spe_values, 'o-', markersize=4)
        ax.axhline(y = q095, color = "orange",
                 linestyle = "dashed", label = "95% confidence limit (J-M)")
        ax.axhline(y = q099, color = "red",
                 linestyle = "dashed", label = "99% confidence limit (J-M)")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("SPE (Q-statistic)")
        ax.set_title("SPE Plot (Jackson-Mudholkar) for Outlier Detection")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return spe_values, q095, q099


def get_outlier_indexes(statistic_values: np.ndarray, threshold: float) -> np.ndarray:
    """
    Identify outliers based on a statistic and threshold.
    
    Parameters
    ----------
    statistic_values : np.ndarray
        Array of statistic values (e.g., T², SPE) for each observation
    threshold : float
        Threshold value for outlier detection
    
    Returns
    -------
    np.ndarray
        Indices of observations exceeding the threshold
    """
    outlier_indices = np.where(statistic_values > threshold)[0]
    print(f"Number of outliers detected: {len(outlier_indices)}")
    return outlier_indices
