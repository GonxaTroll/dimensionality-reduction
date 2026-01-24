"""
PCA visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_loadings(loadings: np.ndarray, feature_names: Optional[List[str]] = None,
                  component_idx: int = 0, top_n: Optional[int] = None,
                  figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot loadings (feature contributions) for a principal component.
    
    Parameters
    ----------
    loadings : np.ndarray
        Component loadings matrix, shape (n_features, n_components)
    feature_names : Optional[List[str]]
        Names of features. If None, uses indices.
    component_idx : int
        Index of component to plot. Default is 0 (first component).
    top_n : Optional[int]
        If provided, only plot top N features by absolute loading value.
    figsize : tuple
        Figure size. Default is (10, 6).
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    component_loadings = loadings[:, component_idx]
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(component_loadings))]
    
    # Sort by absolute value if top_n is specified
    if top_n is not None:
        indices = np.argsort(np.abs(component_loadings))[::-1][:top_n]
        component_loadings = component_loadings[indices]
        feature_names = [feature_names[i] for i in indices]
    
    # Create bar plot
    colors = ['red' if x < 0 else 'blue' for x in component_loadings]
    ax.barh(range(len(component_loadings)), component_loadings, color=colors)
    ax.set_yticks(range(len(component_loadings)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Loading Value')
    ax.set_title(f'Feature Loadings for Component {component_idx + 1}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_components(transformed_data: np.ndarray, labels: Optional[np.ndarray] = None,
                    component_x: int = 0, component_y: int = 1,
                    figsize: tuple = (10, 8)) -> plt.Figure:
    """
    Plot data in the space of two principal components.
    
    Parameters
    ----------
    transformed_data : np.ndarray
        Transformed data, shape (n_samples, n_components)
    labels : Optional[np.ndarray]
        Labels for coloring points. If None, all points same color.
    component_x : int
        Component index for x-axis. Default is 0.
    component_y : int
        Component index for y-axis. Default is 1.
    figsize : tuple
        Figure size. Default is (10, 8).
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_data = transformed_data[:, component_x]
    y_data = transformed_data[:, component_y]
    
    if labels is not None:
        scatter = ax.scatter(x_data, y_data, c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Labels')
    else:
        ax.scatter(x_data, y_data, alpha=0.6)
    
    ax.set_xlabel(f'Component {component_x + 1}')
    ax.set_ylabel(f'Component {component_y + 1}')
    ax.set_title('Data in Principal Component Space')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_variance_explained(explained_variance_ratio: np.ndarray,
                            figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot explained variance ratio for each component.
    
    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Ratio of variance explained by each component
    figsize : tuple
        Figure size. Default is (10, 6).
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    n_components = len(explained_variance_ratio)
    components = range(1, n_components + 1)
    
    # Individual variance explained
    ax1.bar(components, explained_variance_ratio, alpha=0.7)
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Variance Explained Ratio')
    ax1.set_title('Variance Explained by Each Component')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance explained
    cumulative_variance = np.cumsum(explained_variance_ratio)
    ax2.plot(components, cumulative_variance, 'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Cumulative Variance Explained')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig
