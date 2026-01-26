"""
PCA visualization utilities.
"""
from typing import Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    component_idx : List[int]
        Index of component to plot. Default is 0 (first component). If a list is provided,
        plots all specified components. Maximum up to 9 components.
    top_n : Optional[int]
        If provided, only plot top N features by absolute loading value.
    figsize : tuple
        Figure size. Default is (10, 6).
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """

    if isinstance(component_idx, int):
        component_idx = [component_idx]

    # arrange the plots in a grid if multiple components are provided
    if len(component_idx) > 1:
        n_plots = len(component_idx)
        n_cols = min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
    else:
        fig, ax = plt.subplots(figsize = figsize)
        axes = [ax]
    for i, comp_idx in enumerate(component_idx):
        component_loadings = loadings[:, comp_idx]
        ax = axes[i]

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
        ax.set_title(f'Feature Loadings for Component {comp_idx + 1}')
        ax.axvline(x = 0, color = 'black', linestyle = '-', linewidth = 0.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
    return fig


def plot_loadings_2d(loadings: np.ndarray, variables: List[str],
                     pc1: int = 1, pc2: int = 2, draw_labels: bool = False,
                     color_by: Optional[str] = "contrib",
                     figsize: tuple = (10, 8)) -> plt.Figure:
    """
    Plot 2D loadings showing variable contributions to two principal components.
    
    Parameters
    ----------
    loadings : np.ndarray
        Loadings matrix, shape (n_components, n_features)
    variables : List[str]
        List of variable/feature names
    pc1 : int
        First principal component to plot (1-indexed). Default is 1.
    pc2 : int
        Second principal component to plot (1-indexed). Default is 2.
    draw_labels : bool
        Whether to draw variable labels on the plot. Default is False.
    color_by : Optional[str]
        Color mapping method: None, "cos2" (quality of representation), 
        or "contrib" (contribution). Default is "contrib".
    figsize : tuple
        Figure size. Default is (10, 8).
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    n = loadings.shape[1]

    if color_by is None:
        ax.scatter(loadings[pc1 - 1, :], loadings[pc2 - 1, :])
    elif color_by in ["cos2", "contrib"]:
        cos2 = loadings ** 2
        cos2_sums = np.sum(cos2[[pc1 - 1, pc2 - 1], :], axis=0)

        if color_by == "cos2":
            scatter = ax.scatter(loadings[pc1 - 1, :], loadings[pc2 - 1, :],
                               c=cos2_sums, cmap='viridis')
        else:  # contrib
            contrib = (cos2.T / np.sum(cos2, axis=1)).T
            contrib_sum = np.sum(contrib[[pc1 - 1, pc2 - 1], :], axis=0)
            scatter = ax.scatter(loadings[pc1 - 1, :], loadings[pc2 - 1, :],
                               c=contrib_sum, cmap='viridis')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_by.capitalize())
        scatter.set_clim(0, 1)

    for i in range(n):
        v1, v2 = loadings[pc1 - 1, i], loadings[pc2 - 1, i]
        ax.arrow(0, 0, v1, v2, alpha = 0.5, head_width = 0.02, head_length = 0.02)
        if draw_labels:
            ax.text(v1 * 1.05, v2 * 1.05, variables[i], color = 'g', ha = 'center', va = 'center')

    ax.set_xlabel(f"PC {pc1}")
    ax.set_ylabel(f"PC {pc2}")
    ax.set_title("Loadings plot")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_scores(scores: np.ndarray, pc1: int = 1, pc2: int = 2,
                score_color: Optional[np.ndarray] = None,
                annotate: List[int] = None, figsize: tuple = (10, 8)) -> plt.Figure:
    """
    Plot PCA scores (transformed data) for two principal components.
    
    Parameters
    ----------
    scores : np.ndarray
        Score matrix, shape (n_samples, n_components)
    pc1 : int
        First principal component to plot (1-indexed). Default is 1.
    pc2 : int
        Second principal component to plot (1-indexed). Default is 2.
    score_color : Optional[np.ndarray]
        Array for coloring points. If None, all points are same color.
    annotate : List[int]
        List of sample indices to annotate on the plot.
    figsize : tuple
        Figure size. Default is (10, 8).
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    if score_color is not None:
        sns.scatterplot(x = scores[:, pc1 - 1], y = scores[:, pc2 - 1],
                       hue = score_color, ax = ax)
    else:
        ax.scatter(scores[:, pc1 - 1], scores[:, pc2 - 1])
    if annotate is not None:
        for index in annotate:
            ax.annotate(text = str(index),
                    xy = (scores[index, pc1 - 1], scores[index, pc2 - 1]))
    ax.set_xlabel(f"PC {pc1}")
    ax.set_ylabel(f"PC {pc2}")
    ax.set_title("Score plot")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_variance_explained(explained_variance_ratio: np.ndarray,
                            figsize: tuple = (10, 6),
                            num_variables: int = None) -> plt.Figure:
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize)

    n_components = len(explained_variance_ratio)
    components = range(1, n_components + 1)

    # Individual variance explained
    ax1.bar(components, explained_variance_ratio, alpha=0.7)
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Variance Explained Ratio')
    ax1.set_title('Variance Explained by Each Component')
    ax1.grid(True, alpha=0.3)

    if num_variables is not None:
        expected_var = 1 / n_components
        selected_comps = len(np.where(explained_variance_ratio > expected_var)[0])
        print(f"Selected components (above expected variance): {selected_comps}")
        ax1.axhline(y = expected_var, color="red", linestyle = "dashed",
                    label = f"Expected Variance: 1 / {n_components}")
        ax1.legend()

    # Cumulative variance explained
    cumulative_variance = np.cumsum(explained_variance_ratio)
    ax2.plot(components, cumulative_variance, 'bo-', linewidth = 2, markersize = 6)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Cumulative Variance Explained')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    return fig


def plot_eigenvalues(eigenvalues: np.ndarray, first_n: Union[int, str] = "all",
                     figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot eigenvalues (explained variance) for each principal component.
    Uses Kaiser criterion (eigenvalue > 1) to suggest component selection.
    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues for each component
    first_n : Union[int, str]
        Number of first components to plot, or "all" to plot all. Default is "all".
    figsize : tuple
        Figure size. Default is (10, 6).

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    eigenvalues_list = list(eigenvalues)
    if first_n != "all":
        eigenvalues_list = eigenvalues_list[:first_n]

    ax.bar(x = list(range(1, len(eigenvalues_list) + 1)), height = eigenvalues_list)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Eigenvalue (explained variance)")
    ax.axhline(y = 1, color="red", linestyle = "dashed", label = "Kaiser criterion (λ=1)")
    ax.set_title("Explained variance for each PC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_contribs(data: np.ndarray, loadings: np.ndarray,
                  indivs: Union[int, List[int]], pc: int,
                  variable_names: Optional[List[str]] = None,
                  simca_style: bool = True, figsize: tuple = (12, 6)) -> tuple:
    """
    Plot variable contributions for specific observations to a principal component.
    
    Parameters
    ----------
    data : np.ndarray
        Original data matrix, shape (n_samples, n_features)
    loadings : np.ndarray
        Loadings matrix, shape (n_components, n_features)
    indivs : Union[int, List[int]]
        Index or list of indices of observations to analyze
    pc : int
        Principal component to analyze (1-indexed)
    variable_names : Optional[List[str]]
        List of variable names. If None, uses indices.
    simca_style : bool
        Whether to use SIMCA-style contribution calculation. Default is True.
    figsize : tuple
        Figure size. Default is (12, 6).
    
    Returns
    -------
    tuple
        (plt.Figure, np.ndarray) - Figure object and contributions array
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(indivs, int) or isinstance(indivs, np.integer):
        indivs = [indivs]

    subset = data[indivs, :]
    n_features = data.shape[1]

    if variable_names is None:
        variable_names = [f"Var {i}" for i in range(n_features)]

    means = np.mean(data, axis=0)
    contribs = []

    for i in range(n_features):
        if simca_style:
            cont = (subset[:, i] - means[i]) * np.abs(loadings[pc - 1, i])
        else:
            cont = subset[:, i] * loadings[pc - 1, i]
        contribs.append(cont)

    contribs = np.array(contribs).T
    cc = np.mean(contribs, axis=0)

    ax.bar(x=variable_names, height=cc)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Contribution")
    ax.set_title(f"Variable Contributions to PC {pc} for Selected Observations")
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return fig, contribs
