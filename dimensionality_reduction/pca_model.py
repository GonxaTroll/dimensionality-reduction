"""
High-level PCA model wrapper.

Encapsulates original data, preprocessed data, and a fitted PCA model, and
exposes convenience methods for plotting and diagnostics using the existing
utility functions in this package.
"""
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA

from .preprocessing.scaler import normalize, standardize
from .visualization import (
    plot_scores as _plot_scores,
    plot_loadings as _plot_loadings,
    plot_loadings_2d as _plot_loadings_2d,
    plot_variance_explained as _plot_variance_explained,
    plot_eigenvalues as _plot_eigenvalues,
    plot_contribs as _plot_contribs,
    compute_t2_hotelling as _compute_t2_hotelling,
    compute_spe as _compute_spe,
    compute_spe_jackson_mudholkar as _compute_spe_jm,
    compute_sce as _compute_sce,
    get_last_best_eigenvalue as _kaiser,
    get_outlier_indexes as _get_outlier_indexes,
)


class PCAModel:
    """Wrapper that keeps data, preprocessing, and PCA results together."""

    def __init__(
        self,
        original_data: np.ndarray,
        preprocessed_data: np.ndarray,
        pca: PCA,
        *,
        preprocessing: str = "none",
        feature_names: Optional[Sequence[str]] = None,
        scale_params: Optional[dict] = None,
        scores: Optional[np.ndarray] = None,
    ) -> None:
        self.original_data = np.asarray(original_data)
        self.preprocessed_data = np.asarray(preprocessed_data)
        self.pca = pca
        self.preprocessing = preprocessing
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.scale_params = scale_params or {}

        self.scores_ = scores if scores is not None else self.pca.transform(self.preprocessed_data)
        self.loadings_ = self.pca.components_
        self.explained_variance_ = self.pca.explained_variance_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        data: np.ndarray,
        *,
        n_components: Optional[Union[int, float]] = None,
        preprocessing: str = "standardize",
        feature_names: Optional[Sequence[str]] = None,
        **pca_kwargs,
    ) -> "PCAModel":
        """Preprocess data, fit PCA, and return a ready-to-use wrapper."""
        preprocessing = (preprocessing or "none").lower()
        scale_params = {}

        if preprocessing == "standardize":
            preprocessed, mean_, std_ = standardize(data)
            scale_params = {"mean": mean_, "std": std_}
        elif preprocessing == "normalize":
            preprocessed, min_, max_ = normalize(data)
            scale_params = {"min": min_, "max": max_}
        elif preprocessing in ("none", "raw"):
            preprocessed = np.asarray(data)
        else:
            raise ValueError("preprocessing must be one of: standardize, normalize, none")

        pca = PCA(n_components=n_components, **pca_kwargs)
        scores = pca.fit_transform(preprocessed)

        return cls(
            original_data=data,
            preprocessed_data=preprocessed,
            pca=pca,
            preprocessing=preprocessing,
            feature_names=feature_names,
            scale_params=scale_params,
            scores=scores,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return self.preprocessed_data.shape[0]

    @property
    def n_features(self) -> int:
        return self.preprocessed_data.shape[1]

    @property
    def n_components(self) -> int:
        return self.pca.n_components_ if hasattr(self.pca, "n_components_") else self.pca.n_components

    # ------------------------------------------------------------------
    # Transformation helpers
    # ------------------------------------------------------------------
    def _apply_preprocessing(self, data: np.ndarray) -> np.ndarray:
        """Apply the stored preprocessing to new data."""
        arr = np.asarray(data)
        if self.preprocessing == "standardize":
            mean_ = self.scale_params.get("mean")
            std_ = self.scale_params.get("std")
            if mean_ is None or std_ is None:
                mean_ = np.mean(arr, axis=0)
                std_ = np.std(arr, axis=0)
            std_ = np.where(std_ == 0, 1, std_)
            return (arr - mean_) / std_
        if self.preprocessing == "normalize":
            min_ = self.scale_params.get("min")
            max_ = self.scale_params.get("max")
            if min_ is None or max_ is None:
                min_ = np.min(arr, axis=0)
                max_ = np.max(arr, axis=0)
            range_ = np.where((max_ - min_) == 0, 1, (max_ - min_))
            return (arr - min_) / range_
        return arr

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Preprocess with stored parameters and transform with the fitted PCA."""
        processed = self._apply_preprocessing(data)
        return self.pca.transform(processed)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_scores(
        self,
        pc1: int = 1,
        pc2: int = 2,
        *,
        score_color: Optional[np.ndarray] = None,
        annotate: Optional[Sequence[int]] = None,
        figsize: tuple = (10, 8),
    ):
        return _plot_scores(self.scores_, pc1=pc1, pc2=pc2, score_color=score_color, annotate=annotate, figsize=figsize)

    def plot_loadings(
        self,
        component_idx: Union[int, Sequence[int]] = 0,
        *,
        top_n: Optional[int] = None,
        figsize: tuple = (10, 6),
    ):
        loadings = self.loadings_.T
        return _plot_loadings(loadings, feature_names=self.feature_names, component_idx=component_idx, top_n=top_n, figsize=figsize)

    def plot_loadings_2d(
        self,
        pc1: int = 1,
        pc2: int = 2,
        *,
        draw_labels: bool = False,
        color_by: Optional[str] = "contrib",
        figsize: tuple = (10, 8),
    ):
        variables = self.feature_names if self.feature_names is not None else [f"Var {i}" for i in range(self.n_features)]
        return _plot_loadings_2d(self.loadings_, variables, pc1=pc1, pc2=pc2, draw_labels=draw_labels, color_by=color_by, figsize=figsize)

    def plot_variance_explained(self, *, figsize: tuple = (10, 6)):
        return _plot_variance_explained(self.explained_variance_ratio_, figsize=figsize, num_variables=self.n_features)

    def plot_eigenvalues(self, *, first_n: Union[int, str] = "all", figsize: tuple = (10, 6)):
        return _plot_eigenvalues(self.explained_variance_, first_n=first_n, figsize=figsize)

    def plot_contribs(
        self,
        indivs: Union[int, Sequence[int]],
        pc: int,
        *,
        variable_names: Optional[Sequence[str]] = None,
        simca_style: bool = True,
        use_preprocessed: bool = True,
        figsize: tuple = (12, 6),
    ):
        data = self.preprocessed_data if use_preprocessed else self.original_data
        variable_names = list(variable_names) if variable_names is not None else None
        return _plot_contribs(data, self.loadings_, indivs=indivs, pc=pc, variable_names=variable_names, simca_style=simca_style, figsize=figsize)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def kaiser_components(self) -> np.ndarray:
        return _kaiser(self.explained_variance_)

    def sce(self, selected_components: Optional[int] = None) -> np.ndarray:
        k = selected_components or self.n_components
        return _compute_sce(self.scores_, k)

    def t2(self, selected_components: Optional[int] = None, *, plot: bool = False, figsize: tuple = (10, 6)) -> Tuple[np.ndarray, float, float]:
        k = selected_components or self.n_components
        return _compute_t2_hotelling(self.scores_, self.explained_variance_, k, plot=plot, figsize=figsize)

    def spe(self, selected_components: Optional[int] = None, *, plot: bool = False, figsize: tuple = (10, 6)) -> Tuple[np.ndarray, float, float]:
        k = selected_components or self.n_components
        return _compute_spe(self.preprocessed_data, self.scores_, self.loadings_, k, plot=plot, figsize=figsize)

    def spe_jackson_mudholkar(
        self,
        selected_components: Optional[int] = None,
        *,
        plot: bool = False,
        figsize: tuple = (10, 6),
    ) -> Tuple[np.ndarray, float, float]:
        k = selected_components or self.n_components
        return _compute_spe_jm(self.preprocessed_data, self.scores_, self.loadings_, self.explained_variance_, k, plot=plot, figsize=figsize)

    def outliers_from_t2(self, selected_components: Optional[int] = None, *, alpha: float = 0.99) -> Tuple[np.ndarray, float]:
        t2, f95, f99 = self.t2(selected_components, plot=False)
        threshold = f99 if alpha >= 0.99 else f95
        return _get_outlier_indexes(t2, threshold), threshold

    def outliers_from_spe(self, selected_components: Optional[int] = None, *, alpha: float = 0.99) -> Tuple[np.ndarray, float]:
        spe_values, c95, c99 = self.spe(selected_components, plot=False)
        threshold = c99 if alpha >= 0.99 else c95
        return _get_outlier_indexes(spe_values, threshold), threshold

    # ------------------------------------------------------------------
    # Reconstruction helper
    # ------------------------------------------------------------------
    def reconstruct(self, selected_components: Optional[int] = None) -> np.ndarray:
        """Reconstruct data using selected components (defaults to all)."""
        k = selected_components or self.n_components
        scores_subset = self.scores_[:, :k]
        loadings_subset = self.loadings_[:k, :]
        return np.matmul(scores_subset, loadings_subset)
