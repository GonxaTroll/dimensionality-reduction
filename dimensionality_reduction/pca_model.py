"""
High-level PCA model wrapper.

Encapsulates original data (pandas DataFrame), preprocessing, and a fitted PCA model,
and exposes convenience methods for plotting and diagnostics using the existing
utility functions in this package.
"""
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("PCAModel expects a pandas DataFrame as input")

        self.original_df = data.copy()
        self.original_data = self.original_df.to_numpy()
        self.feature_names = list(self.original_df.columns)

        # Will be populated after fit
        self.preprocessing: str = "none"
        self.preprocessed_df: Optional[pd.DataFrame] = None
        self.preprocessed_data: Optional[np.ndarray] = None
        self.scaler_: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.pca: Optional[PCA] = None
        self.scores_: Optional[np.ndarray] = None
        self.loadings_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        *,
        n_components: Optional[Union[int, float]] = None,
        preprocessing: str = "standardize",
        **pca_kwargs,
    ) -> "PCAModel":
        """Preprocess stored data (as DataFrame), fit PCA, and cache results."""
        self.preprocessing = (preprocessing or "none").lower()
        self.preprocessed_df = self._preprocess_df(self.original_df, self.preprocessing)
        self.preprocessed_data = self.preprocessed_df.to_numpy()

        self.pca = PCA(n_components=n_components, **pca_kwargs)
        self.scores_ = self.pca.fit_transform(self.preprocessed_data)
        self.loadings_ = self.pca.components_
        self.explained_variance_ = self.pca.explained_variance_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return self.original_data.shape[0]

    @property
    def n_features(self) -> int:
        return self.original_data.shape[1]

    @property
    def n_components(self) -> int:
        self._require_fitted()
        return self.pca.n_components_  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _preprocess_df(self, df: pd.DataFrame, preprocessing: str) -> pd.DataFrame:
        """Preprocess a pandas DataFrame using sklearn scalers."""
        if preprocessing == "standardize":
            self.scaler_ = StandardScaler()
            preprocessed_np = self.scaler_.fit_transform(df)
            return pd.DataFrame(preprocessed_np, columns=df.columns, index=df.index)

        if preprocessing == "normalize":
            self.scaler_ = MinMaxScaler()
            preprocessed_np = self.scaler_.fit_transform(df)
            return pd.DataFrame(preprocessed_np, columns=df.columns, index=df.index)

        if preprocessing in ("none", "raw"):
            self.scaler_ = None
            return df.copy()

        raise ValueError("preprocessing must be one of: standardize, normalize, none")

    def _apply_preprocessing(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply stored preprocessing to data using the fitted scaler."""
        if self.preprocessing == "none":
            return np.asarray(data)
        
        if self.scaler_ is None:
            raise RuntimeError("Model not fitted: scaler missing")
        
        return self.scaler_.transform(data)

    def _require_fitted(self) -> None:
        if self.pca is None or self.scores_ is None or self.loadings_ is None:
            raise RuntimeError("PCA model not fitted. Call fit() first.")

    # ------------------------------------------------------------------
    # Transformation helpers
    # ------------------------------------------------------------------
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Preprocess with stored parameters and transform with the fitted PCA."""
        self._require_fitted()
        processed = self._apply_preprocessing(data)
        return self.pca.transform(processed)  # type: ignore[arg-type]

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
        self._require_fitted()
        return _plot_scores(self.scores_, pc1=pc1, pc2=pc2, score_color=score_color, annotate=annotate, figsize=figsize)

    def plot_loadings(
        self,
        component_idx: Union[int, Sequence[int]] = 0,
        *,
        top_n: Optional[int] = None,
        figsize: tuple = (10, 6),
    ):
        self._require_fitted()
        loadings = self.loadings_.T  # (n_features, n_components)
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
        self._require_fitted()
        variables = self.feature_names if self.feature_names is not None else [f"Var {i}" for i in range(self.n_features)]
        return _plot_loadings_2d(self.loadings_, variables, pc1=pc1, pc2=pc2, draw_labels=draw_labels, color_by=color_by, figsize=figsize)

    def plot_variance_explained(self, *, figsize: tuple = (10, 6)):
        self._require_fitted()
        return _plot_variance_explained(self.explained_variance_ratio_, figsize=figsize, num_variables=self.n_features)

    def plot_eigenvalues(self, *, first_n: Union[int, str] = "all", figsize: tuple = (10, 6)):
        self._require_fitted()
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
        self._require_fitted()
        data = self.preprocessed_data if use_preprocessed else self.original_data
        variable_names = list(variable_names) if variable_names is not None else None
        return _plot_contribs(data, self.loadings_, indivs=indivs, pc=pc, variable_names=variable_names, simca_style=simca_style, figsize=figsize)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def kaiser_components(self) -> np.ndarray:
        self._require_fitted()
        return _kaiser(self.explained_variance_)

    def sce(self, selected_components: Optional[int] = None) -> np.ndarray:
        self._require_fitted()
        k = selected_components or self.n_components
        return _compute_sce(self.scores_, k)

    def t2(self, selected_components: Optional[int] = None, *, plot: bool = False, figsize: tuple = (10, 6)) -> Tuple[np.ndarray, float, float]:
        self._require_fitted()
        k = selected_components or self.n_components
        return _compute_t2_hotelling(self.scores_, self.explained_variance_, k, plot=plot, figsize=figsize)

    def spe(self, selected_components: Optional[int] = None, *, plot: bool = False, figsize: tuple = (10, 6)) -> Tuple[np.ndarray, float, float]:
        self._require_fitted()
        k = selected_components or self.n_components
        return _compute_spe(self.preprocessed_data, self.scores_, self.loadings_, k, plot=plot, figsize=figsize)

    def spe_jackson_mudholkar(
        self,
        selected_components: Optional[int] = None,
        *,
        plot: bool = False,
        figsize: tuple = (10, 6),
    ) -> Tuple[np.ndarray, float, float]:
        self._require_fitted()
        k = selected_components or self.n_components
        return _compute_spe_jm(self.preprocessed_data, self.scores_, self.loadings_, self.explained_variance_, k, plot=plot, figsize=figsize)

    def outliers_from_t2(self, selected_components: Optional[int] = None, *, alpha: float = 0.99) -> Tuple[np.ndarray, float]:
        self._require_fitted()
        t2, f95, f99 = self.t2(selected_components, plot=False)
        threshold = f99 if alpha >= 0.99 else f95
        return _get_outlier_indexes(t2, threshold), threshold

    def outliers_from_spe(self, selected_components: Optional[int] = None, *, alpha: float = 0.99) -> Tuple[np.ndarray, float]:
        self._require_fitted()
        spe_values, c95, c99 = self.spe(selected_components, plot=False)
        threshold = c99 if alpha >= 0.99 else c95
        return _get_outlier_indexes(spe_values, threshold), threshold

    # ------------------------------------------------------------------
    # Reconstruction helper
    # ------------------------------------------------------------------
    def reconstruct(self, selected_components: Optional[int] = None) -> np.ndarray:
        """Reconstruct data using selected components (defaults to all)."""
        self._require_fitted()
        k = selected_components or self.n_components
        scores_subset = self.scores_[:, :k]
        loadings_subset = self.loadings_[:k, :]
        return np.matmul(scores_subset, loadings_subset)
