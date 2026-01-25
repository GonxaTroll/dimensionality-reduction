"""Tests for PCAModel wrapper."""

import numpy as np
import matplotlib.pyplot as plt

from dimensionality_reduction import PCAModel


def test_fit_and_transform_shapes():
    np.random.seed(0)
    data = np.random.randn(40, 6)

    model = PCAModel.fit(data, n_components=3, preprocessing="standardize")

    assert model.scores_.shape == (40, 3)
    assert model.loadings_.shape == (3, 6)
    assert model.explained_variance_ratio_.shape[0] == 3

    new_data = np.random.randn(5, 6)
    transformed = model.transform(new_data)
    assert transformed.shape == (5, 3)


def test_plot_helpers_return_figures():
    np.random.seed(1)
    data = np.random.randn(30, 4)
    model = PCAModel.fit(data, n_components=3, preprocessing="standardize")

    fig1 = model.plot_scores()
    fig2 = model.plot_loadings(component_idx=0)
    fig3 = model.plot_loadings_2d()
    fig4 = model.plot_variance_explained()
    fig5 = model.plot_eigenvalues()
    fig6, contribs = model.plot_contribs(indivs=0, pc=1)

    assert isinstance(fig1, plt.Figure)
    assert isinstance(fig2, plt.Figure)
    assert isinstance(fig3, plt.Figure)
    assert isinstance(fig4, plt.Figure)
    assert isinstance(fig5, plt.Figure)
    assert isinstance(fig6, plt.Figure)
    assert contribs.shape[1] == data.shape[1]

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)


def test_analysis_helpers_return_values():
    np.random.seed(2)
    data = np.random.randn(25, 5)
    model = PCAModel.fit(data, n_components=3, preprocessing="standardize")

    t2, f95, f99 = model.t2(plot=False)
    spe, c95, c99 = model.spe(plot=False)
    spe_jm, q95, q99 = model.spe_jackson_mudholkar(plot=False)
    sce = model.sce()
    out_t2, th_t2 = model.outliers_from_t2(alpha=0.99)
    out_spe, th_spe = model.outliers_from_spe(alpha=0.99)

    assert t2.shape == (25,)
    assert spe.shape == (25,)
    assert spe_jm.shape == (25,)
    assert sce.shape == (25,)

    assert f99 > f95
    assert c99 > c95
    assert q99 > q95

    assert th_t2 == f99
    assert th_spe == c99

    # outlier arrays should be valid integer indices (possibly empty)
    assert out_t2.dtype.kind in {"i", "u"}
    assert out_spe.dtype.kind in {"i", "u"}
