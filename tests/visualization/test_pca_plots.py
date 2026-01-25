"""Tests for PCA visualization utilities."""

import numpy as np
# import pytest
import matplotlib.pyplot as plt
from dimensionality_reduction.visualization.pca_plots import (
    plot_loadings, plot_variance_explained,
    plot_eigenvalues, plot_scores, plot_loadings_2d, plot_contribs
)


class TestPlotLoadings:
    """Tests for plot_loadings function."""

    def test_plot_loadings_basic(self):
        """Test basic loadings plot."""
        loadings = np.array([[0.5, -0.3], [0.3, 0.6], [-0.7, 0.2]])
        fig = plot_loadings(loadings, component_idx=0)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_loadings_with_names(self):
        """Test loadings plot with feature names."""
        loadings = np.array([[0.5, -0.3], [0.3, 0.6]])
        feature_names = ["Feature A", "Feature B"]
        fig = plot_loadings(loadings, feature_names=feature_names)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_loadings_top_n(self):
        """Test loadings plot with top_n parameter."""
        loadings = np.array([[0.5, -0.3], [0.3, 0.6], [-0.7, 0.2], [0.1, 0.4]])
        fig = plot_loadings(loadings, component_idx=0, top_n=2)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotVarianceExplained:
    """Tests for plot_variance_explained function."""

    def test_plot_variance_explained_basic(self):
        """Test basic variance explained plot."""
        explained_variance = np.array([0.4, 0.3, 0.2, 0.1])
        fig = plot_variance_explained(explained_variance)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_variance_explained_single_component(self):
        """Test variance explained plot with single component."""
        explained_variance = np.array([1.0])
        fig = plot_variance_explained(explained_variance)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotEigenvalues:
    """Tests for plot_eigenvalues function."""
    
    def test_plot_eigenvalues_basic(self):
        """Test basic eigenvalues plot."""
        eigenvalues = np.array([3.5, 2.1, 1.5, 0.8, 0.5])
        fig = plot_eigenvalues(eigenvalues)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_eigenvalues_first_n(self):
        """Test eigenvalues plot with first_n parameter."""
        eigenvalues = np.array([3.5, 2.1, 1.5, 0.8, 0.5])
        fig = plot_eigenvalues(eigenvalues, first_n=3)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotScores:
    """Tests for plot_scores function."""
    
    def test_plot_scores_basic(self):
        """Test basic scores plot."""
        scores = np.random.randn(50, 5)
        fig = plot_scores(scores)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_scores_with_color(self):
        """Test scores plot with color coding."""
        scores = np.random.randn(50, 5)
        score_color = np.random.randint(0, 3, 50)
        fig = plot_scores(scores, score_color=score_color)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_scores_with_annotations(self):
        """Test scores plot with annotations."""
        scores = np.random.randn(50, 5)
        fig = plot_scores(scores, annotate=[0, 10, 25])
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_scores_different_pcs(self):
        """Test scores plot with different PC combinations."""
        scores = np.random.randn(50, 5)
        fig = plot_scores(scores, pc1=2, pc2=3)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotLoadings2D:
    """Tests for plot_loadings_2d function."""
    
    def test_plot_loadings_2d_basic(self):
        """Test basic 2D loadings plot."""
        loadings = np.random.randn(5, 10)
        variables = [f"Var{i}" for i in range(10)]
        fig = plot_loadings_2d(loadings, variables)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_loadings_2d_with_labels(self):
        """Test 2D loadings plot with labels."""
        loadings = np.random.randn(5, 10)
        variables = [f"Var{i}" for i in range(10)]
        fig = plot_loadings_2d(loadings, variables, draw_labels=True)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_loadings_2d_no_color(self):
        """Test 2D loadings plot without color coding."""
        loadings = np.random.randn(5, 10)
        variables = [f"Var{i}" for i in range(10)]
        fig = plot_loadings_2d(loadings, variables, color_by=None)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_loadings_2d_cos2(self):
        """Test 2D loadings plot with cos2 coloring."""
        loadings = np.random.randn(5, 10)
        variables = [f"Var{i}" for i in range(10)]
        fig = plot_loadings_2d(loadings, variables, color_by="cos2")
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotContribs:
    """Tests for plot_contribs function."""
    
    def test_plot_contribs_single_observation(self):
        """Test contributions plot for single observation."""
        data = np.random.randn(50, 10)
        loadings = np.random.randn(5, 10)
        fig, contribs = plot_contribs(data, loadings, indivs=5, pc=1)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(contribs, np.ndarray)
        assert contribs.shape[1] == 10  # number of features
        plt.close(fig)
    
    def test_plot_contribs_multiple_observations(self):
        """Test contributions plot for multiple observations."""
        data = np.random.randn(50, 10)
        loadings = np.random.randn(5, 10)
        fig, contribs = plot_contribs(data, loadings, indivs=[5, 10, 15], pc=1)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(contribs, np.ndarray)
        assert contribs.shape == (3, 10)  # 3 observations, 10 features
        plt.close(fig)
    
    def test_plot_contribs_with_names(self):
        """Test contributions plot with variable names."""
        data = np.random.randn(50, 10)
        loadings = np.random.randn(5, 10)
        variable_names = [f"Feature_{i}" for i in range(10)]
        fig, _ = plot_contribs(data, loadings, indivs=5, pc=1, 
                                     variable_names=variable_names)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_contribs_non_simca(self):
        """Test contributions plot with non-SIMCA style."""
        data = np.random.randn(50, 10)
        loadings = np.random.randn(5, 10)
        fig, _ = plot_contribs(data, loadings, indivs=5, pc=1, 
                                     simca_style=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
