"""Tests for PCA visualization utilities."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from dimensionality_reduction.visualization.pca_plots import (
    plot_loadings, plot_components, plot_variance_explained
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


class TestPlotComponents:
    """Tests for plot_components function."""
    
    def test_plot_components_basic(self):
        """Test basic components plot."""
        transformed_data = np.random.randn(50, 3)
        fig = plot_components(transformed_data)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_components_with_labels(self):
        """Test components plot with labels."""
        transformed_data = np.random.randn(50, 3)
        labels = np.random.randint(0, 3, 50)
        fig = plot_components(transformed_data, labels=labels)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_components_different_axes(self):
        """Test components plot with different component axes."""
        transformed_data = np.random.randn(50, 4)
        fig = plot_components(transformed_data, component_x=1, component_y=2)
        
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
