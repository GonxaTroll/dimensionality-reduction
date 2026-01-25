"""
Tests for PCA analysis functions.
"""

import numpy as np
import pytest
from dimensionality_reduction.visualization.pca_analysis import (
    get_last_best_eigenvalue,
    compute_sce,
    compute_t2_hotelling,
    compute_spe,
    compute_spe_jackson_mudholkar,
    get_outlier_indexes
)


class TestPCAAnalysis:
    """Test cases for PCA analysis functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 50
        n_features = 10
        n_components = 5
        
        original_data = np.random.randn(n_samples, n_features)
        scores = np.random.randn(n_samples, n_components)
        loadings = np.random.randn(n_components, n_features)
        eigenvalues = np.array([3.5, 2.1, 1.5, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02])
        
        return {
            'original_data': original_data,
            'scores': scores,
            'loadings': loadings,
            'eigenvalues': eigenvalues,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_components': n_components
        }
    
    def test_get_last_best_eigenvalue(self, sample_data, capsys):
        """Test eigenvalue filtering by Kaiser criterion."""
        eigenvalues = sample_data['eigenvalues']
        result = get_last_best_eigenvalue(eigenvalues)
        
        # Should return eigenvalues > 1
        assert np.all(result > 1)
        assert len(result) == 3  # 3.5, 2.1, 1.5
        
        # Check printed output
        captured = capsys.readouterr()
        assert "First 3 components" in captured.out
    
    def test_compute_sce(self, sample_data):
        """Test SCE computation."""
        scores = sample_data['scores']
        selected_components = 3
        
        sce = compute_sce(scores, selected_components)
        
        # Check output shape
        assert sce.shape == (sample_data['n_samples'],)
        
        # Check values are non-negative
        assert np.all(sce >= 0)
        
        # Manual calculation check for first observation
        expected_first = np.sum(scores[0, :selected_components] ** 2)
        assert np.isclose(sce[0], expected_first)
    
    def test_compute_t2_hotelling(self, sample_data):
        """Test Hotelling T² computation."""
        scores = sample_data['scores']
        eigenvalues = sample_data['eigenvalues']
        selected_components = 3
        
        # Test without plot
        t2, f95, f99 = compute_t2_hotelling(
            scores, eigenvalues, selected_components, plot=False
        )
        
        # Check output shapes
        assert t2.shape == (sample_data['n_samples'],)
        assert isinstance(f95, (float, np.floating))
        assert isinstance(f99, (float, np.floating))
        
        # Check values are non-negative
        assert np.all(t2 >= 0)
        assert f95 > 0
        assert f99 > 0
        
        # 99% threshold should be higher than 95%
        assert f99 > f95
    
    def test_compute_spe(self, sample_data):
        """Test SPE computation with chi-squared approximation."""
        original_data = sample_data['original_data']
        scores = sample_data['scores']
        loadings = sample_data['loadings']
        selected_components = 3
        
        # Test without plot
        spe, chi2_95, chi2_99 = compute_spe(
            original_data, scores, loadings, selected_components, plot=False
        )
        
        # Check output shapes
        assert spe.shape == (sample_data['n_samples'],)
        assert isinstance(chi2_95, (float, np.floating))
        assert isinstance(chi2_99, (float, np.floating))
        
        # Check values are non-negative
        assert np.all(spe >= 0)
        assert chi2_95 > 0
        assert chi2_99 > 0
        
        # 99% threshold should be higher than 95%
        assert chi2_99 > chi2_95
    
    def test_compute_spe_jackson_mudholkar(self, sample_data):
        """Test SPE computation with Jackson-Mudholkar approximation."""
        original_data = sample_data['original_data']
        scores = sample_data['scores']
        loadings = sample_data['loadings']
        eigenvalues = sample_data['eigenvalues']
        selected_components = 3
        
        # Test without plot
        spe, q095, q099 = compute_spe_jackson_mudholkar(
            original_data, scores, loadings, eigenvalues, 
            selected_components, plot=False
        )
        
        # Check output shapes
        assert spe.shape == (sample_data['n_samples'],)
        assert isinstance(q095, (float, np.floating))
        assert isinstance(q099, (float, np.floating))
        
        # Check values are non-negative
        assert np.all(spe >= 0)
        assert q095 > 0
        assert q099 > 0
        
        # 99% threshold should be higher than 95%
        assert q099 > q095
    
    def test_get_outlier_indexes(self, capsys):
        """Test outlier detection."""
        statistic_values = np.array([1, 2, 10, 3, 15, 2, 1, 20, 3, 2])
        threshold = 5.0
        
        outliers = get_outlier_indexes(statistic_values, threshold)
        
        # Check output
        expected_outliers = np.array([2, 4, 7])  # indices with values > 5
        assert np.array_equal(outliers, expected_outliers)
        
        # Check printed output
        captured = capsys.readouterr()
        assert "Number of outliers detected: 3" in captured.out
    
    def test_get_outlier_indexes_no_outliers(self, capsys):
        """Test outlier detection with no outliers."""
        statistic_values = np.array([1, 2, 3, 4, 5])
        threshold = 10.0
        
        outliers = get_outlier_indexes(statistic_values, threshold)
        
        # Check no outliers found
        assert len(outliers) == 0
        
        # Check printed output
        captured = capsys.readouterr()
        assert "Number of outliers detected: 0" in captured.out
