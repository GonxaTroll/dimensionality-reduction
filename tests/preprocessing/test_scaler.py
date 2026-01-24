"""Tests for scaler utilities."""

import numpy as np
import pytest
from dimensionality_reduction.preprocessing.scaler import standardize, normalize


class TestStandardize:
    """Tests for standardize function."""
    
    def test_standardize_basic(self):
        """Test basic standardization."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        standardized, mean, std = standardize(data)
        
        # Check that mean is approximately 0
        assert np.allclose(np.mean(standardized, axis=0), 0, atol=1e-10)
        # Check that std is approximately 1
        assert np.allclose(np.std(standardized, axis=0), 1, atol=1e-10)
    
    def test_standardize_with_precomputed(self):
        """Test standardization with precomputed mean and std."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        mean = np.array([3, 4])
        std = np.array([2, 2])
        
        standardized, ret_mean, ret_std = standardize(data, mean, std)
        
        assert np.array_equal(ret_mean, mean)
        assert np.array_equal(ret_std, std)
        assert standardized.shape == data.shape
    
    def test_standardize_zero_std(self):
        """Test standardization with zero standard deviation."""
        data = np.array([[1, 5], [1, 6], [1, 7]])
        standardized, mean, std = standardize(data)
        
        # First column should remain unchanged (std is 0)
        assert np.all(standardized[:, 0] == 0)


class TestNormalize:
    """Tests for normalize function."""
    
    def test_normalize_basic(self):
        """Test basic normalization."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        normalized, min_val, max_val = normalize(data)
        
        # Check that min is 0 and max is 1
        assert np.allclose(np.min(normalized, axis=0), 0)
        assert np.allclose(np.max(normalized, axis=0), 1)
    
    def test_normalize_with_precomputed(self):
        """Test normalization with precomputed min and max."""
        data = np.array([[2, 3], [4, 5]])
        min_val = np.array([0, 0])
        max_val = np.array([10, 10])
        
        normalized, ret_min, ret_max = normalize(data, min_val, max_val)
        
        assert np.array_equal(ret_min, min_val)
        assert np.array_equal(ret_max, max_val)
        assert normalized.shape == data.shape
    
    def test_normalize_constant_feature(self):
        """Test normalization with constant feature."""
        data = np.array([[5, 2], [5, 4], [5, 6]])
        normalized, min_val, max_val = normalize(data)
        
        # First column should remain unchanged (no range)
        assert np.all(normalized[:, 0] == 0)
