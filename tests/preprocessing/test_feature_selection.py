"""Tests for feature selection utilities."""

import numpy as np
import pytest
from dimensionality_reduction.preprocessing.feature_selection import select_features


class TestSelectFeatures:
    """Tests for select_features function."""
    
    def test_select_features_basic(self):
        """Test basic feature selection."""
        data = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7]])
        selected_data, indices = select_features(data, variance_threshold=0.0)
        
        # First column should be removed (constant)
        assert selected_data.shape == (3, 2)
        assert len(indices) == 2
        assert 0 not in indices
    
    def test_select_features_with_threshold(self):
        """Test feature selection with threshold."""
        data = np.array([[1, 2, 10], [1.1, 3, 20], [0.9, 4, 30]])
        selected_data, indices = select_features(data, variance_threshold=1.0)
        
        # Only features with variance > 1.0 should be selected
        assert selected_data.shape[1] <= data.shape[1]
        assert len(indices) == selected_data.shape[1]
    
    def test_select_features_no_removal(self):
        """Test feature selection when no features should be removed."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        selected_data, indices = select_features(data, variance_threshold=0.0)
        
        # All features should be kept
        assert selected_data.shape == data.shape
        assert len(indices) == data.shape[1]
