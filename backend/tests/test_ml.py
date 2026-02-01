"""
Unit tests for ML model loading and predictions.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from app.ml import ModelPredictor, preprocess_input, ModelNotFoundError


class TestModelPredictor:
    """Test ML model predictor."""
    
    def test_invalid_model_path(self):
        """Test ModelPredictor with non-existent model."""
        with pytest.raises(ModelNotFoundError):
            ModelPredictor(
                model_path="nonexistent/model.pkl",
                scaler_path="nonexistent/scaler.pkl"
            )
    
    @patch('app.ml.pickle.load')
    def test_model_loading(self, mock_pickle, tmp_path):
        """Test loading model from file."""
        # This is a more advanced test that would require proper mocking
        # For actual tests, use real model files or more sophisticated mocks
        pass


class TestPreprocessInput:
    """Test feature preprocessing."""
    
    def test_preprocess_valid_input(self):
        """Test preprocessing valid transaction input."""
        amount = 100.50
        timestamp = 1704067200.0
        features = [0.1, -0.5, 0.3, 0.2]
        
        result = preprocess_input(amount, timestamp, features)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) == 6  # amount + timestamp + 4 features
    
    def test_preprocess_empty_features(self):
        """Test preprocessing with empty feature list."""
        amount = 50.0
        timestamp = 1704067200.0
        features = []
        
        result = preprocess_input(amount, timestamp, features)
        
        assert len(result) == 2  # Just amount and timestamp
    
    def test_preprocess_invalid_amount_type(self):
        """Test preprocessing with invalid amount type."""
        with pytest.raises(ValueError):
            preprocess_input("not a number", 1704067200.0, [0.1, 0.2])
    
    def test_preprocess_output_dtype(self):
        """Test that output is float type."""
        result = preprocess_input(100.0, 1704067200.0, [0.1, 0.2])
        
        assert result.dtype == np.float64
