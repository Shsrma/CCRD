"""
ML model utilities: loading, prediction, and feature preprocessing.
"""

import os
import pickle
import numpy as np
from typing import Tuple
from pathlib import Path
from app.core.logger import logger


class ModelNotFoundError(Exception):
    """Raised when ML model or scaler files are not found."""
    pass


class ModelPredictor:
    """Handles ML model loading and fraud prediction."""
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize the predictor with model and scaler paths.
        
        Args:
            model_path: Path to pickled RandomForest model
            scaler_path: Path to pickled StandardScaler
            
        Raises:
            ModelNotFoundError: If model or scaler files don't exist
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        
        if not self.model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {self.model_path}")
        if not self.scaler_path.exists():
            raise ModelNotFoundError(f"Scaler file not found: {self.scaler_path}")
        
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        logger.info(f"Model loaded from {self.model_path}")
    
    def _load_model(self):
        """Load pickled RandomForest model."""
        try:
            with open(self.model_path, "rb") as f:
                return pickle.load(f)
        except pickle.UnpicklingError as e:
            logger.error(f"Failed to unpickle model: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_scaler(self):
        """Load pickled StandardScaler."""
        try:
            with open(self.scaler_path, "rb") as f:
                return pickle.load(f)
        except pickle.UnpicklingError as e:
            logger.error(f"Failed to unpickle scaler: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Predict fraud probability for a transaction.
        
        Args:
            features: Feature vector (1D array of floats)
            
        Returns:
            Tuple of (prediction, probability)
                - prediction: 0 (legitimate) or 1 (fraudulent)
                - probability: Fraud confidence score [0, 1]
        """
        try:
            # Ensure correct shape
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Get probability for fraud class
            fraud_probability = self.model.predict_proba(scaled_features)[0][1]
            
            return fraud_probability
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Predict fraud probability for multiple transactions.
        
        Args:
            features: Feature matrix (N x D)
            
        Returns:
            Array of fraud probabilities for each transaction
        """
        try:
            scaled_features = self.scaler.transform(features)
            return self.model.predict_proba(scaled_features)[:, 1]
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise


def preprocess_input(
    amount: float,
    timestamp: float,
    feature_vector: list
) -> np.ndarray:
    """
    Preprocess transaction input into feature vector for ML model.
    
    Args:
        amount: Transaction amount
        timestamp: Transaction timestamp
        feature_vector: List of additional features
        
    Returns:
        1D numpy array of preprocessed features
    """
    try:
        # Combine all features
        features = np.array([amount, timestamp] + feature_vector, dtype=float)
        return features
    except (TypeError, ValueError) as e:
        logger.error(f"Feature preprocessing failed: {e}")
        raise ValueError("Invalid input features") from e
