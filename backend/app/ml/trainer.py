"""
Model trainer module for Credit Card Fraud Detection with reproducibility features.

This module handles:
- Reproducible model training
- Model saving and loading
- Experiment tracking
- Performance monitoring
"""

import os
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from app.core.logger import get_ml_logger, logger
from app.core.config import get_settings
from app.ml.preprocessing import DataPreprocessor
from app.ml.models import FraudDetectionModels
from app.ml.evaluation import ModelEvaluator, generate_evaluation_report
from app.ml.imbalancer import ImbalanceHandler


class ModelTrainer:
    """
    Handles model training with reproducibility features.
    """
    
    def __init__(self, 
                 model_name: str = 'random_forest',
                 random_state: int = 42,
                 test_size: float = 0.2):
        """
        Initialize the trainer.
        
        Args:
            model_name: Name of the model to train
            random_state: Random seed for reproducibility
            test_size: Proportion of dataset for testing
        """
        self.model_name = model_name
        self.random_state = random_state
        self.test_size = test_size
        self.settings = get_settings()
        self.ml_logger = get_ml_logger()
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        os.environ['PYTHONHASHSEED'] = str(random_state)
        
        # Initialize components
        self.preprocessor = DataPreprocessor(test_size=test_size, random_state=random_state)
        self.fraud_models = FraudDetectionModels(random_state=random_state)
        self.evaluator = ModelEvaluator(cv_folds=self.settings.cv_folds, random_state=random_state)
        self.trained_model = None
        self.scaler = None
        self.experiment_metadata = {}
        
    def set_random_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_state)
        os.environ['PYTHONHASHSEED'] = str(self.random_state)
        
    def train(self, 
              data_path: Optional[str] = None,
              imbalance_method: str = 'class_weights',
              save_model: bool = True,
              model_save_path: Optional[str] = None,
              scaler_save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the model with reproducibility features.
        
        Args:
            data_path: Path to the dataset
            imbalance_method: Method to handle class imbalance
            save_model: Whether to save the trained model
            model_save_path: Path to save the model
            scaler_save_path: Path to save the scaler
            
        Returns:
            Dictionary with training results and metadata
        """
        self.ml_logger.info(f"Starting training for {self.model_name}")
        self.ml_logger.info(f"Using random seed: {self.random_state}")
        
        # Set random seeds
        self.set_random_seeds()
        
        # Record experiment metadata
        self.experiment_metadata = {
            'model_name': self.model_name,
            'random_state': self.random_state,
            'test_size': self.test_size,
            'imbalance_method': imbalance_method,
            'training_timestamp': datetime.now().isoformat(),
            'data_path': data_path or self.preprocessor.data_path
        }
        
        # Load and preprocess data
        self.ml_logger.info("Loading and preprocessing data...")
        df = self.preprocessor.load_data()
        self.preprocessor.validate_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(df)
        
        # Store scaler for later use
        self.scaler = self.preprocessor.scaler
        
        # Handle class imbalance
        if imbalance_method != 'class_weights':
            imbalance_handler = ImbalanceHandler(method=imbalance_method, random_state=self.random_state)
            X_train, y_train = imbalance_handler.fit_resample(X_train, y_train)
        else:
            # For class weights, we'll pass the weights to the model
            class_weights = imbalance_handler = ImbalanceHandler(method='class_weights')
            _, _, class_weights = imbalance_handler.fit_resample(X_train, y_train)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        
        # Get and train model
        model = self.fraud_models.get_model_by_name(self.model_name, random_state=self.random_state)
        
        if imbalance_method == 'class_weights' and hasattr(model, 'class_weight'):
            # Apply class weights if the model supports it
            model.class_weight = class_weights
        
        self.ml_logger.info(f"Training {self.model_name}...")
        self.trained_model = model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        self.ml_logger.info("Evaluating model...")
        evaluation_report = generate_evaluation_report(
            self.trained_model, 
            X_train_scaled, y_train, 
            X_test_scaled, y_test, 
            self.model_name
        )
        
        # Record final metrics
        self.experiment_metadata['final_metrics'] = evaluation_report['metrics']
        
        # Save model and scaler if requested
        if save_model:
            model_path = model_save_path or self.settings.model_path
            scaler_path = scaler_save_path or self.settings.scaler_path
            
            self.save_model(model_path)
            self.save_scaler(scaler_path)
            
            self.experiment_metadata['model_path'] = model_path
            self.experiment_metadata['scaler_path'] = scaler_path
        
        self.ml_logger.info(f"Training completed. Final F1 score: {evaluation_report['metrics']['f1']:.4f}")
        
        return {
            'model': self.trained_model,
            'scaler': self.scaler,
            'evaluation_report': evaluation_report,
            'experiment_metadata': self.experiment_metadata
        }
    
    def save_model(self, path: str):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.trained_model, f)
        self.ml_logger.info(f"Model saved to: {path}")
    
    def save_scaler(self, path: str):
        """Save the scaler to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        self.ml_logger.info(f"Scaler saved to: {path}")
    
    def save_experiment_metadata(self, path: str):
        """Save experiment metadata to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2, default=str)
        self.ml_logger.info(f"Experiment metadata saved to: {path}")
    
    def load_model(self, path: str):
        """Load a trained model from disk."""
        with open(path, 'rb') as f:
            self.trained_model = pickle.load(f)
        self.ml_logger.info(f"Model loaded from: {path}")
    
    def load_scaler(self, path: str):
        """Load a scaler from disk."""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.ml_logger.info(f"Scaler loaded from: {path}")
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """Make a prediction with the trained model."""
        if self.trained_model is None or self.scaler is None:
            raise ValueError("Model or scaler not trained/loaded")
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.trained_model.predict(scaled_features)[0]
        probability = self.trained_model.predict_proba(scaled_features)[0, 1]
        
        return int(prediction), float(probability)


class ExperimentTracker:
    """
    Tracks experiments and ensures reproducibility.
    """
    
    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = experiment_dir
        self.ml_logger = get_ml_logger()
        os.makedirs(experiment_dir, exist_ok=True)
    
    def create_experiment(self, name: str, config: Dict[str, Any]) -> str:
        """
        Create a new experiment directory and save configuration.
        
        Args:
            name: Name of the experiment
            config: Configuration dictionary
            
        Returns:
            Path to the experiment directory
        """
        exp_path = os.path.join(self.experiment_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(exp_path, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(exp_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.ml_logger.info(f"Created experiment: {exp_path}")
        return exp_path
    
    def log_result(self, experiment_path: str, result: Dict[str, Any]):
        """
        Log experiment results.
        
        Args:
            experiment_path: Path to the experiment directory
            result: Results dictionary
        """
        results_path = os.path.join(experiment_path, "results.json")
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        self.ml_logger.info(f"Results logged to: {results_path}")
    
    def compare_experiments(self, experiment_paths: list) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            experiment_paths: List of experiment directory paths
            
        Returns:
            DataFrame with experiment comparison
        """
        results = []
        
        for exp_path in experiment_paths:
            results_path = os.path.join(exp_path, "results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    exp_result = json.load(f)
                
                # Extract key metrics
                metrics = exp_result.get('experiment_metadata', {}).get('final_metrics', {})
                config = {}
                
                config_path = os.path.join(exp_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                
                row = {
                    'experiment': os.path.basename(exp_path),
                    'model_name': config.get('model_name', 'unknown'),
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': metrics.get('f1', 0),
                    'roc_auc': metrics.get('roc_auc', 0),
                    'random_state': config.get('random_state', 42)
                }
                results.append(row)
        
        return pd.DataFrame(results).sort_values('f1', ascending=False)


def train_fraud_detection_model(model_name: str = 'random_forest',
                              data_path: Optional[str] = None,
                              random_state: int = 42,
                              test_size: float = 0.2,
                              imbalance_method: str = 'class_weights',
                              save_model: bool = True) -> Dict[str, Any]:
    """
    Convenience function to train a fraud detection model.
    
    Args:
        model_name: Name of the model to train
        data_path: Path to the dataset
        random_state: Random seed for reproducibility
        test_size: Proportion of dataset for testing
        imbalance_method: Method to handle class imbalance
        save_model: Whether to save the trained model
        
    Returns:
        Dictionary with training results
    """
    trainer = ModelTrainer(
        model_name=model_name,
        random_state=random_state,
        test_size=test_size
    )
    
    return trainer.train(
        data_path=data_path,
        imbalance_method=imbalance_method,
        save_model=save_model
    )


def reproduce_experiment(config_path: str) -> Dict[str, Any]:
    """
    Reproduce an experiment from a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with reproduction results
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    trainer = ModelTrainer(
        model_name=config.get('model_name', 'random_forest'),
        random_state=config.get('random_state', 42),
        test_size=config.get('test_size', 0.2)
    )
    
    return trainer.train(
        data_path=config.get('data_path'),
        imbalance_method=config.get('imbalance_method', 'class_weights'),
        save_model=config.get('save_model', True)
    )