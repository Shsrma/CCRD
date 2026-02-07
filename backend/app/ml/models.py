"""
Multiple ML models for Credit Card Fraud Detection.

This module implements:
- Logistic Regression
- Random Forest
- XGBoost
- Model selection and comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from app.core.logger import logger
from app.ml.imbalancer import get_imbalanced_model


class FraudDetectionModels:
    """
    Collection of ML models for fraud detection with utilities for training and prediction.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the models collection.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        
    def get_baseline_models(self, method: str = 'class_weights') -> Dict[str, Any]:
        """
        Get baseline models suitable for fraud detection.
        
        Args:
            method: Method to handle class imbalance ('class_weights' or 'sampling')
            
        Returns:
            Dictionary of model instances
        """
        models = {}
        
        # Logistic Regression with class weights
        models['logistic_regression'] = get_imbalanced_model(
            LogisticRegression,
            X=np.array([[1, 2], [2, 3]]),  # Dummy data to infer class weights
            y=np.array([0, 1]),
            method=method,
            max_iter=1000,
            solver='liblinear'
        )
        
        # Random Forest with class weights
        models['random_forest'] = get_imbalanced_model(
            RandomForestClassifier,
            X=np.array([[1, 2], [2, 3]]),
            y=np.array([0, 1]),
            method=method,
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        # XGBoost with class weights
        models['xgboost'] = get_imbalanced_model(
            XGBClassifier,
            X=np.array([[1, 2], [2, 3]]),
            y=np.array([0, 1]),
            method=method,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        # Naive Bayes (doesn't directly support class weights, but we'll add them)
        models['naive_bayes'] = GaussianNB()
        
        # SVM with class weights
        models['svm'] = get_imbalanced_model(
            SVC,
            X=np.array([[1, 2], [2, 3]]),
            y=np.array([0, 1]),
            method=method,
            probability=True  # Enable probability estimates
        )
        
        return models
    
    def get_tuned_models(self) -> Dict[str, Any]:
        """
        Get tuned models with optimized hyperparameters for fraud detection.
        
        Returns:
            Dictionary of tuned model instances
        """
        models = {}
        
        # Tuned Logistic Regression
        models['logistic_regression_tuned'] = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',
            C=0.1,
            max_iter=1000,
            solver='liblinear'
        )
        
        # Tuned Random Forest
        models['random_forest_tuned'] = RandomForestClassifier(
            random_state=self.random_state,
            class_weight='balanced',
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt'
        )
        
        # Tuned XGBoost
        models['xgboost_tuned'] = XGBClassifier(
            random_state=self.random_state,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=49,  # Approximate ratio of negative to positive in fraud data
            eval_metric='logloss'
        )
        
        return models
    
    def train_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                   model_name: str = "model") -> Any:
        """
        Train a model with logging.
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model for logging
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")
        
        try:
            model.fit(X_train, y_train)
            logger.info(f"Successfully trained {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                        model_types: str = 'baseline') -> Dict[str, Any]:
        """
        Train all models of specified type.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_types: Type of models to train ('baseline' or 'tuned')
            
        Returns:
            Dictionary of trained models
        """
        if model_types == 'baseline':
            models = self.get_baseline_models()
        elif model_types == 'tuned':
            models = self.get_tuned_models()
        else:
            raise ValueError("model_types must be 'baseline' or 'tuned'")
        
        trained_models = {}
        
        for name, model in models.items():
            trained_model = self.train_model(model, X_train, y_train, name)
            trained_models[name] = trained_model
        
        self.trained_models.update(trained_models)
        return trained_models
    
    def predict(self, model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with a model.
        
        Args:
            model: Trained model
            X: Features to predict
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # Probability of positive class
        
        return predictions, probabilities
    
    def predict_single(self, model: Any, features: np.ndarray) -> Tuple[int, float]:
        """
        Make a prediction for a single sample.
        
        Args:
            model: Trained model
            features: Single sample features
            
        Returns:
            Tuple of (prediction, probability)
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        pred, prob = self.predict(model, features)
        return int(pred[0]), float(prob[0])
    
    def get_model_by_name(self, name: str, method: str = 'class_weights') -> Any:
        """Get a specific model by name.

        Args:
            name: Name of the model ('logistic_regression', 'random_forest', 'xgboost', etc.)
            method: Method to handle class imbalance ('class_weights' or 'sampling')
            
        Returns:
            Model instance
        """
        all_models = {}
        all_models.update(self.get_baseline_models(method=method))
        all_models.update(self.get_tuned_models())
        
        if name in all_models:
            return all_models[name]
        else:
            raise ValueError(f"Model {name} not found. Available models: {list(all_models.keys())}")


class ModelSelector:
    """
    Selects the best model based on performance metrics.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.fraud_models = FraudDetectionModels(random_state=random_state)
    
    def find_best_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray,
                       evaluation_func: callable) -> Tuple[str, Any, Dict]:
        """
        Find the best model based on evaluation metrics.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Testing features
            y_test: Testing labels
            evaluation_func: Function to evaluate models (takes model, X_test, y_test)
            
        Returns:
            Tuple of (best_model_name, best_model, evaluation_results)
        """
        logger.info("Finding best model...")
        
        # Train multiple models
        trained_models = self.fraud_models.train_all_models(X_train, y_train, 'baseline')
        
        best_score = -1
        best_model_name = ""
        best_model = None
        all_results = {}
        
        for name, model in trained_models.items():
            logger.info(f"Evaluating {name}...")
            
            try:
                # Evaluate the model
                result = evaluation_func(model, X_test, y_test)
                
                # Extract the score (assuming it's a dictionary with a 'score' key)
                if isinstance(result, dict):
                    score = result.get('f1', result.get('roc_auc', result.get('accuracy', 0)))
                else:
                    score = result
                
                all_results[name] = {
                    'model': model,
                    'result': result,
                    'score': score
                }
                
                if score > best_score:
                    best_score = score
                    best_model_name = name
                    best_model = model
                    
                logger.info(f"{name} score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
                all_results[name] = {'error': str(e)}
        
        logger.info(f"Best model: {best_model_name} with score: {best_score:.4f}")
        
        return best_model_name, best_model, all_results


def create_fraud_detection_pipeline():
    """
    Create a complete fraud detection pipeline with models and preprocessing.
    
    Returns:
        FraudDetectionModels instance
    """
    return FraudDetectionModels()


def get_model_by_name(name: str, random_state: int = 42) -> Any:
    """
    Get a specific model by name.
    
    Args:
        name: Name of the model ('logistic_regression', 'random_forest', 'xgboost', etc.)
        random_state: Random seed for reproducibility
        
    Returns:
        Model instance
    """
    models_collection = FraudDetectionModels(random_state=random_state)
    
    all_models = {}
    all_models.update(models_collection.get_baseline_models())
    all_models.update(models_collection.get_tuned_models())
    
    if name in all_models:
        return all_models[name]
    else:
        raise ValueError(f"Model {name} not found. Available models: {list(all_models.keys())}")


def compare_model_performance(X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare the performance of different models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        
    Returns:
        DataFrame with performance comparison
    """
    models_collection = FraudDetectionModels()
    trained_models = models_collection.train_all_models(X_train, y_train, 'baseline')
    
    results = []
    
    for name, model in trained_models.items():
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics (basic ones here, evaluation module would have more comprehensive ones)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results.append({
                'model': name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            })
            
        except Exception as e:
            logger.error(f"Error evaluating {name}: {str(e)}")
            results.append({
                'model': name,
                'error': str(e)
            })
    
    return pd.DataFrame(results).sort_values('f1_score', ascending=False)