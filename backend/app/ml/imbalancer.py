"""
Class imbalance handling module for Credit Card Fraud Detection.

This module handles:
- SMOTE (Synthetic Minority Oversampling Technique)
- Class weight balancing
- Other techniques for handling imbalanced datasets
"""

import numpy as np
from typing import Tuple, Optional, Union
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from app.core.logger import logger


class ImbalanceHandler:
    """
    Handles class imbalance in datasets using various techniques.
    """
    
    def __init__(self, method: str = 'smote', random_state: int = 42):
        """
        Initialize the imbalance handler.
        
        Args:
            method: Method to use for handling imbalance ('smote', 'adasyn', 'borderline_smote', 
                   'svm_smote', 'smote_tomek', 'smote_enn', 'class_weights', 'undersample')
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.random_state = random_state
        self.sampler = None
        self.class_weights = None
        
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the imbalance handler and resample the data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Resampled feature matrix and target vector
        """
        original_dist = np.bincount(y)
        logger.info(f"Original distribution: {dict(zip(np.unique(y), original_dist))}")
        
        if self.method == 'smote':
            self.sampler = SMOTE(random_state=self.random_state, sampling_strategy='auto')
        elif self.method == 'adasyn':
            self.sampler = ADASYN(random_state=self.random_state, sampling_strategy='auto')
        elif self.method == 'borderline_smote':
            self.sampler = BorderlineSMOTE(random_state=self.random_state, sampling_strategy='auto')
        elif self.method == 'svm_smote':
            self.sampler = SVMSMOTE(random_state=self.random_state, sampling_strategy='auto')
        elif self.method == 'smote_tomek':
            self.sampler = SMOTETomek(random_state=self.random_state, sampling_strategy='auto')
        elif self.method == 'smote_enn':
            self.sampler = SMOTEENN(random_state=self.random_state, sampling_strategy='auto')
        elif self.method == 'undersample':
            self.sampler = EditedNearestNeighbours()
        elif self.method == 'class_weights':
            # For class weights, we just compute them and return original data
            self.class_weights = self.compute_balanced_class_weights(y)
            balanced_dist = np.bincount(y)
            logger.info(f"Using class weights instead of resampling. Distribution remains: {dict(zip(np.unique(y), balanced_dist))}")
            return X, y
        else:
            raise ValueError(f"Unknown method: {self.method}. Supported methods: smote, adasyn, borderline_smote, svm_smote, smote_tomek, smote_enn, class_weights, undersample")
        
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        
        new_dist = np.bincount(y_resampled)
        logger.info(f"Resampled distribution: {dict(zip(np.unique(y_resampled), new_dist))}")
        
        return X_resampled, y_resampled
    
    def compute_balanced_class_weights(self, y: np.ndarray) -> dict:
        """
        Compute balanced class weights.
        
        Args:
            y: Target vector
            
        Returns:
            Dictionary with class weights
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        logger.info(f"Computed class weights: {class_weights}")
        return class_weights
    
    def get_sampler(self):
        """
        Get the fitted sampler (for methods that use sampling).
        
        Returns:
            Fitted sampler object
        """
        return self.sampler
    
    def get_class_weights(self):
        """
        Get computed class weights (for class_weights method).
        
        Returns:
            Dictionary with class weights
        """
        return self.class_weights


def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to balance the dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed for reproducibility
        
    Returns:
        Balanced feature matrix and target vector
    """
    handler = ImbalanceHandler(method='smote', random_state=random_state)
    return handler.fit_resample(X, y)


def apply_class_weights(y: np.ndarray) -> dict:
    """
    Compute balanced class weights for the dataset.
    
    Args:
        y: Target vector
        
    Returns:
        Dictionary with class weights
    """
    handler = ImbalanceHandler(method='class_weights')
    handler.fit_resample(None, y)  # Only compute weights
    return handler.get_class_weights()


def compare_imbalance_methods(X: np.ndarray, y: np.ndarray, 
                            methods: Optional[list] = None, 
                            random_state: int = 42) -> dict:
    """
    Compare different imbalance handling methods.
    
    Args:
        X: Feature matrix
        y: Target vector
        methods: List of methods to compare (default: all available)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with results for each method
    """
    if methods is None:
        methods = ['smote', 'adasyn', 'borderline_smote', 'svm_smote', 'smote_tomek', 'smote_enn']
    
    results = {}
    
    original_dist = np.bincount(y)
    logger.info(f"Original distribution: {dict(zip(np.unique(y), original_dist))}")
    
    for method in methods:
        logger.info(f"Testing method: {method}")
        try:
            handler = ImbalanceHandler(method=method, random_state=random_state)
            X_resampled, y_resampled = handler.fit_resample(X, y)
            
            new_dist = np.bincount(y_resampled)
            results[method] = {
                'X': X_resampled,
                'y': y_resampled,
                'distribution': dict(zip(np.unique(y_resampled), new_dist)),
                'sample_count': len(y_resampled)
            }
            
        except Exception as e:
            logger.error(f"Error applying {method}: {str(e)}")
            results[method] = {'error': str(e)}
    
    return results


def get_imbalanced_model(model_class, X: np.ndarray, y: np.ndarray, 
                        method: str = 'class_weights', random_state: int = 42, **kwargs):
    """
    Get a model configured to handle imbalanced data.
    
    Args:
        model_class: Model class (e.g., RandomForestClassifier, LogisticRegression)
        X: Feature matrix
        y: Target vector
        method: Method to handle imbalance ('class_weights' or 'sampling')
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments for the model
        
    Returns:
        Configured model instance
    """
    if method == 'class_weights':
        # Compute class weights and pass to model
        class_weights = apply_class_weights(y)
        
        # Add class_weight parameter to model kwargs
        model_kwargs = kwargs.copy()
        model_kwargs['class_weight'] = class_weights
        model_kwargs['random_state'] = random_state
        
        return model_class(**model_kwargs)
    
    elif method == 'sampling':
        # Return model without special configuration, sampling will be applied separately
        model_kwargs = kwargs.copy()
        model_kwargs['random_state'] = random_state
        return model_class(**model_kwargs)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'class_weights' or 'sampling'")


def prepare_imbalanced_data(X: np.ndarray, y: np.ndarray, method: str = 'smote', 
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """
    Prepare imbalanced data using the specified method.
    
    Args:
        X: Feature matrix
        y: Target vector
        method: Method to use for handling imbalance
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_processed, y_processed, class_weights_if_applicable)
    """
    if method == 'class_weights':
        # For class weights, return original data and weights
        class_weights = apply_class_weights(y)
        return X, y, class_weights
    else:
        # For sampling methods, return resampled data and no weights
        X_resampled, y_resampled = apply_smote(X, y, random_state)
        return X_resampled, y_resampled, None