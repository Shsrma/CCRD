"""
Model evaluation module for Credit Card Fraud Detection.

This module handles:
- Cross-validation
- Performance metrics calculation
- Model comparison
- ROC-AUC and Precision-Recall curves
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve, precision_recall_curve,
                           confusion_matrix, classification_report, average_precision_score,
                           matthews_corrcoef, cohen_kappa_score)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from app.core.logger import logger


class ModelEvaluator:
    """
    Handles model evaluation and cross-validation for fraud detection models.
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize the evaluator.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
    def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                           scoring: str = 'roc_auc') -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Trained model to evaluate
            X: Feature matrix
            y: Target vector
            scoring: Scoring metric
            
        Returns:
            Dictionary with cross-validation scores
        """
        logger.info(f"Performing {self.cv_folds}-fold cross-validation...")
        
        scores = cross_val_score(model, X, y, cv=self.cv, scoring=scoring)
        
        results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist(),
            f'{scoring}_score': scores.mean()
        }
        
        logger.info(f"{scoring.upper()} CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Advanced metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # ROC-AUC if probabilities are provided
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['matthews_correlation_coefficient'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['support'] = len(y_true)
        
        # Fraud-specific metrics
        metrics['fraud_detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      model_name: str = "Model", ax=None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model for labeling
            ax: Matplotlib axis to plot on (optional)
            
        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                   model_name: str = "Model", ax=None) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model for labeling
            ax: Matplotlib axis to plot on (optional)
            
        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str = "Model", ax=None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for labeling
            ax: Matplotlib axis to plot on (optional)
            
        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        return fig
    
    def compare_models(self, models: Dict[str, Any], X_train: np.ndarray, 
                      y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models based on various metrics.
        
        Args:
            models: Dictionary of model names and instances
            X_train: Training features
            y_train: Training labels
            X_test: Testing features
            y_test: Testing labels
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing models...")
        
        results = []
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Add cross-validation scores
            cv_results = self.cross_validate_model(model, X_train, y_train)
            metrics['cv_mean'] = cv_results['mean_score']
            metrics['cv_std'] = cv_results['std_score']
            
            # Add model name
            metrics['model'] = name
            
            results.append(metrics)
        
        # Convert to DataFrame for easy comparison
        results_df = pd.DataFrame(results)
        
        # Reorder columns to put model name first
        cols = ['model'] + [col for col in results_df.columns if col != 'model']
        results_df = results_df[cols]
        
        # Sort by F1 score (descending)
        results_df = results_df.sort_values(by='f1', ascending=False)
        
        logger.info("Model comparison completed.")
        return results_df.reset_index(drop=True)
    
    def hyperparameter_tuning(self, model: Any, param_grid: Dict, X_train: np.ndarray, 
                            y_train: np.ndarray, cv: int = 5, scoring: str = 'f1') -> GridSearchCV:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model: Model to tune
            param_grid: Parameter grid for tuning
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            GridSearchCV object with best parameters
        """
        logger.info(f"Tuning hyperparameters for {model.__class__.__name__}...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search


def evaluate_model_performance(model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                             X_test: np.ndarray, y_test: np.ndarray, 
                             model_name: str = "Model") -> Dict[str, Any]:
    """
    Comprehensive evaluation of a single model.
    
    Args:
        model: Model to evaluate
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = ModelEvaluator()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Cross-validation
    cv_results = evaluator.cross_validate_model(model, X_train, y_train)
    metrics.update(cv_results)
    
    # Add model name
    metrics['model_name'] = model_name
    
    return metrics


def generate_evaluation_report(model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray, 
                          model_name: str = "Model") -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report for a model.
    
    Args:
        model: Model to evaluate
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        model_name: Name of the model
        
    Returns:
        Dictionary with comprehensive evaluation report
    """
    evaluator = ModelEvaluator()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Cross-validation
    cv_results = evaluator.cross_validate_model(model, X_train, y_train)
    metrics.update(cv_results)
    
    # Add model name
    metrics['model_name'] = model_name
    
    # Create detailed report
    report = {
        'model_name': model_name,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'metrics': metrics,
        'sample_count': {
            'total': len(y_test),
            'actual_fraud': int(y_test.sum()),
            'actual_legitimate': int(len(y_test) - y_test.sum())
        }
    }
    
    return report


def compare_multiple_models(models_dict: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare multiple models comprehensively.
    
    Args:
        models_dict: Dictionary of model names and instances
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        
    Returns:
        DataFrame with comprehensive comparison
    """
    evaluator = ModelEvaluator()
    results = []
    
    for name, model in models_dict.items():
        logger.info(f"Evaluating {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred.astype(float)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Cross-validation
        try:
            cv_results = evaluator.cross_validate_model(model, X_train, y_train)
            metrics.update(cv_results)
        except Exception as e:
            logger.warning(f"Cross-validation failed for {name}: {str(e)}")
            metrics['cv_mean'] = 0
            metrics['cv_std'] = 0
        
        # Add model name
        metrics['model'] = name
        results.append(metrics)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['model'] + [col for col in df_results.columns if col != 'model']
    df_results = df_results[cols]
    
    # Sort by F1 score (important for fraud detection)
    df_results = df_results.sort_values(by='f1', ascending=False)
    
    return df_results