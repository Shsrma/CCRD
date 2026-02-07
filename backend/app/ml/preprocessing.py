"""
Data preprocessing module for Credit Card Fraud Detection.

This module handles:
- Loading and validating credit card data
- Feature scaling
- Handling imbalanced datasets
- Data validation
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from app.core.logger import logger
from app.core.config import get_settings


class DataPreprocessor:
    """
    Handles data preprocessing for credit card fraud detection.
    """
    
    def __init__(self, 
                 data_path: Optional[str] = None, 
                 test_size: float = 0.2, 
                 random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            data_path: Path to the credit card dataset
            test_size: Proportion of dataset for testing
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path or self._get_default_data_path()
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        
    def _get_default_data_path(self) -> str:
        """Get the default path for the credit card dataset."""
        settings = get_settings()
        # Look for data in common locations
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "creditcard.csv"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "data", "creditcard.csv"),
            "data/creditcard.csv",
            "../data/creditcard.csv"
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                logger.info(f"Found dataset at: {abs_path}")
                return abs_path
        
        raise FileNotFoundError(
            "Credit card dataset not found. Expected at 'data/creditcard.csv' "
            "relative to project root. Please download the dataset from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the credit card fraud dataset.
        
        Returns:
            DataFrame containing the loaded data
        """
        logger.info(f"Loading data from: {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Validate required columns
            required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                             'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                             'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                             'V28', 'Amount', 'Class']
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info(f"Dataset has {len(df)} samples and {len(df.columns)} features")
            logger.info(f"Fraud distribution: {df['Class'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the loaded data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Dataset contains {missing_values} missing values")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_values = np.isinf(df[numeric_cols]).sum().sum()
        if inf_values > 0:
            logger.warning(f"Dataset contains {inf_values} infinite values")
        
        # Validate target column
        if 'Class' not in df.columns:
            raise ValueError("Dataset must contain 'Class' column")
        
        if not set(df['Class'].unique()).issubset({0, 1}):
            logger.warning("Target column contains values other than 0 and 1")
        
        return True
    
    def split_data(self, df: pd.DataFrame, 
                   target_column: str = 'Class',
                   stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into training and testing sets.
        
        Args:
            df: Input dataframe
            target_column: Name of the target column
            stratify: Whether to stratify the split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train and test sets...")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Train fraud rate: {y_train.mean():.4f}")
        logger.info(f"Test fraud rate: {y_test.mean():.4f}")
        
        return X_train.values, X_test.values, y_train.values, y_test.values
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            Scaled training and testing features
        """
        logger.info("Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Feature scaling completed")
        return X_train_scaled, X_test_scaled
    
    def apply_smote(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance the training dataset.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Balanced training features and labels
        """
        logger.info("Applying SMOTE to balance training data...")
        
        smote = SMOTE(random_state=self.random_state, sampling_strategy='auto')
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE - Original: {len(y_train)}, Balanced: {len(y_train_balanced)}")
        logger.info(f"After SMOTE - Fraud samples: {y_train_balanced.sum()}")
        
        return X_train_balanced, y_train_balanced
    
    def preprocess_for_prediction(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess features for prediction (scaling).
        
        Args:
            features: Input features to scale
            
        Returns:
            Scaled features
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        return self.scaler.transform(features)


def load_and_preprocess_data(data_path: Optional[str] = None,
                           test_size: float = 0.2,
                           apply_smote: bool = True,
                           random_state: int = 42) -> Tuple:
    """
    High-level function to load and preprocess credit card fraud data.
    
    Args:
        data_path: Path to the dataset
        test_size: Proportion of dataset for testing
        apply_smote: Whether to apply SMOTE for balancing
        random_state: Random seed for reproducibility
        
    Returns:
        Processed data: (X_train, X_test, y_train, y_test, scaler)
    """
    preprocessor = DataPreprocessor(data_path, test_size, random_state)
    
    # Load and validate data
    df = preprocessor.load_data()
    preprocessor.validate_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(df)
    
    # Scale features
    X_train, X_test = preprocessor.scale_features(X_train, X_test)
    
    # Apply SMOTE if requested
    if apply_smote:
        X_train, y_train = preprocessor.apply_smote(X_train, y_train)
    
    return X_train, X_test, y_train, y_test, preprocessor.scaler