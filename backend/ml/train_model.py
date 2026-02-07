import os
import sys
from pathlib import Path

# Add the backend directory to the path so we can import our modules
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.ml.trainer import train_fraud_detection_model
from app.ml.preprocessing import DataPreprocessor
from app.core.config import get_settings
from app.core.logger import logger
import argparse

def main():
    """Main training function with command-line interface."""
    parser = argparse.ArgumentParser(description='Train Credit Card Fraud Detection Model')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['logistic_regression', 'random_forest', 'xgboost', 'naive_bayes', 'svm'],
                        help='Model to train (default: random_forest)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the credit card dataset')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of dataset for testing (default: 0.2)')
    parser.add_argument('--imbalance-method', type=str, default='class_weights',
                        choices=['class_weights', 'smote', 'adasyn', 'borderline_smote', 'svm_smote', 'smote_tomek', 'smote_enn'],
                        help='Method to handle class imbalance (default: class_weights)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='Whether to save the trained model')
    
    args = parser.parse_args()
    
    # Get settings
    settings = get_settings()
    
    logger.info(f"Starting fraud detection model training for {args.model}")
    logger.info(f"Configuration: model={args.model}, random_state={args.random_state}, test_size={args.test_size}, imbalance_method={args.imbalance_method}")
    
    try:
        # Train the model using our new framework
        result = train_fraud_detection_model(
            model_name=args.model,
            data_path=args.data_path,
            random_state=args.random_state,
            test_size=args.test_size,
            imbalance_method=args.imbalance_method,
            save_model=args.save_model
        )
        
        # Print summary
        metrics = result['evaluation_report']['metrics']
        logger.info(f"Training completed successfully!")
        logger.info(f"Final metrics for {args.model}: F1-Score: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Save experiment metadata if model was saved
        if args.save_model:
            exp_metadata = result['experiment_metadata']
            model_path = exp_metadata.get('model_path', 'model.pkl')
            scaler_path = exp_metadata.get('scaler_path', 'scaler.pkl')
            
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Scaler saved to: {scaler_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()