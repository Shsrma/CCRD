"""
Application configuration management.
Uses environment variables with sane defaults.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Core API Settings
    api_title: str = "Credit Card Fraud Detection API"
    api_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Database Configuration
    database_url: str = os.getenv(
        "DATABASE_URL", 
        "sqlite:///./fraud.db"
    )

    # Security
    secret_key: str = os.getenv(
        "SECRET_KEY", 
        "CHANGE_ME_IN_PRODUCTION_123456789"
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # CORS Configuration
    cors_origins: list = [
        os.getenv("FRONTEND_URL", "http://localhost:3000")
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["GET", "POST", "PUT", "DELETE"]
    cors_allow_headers: list = ["*"]

    # ML Model Configuration
    fraud_threshold: float = 0.5  # Default fraud probability threshold
    model_path: str = os.getenv(
        "MODEL_PATH",
        "backend/ml/models/model.pkl"
    )
    scaler_path: str = os.getenv(
        "SCALER_PATH",
        "backend/ml/models/scaler.pkl"
    )
    
    # ML Training Configuration
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    imbalance_method: str = "class_weights"  # Options: 'class_weights', 'smote', 'adasyn', etc.
    
    # Model Evaluation Configuration
    evaluation_metrics: list = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    # Data Configuration
    data_path: str = os.getenv(
        "DATA_PATH",
        "../data/creditcard.csv"
    )
    
    # Model Selection Configuration
    model_comparison_metrics: str = "f1"  # Metric to use for model selection

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Feature Configuration
    enable_otp: bool = os.getenv("ENABLE_OTP", "true").lower() == "true"
    enable_translations: bool = os.getenv("ENABLE_TRANSLATIONS", "false").lower() == "true"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached).
    Usage: settings = get_settings()
    """
    return Settings()
