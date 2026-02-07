"""
Logging configuration for the application.
Enhanced with ML-specific logging capabilities.
"""

import logging
import logging.config
from app.core.config import get_settings
import os

settings = get_settings()


def configure_logging():
    """Configure structured logging for the application."""
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
            },
            "ml_formatter": {
                "format": "%(asctime)s - ML - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.log_level,
                "formatter": "detailed",
                "filename": "logs/app.log",
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 5,
            },
            "ml_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.log_level,
                "formatter": "ml_formatter",
                "filename": "logs/ml.log",
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "app": {
                "level": settings.log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "ml": {
                "level": settings.log_level,
                "handlers": ["console", "ml_file"],
                "propagate": False,
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["console", "file"],
        },
    }
    
    logging.config.dictConfig(logging_config)
    return logging.getLogger("app")


def get_ml_logger():
    """Get a logger specifically for ML operations."""
    return logging.getLogger("ml")


logger = configure_logging()
