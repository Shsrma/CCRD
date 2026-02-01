"""
Root package initializer for app.
"""

from app.core.logger import logger

__version__ = "1.0.0"

logger.info(f"Initializing CCRD v{__version__}")
