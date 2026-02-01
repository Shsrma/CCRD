"""
Production-ready entry point for the FastAPI application.

Usage:
    python main.py
    # or
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import uvicorn
from app.main import app
from app.core.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
    )
