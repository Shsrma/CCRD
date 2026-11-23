"""
FastAPI application entry point.
Initializes the API with all routes, middleware, and dependencies.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Annotated
import os
import sys
from pathlib import Path

# Add project root and backend directory to path so root and app modules are discoverable
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(1, str(BACKEND_DIR))

from app.core.config import get_settings
from app.core.logger import logger
from app.database.engine import create_all_tables, get_db
from app.models import User
from app.api.dependencies import get_current_user
from app.api.routes import auth, transactions, alerts, settings as settings_routes, organizations

from app.ml.trainer import ModelTrainer
from app.ml.preprocessing import DataPreprocessor
from app.core.config import get_settings
import pickle

settings = get_settings()

# Global variables for ML model and scaler
ml_model = None
ml_scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting Credit Card Fraud Detection API")
    create_all_tables()
    logger.info("✅ Database tables initialized")
    
    # Load ML model and scaler
    global ml_model, ml_scaler
    try:
        model_path = settings.model_path
        scaler_path = settings.scaler_path
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, "rb") as f:
                ml_model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                ml_scaler = pickle.load(f)
            logger.info(f"✅ Loaded ML model from {model_path}")
            logger.info(f"✅ Loaded ML scaler from {scaler_path}")
        else:
            logger.warning(f"⚠️ ML model files not found at {model_path} and {scaler_path}")
            logger.info("💡 Please train the model using: python -m ml.train_model")
    except Exception as e:
        logger.error(f"❌ Failed to load ML model: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down API")


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="ML-powered credit card fraud detection system",
    lifespan=lifespan,
    debug=settings.debug,
)


# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"},
    )


# ============================================================================
# HEALTH & MODEL METADATA ENDPOINTS
# ============================================================================

@app.get("/health", tags=["System"])
def health_check():
    """
    Health check endpoint.
    Returns 200 OK if service is operational.
    """
    try:
        from app.api.routes.transactions import fraud_engine
    except ImportError:
        from backend.app.api.routes.transactions import fraud_engine
    return {
        "status": "healthy",
        "version": settings.api_version,
        "product_model_operational": fraud_engine.is_operational(),
        "model_version": fraud_engine.version,
    }


@app.get("/api/v1/health", tags=["System"])
def health_check_v1():
    """Detailed health check for API v1."""
    try:
        from app.api.routes.transactions import fraud_engine
    except ImportError:
        from backend.app.api.routes.transactions import fraud_engine
    return {
        "status": "healthy",
        "service": "CCRD Real-Time Fraud Intelligence API",
        "version": settings.api_version,
        "product_model_operational": fraud_engine.is_operational(),
        "model_version": fraud_engine.version,
    }


@app.get("/api/v1/model/info", tags=["System"])
def get_model_info():
    """Returns active fraud detection model version, rules, and evaluation metrics."""
    try:
        from app.api.routes.transactions import fraud_engine
    except ImportError:
        from backend.app.api.routes.transactions import fraud_engine
    metadata = fraud_engine.metadata
    return {
        "model_name": metadata.get("model_name", "logistic_regression_calibrated"),
        "version": fraud_engine.version,
        "is_operational": fraud_engine.is_operational(),
        "threshold_config": {
            "low_risk_upper_bound": fraud_engine.low_threshold,
            "medium_risk_upper_bound": fraud_engine.high_threshold,
            "default_action_threshold": 0.50,
        },
        "input_features": metadata.get("input_features", [
            "amount", "currency", "merchant_category", "transaction_hour", "country",
            "customer_age_days", "transactions_last_hour", "average_transaction_amount",
            "failed_transactions_last_day", "device_trust_score"
        ]),
        "registered_rules": fraud_engine.rules_engine.get_registered_rules(),
        "evaluation_metrics": metadata.get("evaluation_metrics", {}),
        "synthetic_disclaimer": metadata.get("dataset_metadata", {}).get(
            "synthetic_disclaimer",
            "Synthetic transaction evaluation for demonstration."
        ),
    }


# ============================================================================
# API ROUTES
# ============================================================================

# Authentication routes (public)
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])

# Protected routes
app.include_router(
    organizations.router,
    prefix="/api/v1/organizations",
    tags=["Organizations & Multi-Tenancy"],
)

app.include_router(
    transactions.router,
    prefix="/api/v1/transactions",
    tags=["Transactions"],
)

app.include_router(
    alerts.router,
    prefix="/api/v1/alerts",
    tags=["Alerts"],
)

app.include_router(
    settings_routes.router,
    prefix="/api/v1/settings",
    tags=["Settings"],
    dependencies=[Depends(get_current_user)]
)


# ============================================================================
# FRONTEND STATIC MOUNT & ROOT ENDPOINT
# ============================================================================

from fastapi.staticfiles import StaticFiles

frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))

@app.get("/api/info", tags=["System"])
def api_info():
    """Provides metadata about the running fraud intelligence API."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "ml_model_loaded": ml_model is not None,
    }

if os.path.exists(frontend_dist):
    logger.info(f"Serving built React frontend from {frontend_dist}")
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/", tags=["System"])
    def root():
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "ml_model_loaded": ml_model is not None,
        }



if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )