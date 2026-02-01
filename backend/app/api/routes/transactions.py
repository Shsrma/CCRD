"""
Transaction prediction routes.
"""

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logger import logger
from app.database.engine import get_db
from app.models import User, Transaction, Alert
from app.schemas import TransactionInput, PredictionResponse
from app.ml import ModelPredictor, preprocess_input

settings = get_settings()
router = APIRouter()

# Initialize ML model (loaded once at startup)
try:
    predictor = ModelPredictor(settings.model_path, settings.scaler_path)
except Exception as e:
    logger.error(f"Failed to load ML model: {e}")
    predictor = None


@router.post("/predict", response_model=PredictionResponse)
def predict_transaction(
    data: TransactionInput,
    current_user: User = Depends(),  # Injected by router dependency
    db: Session = Depends(get_db)
):
    """
    Predict if a transaction is fraudulent.
    
    Returns:
    - **transaction_id**: ID of stored transaction record
    - **fraud_prediction**: 0 (legitimate) or 1 (fraudulent)
    - **probability**: Fraud confidence [0.0, 1.0]
    """
    
    if predictor is None:
        logger.error("ML model not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not available"
        )
    
    try:
        # Preprocess features
        features = preprocess_input(data.amount, data.timestamp, data.features)
        
        # Get fraud probability
        fraud_probability = predictor.predict(features)
        
        # Determine prediction based on threshold
        threshold = settings.fraud_threshold
        prediction = 1 if fraud_probability > threshold else 0
        
        # Create transaction record
        transaction = Transaction(
            amount=data.amount,
            timestamp=data.timestamp,
            features=str(data.features),  # Store as JSON string
            fraud_prediction=prediction,
            fraud_probability=float(fraud_probability),
            created_by=current_user.id,
        )
        
        db.add(transaction)
        db.commit()
        db.refresh(transaction)
        
        logger.info(
            f"Prediction for transaction: {transaction.id}, "
            f"amount={data.amount}, fraud_prob={fraud_probability:.4f}"
        )
        
        # Create alert if fraudulent
        if prediction == 1:
            alert = Alert(
                transaction_id=transaction.id,
                fraud_score=float(fraud_probability),
                threshold=threshold,
                alert_status="pending",
            )
            db.add(alert)
            db.commit()
            logger.warning(f"Fraud alert created for transaction {transaction.id}")
        
        return PredictionResponse(
            transaction_id=transaction.id,
            fraud_prediction=prediction,
            probability=float(fraud_probability),
        )
    
    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        ) from e


@router.get("/history")
def get_transaction_history(
    current_user: User = Depends(),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get transaction history for current user.
    
    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return (max 1000)
    """
    
    if limit > 1000:
        limit = 1000
    
    transactions = db.query(Transaction).filter(
        Transaction.created_by == current_user.id
    ).offset(skip).limit(limit).all()
    
    return {
        "total": db.query(Transaction).filter(
            Transaction.created_by == current_user.id
        ).count(),
        "transactions": transactions
    }


@router.get("/{transaction_id}")
def get_transaction(
    transaction_id: int,
    current_user: User = Depends(),
    db: Session = Depends(get_db)
):
    """Get details of a specific transaction."""
    
    transaction = db.query(Transaction).filter(
        Transaction.id == transaction_id,
        Transaction.created_by == current_user.id
    ).first()
    
    if not transaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found"
        )
    
    return transaction
