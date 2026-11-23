"""
Transaction scoring and history API endpoints for CCRD Real-Time Fraud Intelligence Platform.
Implements POST /score, GET /, GET /{id_or_ref}, and legacy POST /predict.
"""

import json
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logger import logger
from app.database.engine import get_db
from app.models import User, Transaction, FraudPrediction, Alert, Organization
from app.api.dependencies import (
    get_current_user,
    get_current_user_optional,
    get_tenant_context,
    TenantContext,
    get_or_create_default_org,
)
from app.schemas import (
    TransactionInput,
    PredictionResponse,
    B2BTransaction,
    TransactionScoreResponse,
    TransactionDetailResponse,
    PaginatedTransactionsResponse,
    RiskLevel,
    RecommendedAction,
)
from backend.app.fraud_engine import FraudDecisionEngine
from app.ml import ModelPredictor, preprocess_input

settings = get_settings()
router = APIRouter()

# Initialize Fraud Decision Engine (loads product ML pipeline & rules)
fraud_engine = FraudDecisionEngine()

# Initialize Legacy ML model if available
try:
    legacy_predictor = ModelPredictor(settings.model_path, settings.scaler_path)
except Exception:
    legacy_predictor = None


# ============================================================================
# REAL-TIME FRAUD SCORING (B2B SaaS Core)
# ============================================================================

@router.post(
    "/score",
    response_model=TransactionScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score B2B Transaction Risk in Real Time",
    description="Evaluates a financial transaction in real-time through deterministic rules and calibrated ML inference."
)
def score_transaction(
    tx: B2BTransaction,
    current_user: Optional[User] = Depends(get_current_user_optional),
    context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """
    10-step real-time fraud scoring pipeline with multi-tenant isolation:
    1. Input validation & normalization (Pydantic v2)
    2. Tenant resolution & threshold configuration
    3. Idempotency check on transaction_reference
    4. Rules engine execution
    5. Calibrated ML inference
    6. Composite risk scoring
    7. Decision thresholds application (using tenant-specific thresholds)
    8. Reason codes generation
    9. Transaction & Prediction persistence with tenant isolation
    10. Alert dispatch for suspicious authorizations
    """
    # Resolve active tenant and user safely against direct Python test calls
    user = current_user if isinstance(current_user, User) else None
    ctx = context if isinstance(context, TenantContext) else None

    if ctx is not None:
        org_id = ctx.organization.id if ctx.organization else None
        user_id = ctx.user.id if ctx.user else (user.id if user else None)
        low_th = ctx.low_threshold
        high_th = ctx.high_threshold
    elif user is not None:
        org_id = user.organization_id
        user_id = user.id
        low_th = fraud_engine.low_threshold
        high_th = fraud_engine.high_threshold
    else:
        user_id = None
        org_id = None
        low_th = fraud_engine.low_threshold
        high_th = fraud_engine.high_threshold

    if org_id is None:
        default_org = get_or_create_default_org(db)
        org_id = default_org.id

    # Idempotency check: if transaction_reference was already scored, return existing prediction
    existing_tx = db.query(Transaction).filter(
        Transaction.transaction_reference == tx.transaction_reference
    ).first()

    if existing_tx and existing_tx.risk_score is not None:
        logger.info(f"Idempotent response returned for transaction {tx.transaction_reference}")
        try:
            reason_codes = json.loads(existing_tx.reason_codes) if existing_tx.reason_codes else []
            triggered_rules = json.loads(existing_tx.triggered_rules) if existing_tx.triggered_rules else []
        except Exception:
            reason_codes = []
            triggered_rules = []

        return TransactionScoreResponse(
            transaction_reference=existing_tx.transaction_reference,
            risk_score=float(existing_tx.risk_score),
            risk_level=RiskLevel(existing_tx.risk_level),
            recommended_action=RecommendedAction(existing_tx.recommended_action),
            reason_codes=reason_codes,
            triggered_rules=triggered_rules,
            model_version=existing_tx.model_version or fraud_engine.version,
            is_synthetic_evaluation=True,
        )

    # Execute evaluation through Fraud Decision Engine using tenant thresholds
    decision = fraud_engine.evaluate_transaction(
        tx=tx,
        low_threshold=low_th,
        high_threshold=high_th,
    )

    # Persist transaction record
    risk_level_str = decision.risk_level.value if hasattr(decision.risk_level, "value") else str(decision.risk_level)
    action_str = decision.recommended_action.value if hasattr(decision.recommended_action, "value") else str(decision.recommended_action)

    new_tx = Transaction(
        transaction_reference=tx.transaction_reference,
        organization_id=org_id,
        amount=tx.amount,
        currency=tx.currency,
        merchant_category=tx.merchant_category,
        transaction_hour=tx.transaction_hour,
        country=tx.country,
        customer_age_days=tx.customer_age_days,
        transactions_last_hour=tx.transactions_last_hour,
        average_transaction_amount=tx.average_transaction_amount,
        failed_transactions_last_day=tx.failed_transactions_last_day,
        device_trust_score=tx.device_trust_score,
        risk_score=decision.risk_score,
        risk_level=risk_level_str,
        recommended_action=action_str,
        reason_codes=json.dumps(decision.reason_codes),
        triggered_rules=json.dumps(decision.triggered_rules),
        model_version=decision.model_version,
        created_by=user_id,
    )

    try:
        db.add(new_tx)
        db.commit()
        db.refresh(new_tx)

        # Persist immutable prediction audit record
        prediction_record = FraudPrediction(
            transaction_id=new_tx.id,
            transaction_reference=tx.transaction_reference,
            organization_id=org_id,
            risk_score=decision.risk_score,
            risk_level=risk_level_str,
            recommended_action=action_str,
            reason_codes=json.dumps(decision.reason_codes),
            triggered_rules=json.dumps(decision.triggered_rules),
            model_version=decision.model_version,
        )
        db.add(prediction_record)

        # Trigger alert if transaction requires REVIEW or DECLINE
        if action_str in ["REVIEW", "DECLINE"]:
            alert_severity = "HIGH" if action_str == "DECLINE" else "MEDIUM"
            alert = Alert(
                transaction_id=new_tx.id,
                transaction_reference=tx.transaction_reference,
                organization_id=org_id,
                fraud_score=decision.risk_score,
                threshold=low_th,
                severity=alert_severity,
                alert_status="pending",
            )
            db.add(alert)
            logger.warning(
                f"Fraud alert triggered for tx {tx.transaction_reference} | "
                f"Action: {action_str} | Score: {decision.risk_score:.4f}"
            )

        db.commit()
        return decision

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to persist transaction scoring: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record transaction evaluation"
        ) from e


# ============================================================================
# TRANSACTION HISTORY & RETRIEVAL (Tenant Isolated)
# ============================================================================

@router.get(
    "",
    response_model=PaginatedTransactionsResponse,
    summary="List scored transactions with filtering and pagination"
)
def list_transactions(
    skip: int = 0,
    limit: int = 50,
    risk_level: Optional[str] = None,
    action: Optional[str] = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
    context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Retrieve paginated list of transactions with strict tenant data isolation."""
    query = db.query(Transaction)

    # Multi-tenant isolation: resolve tenant id safely
    user = current_user if isinstance(current_user, User) else None
    ctx = context if isinstance(context, TenantContext) else None

    org_id = None
    role = "analyst"
    if ctx is not None:
        role = ctx.role
        org_id = ctx.organization.id if ctx.organization else None
    elif user is not None:
        role = user.role
        org_id = user.organization_id

    if role != "superadmin" and org_id is not None:
        query = query.filter((Transaction.organization_id == org_id) | (Transaction.organization_id.is_(None)))

    if risk_level and isinstance(risk_level, str):
        query = query.filter(Transaction.risk_level == risk_level.upper())
    if action and isinstance(action, str):
        query = query.filter(Transaction.recommended_action == action.upper())

    total = query.count()
    records = query.order_by(Transaction.created_at.desc()).offset(skip).limit(limit).all()

    return PaginatedTransactionsResponse(
        total=total,
        skip=skip,
        limit=limit,
        transactions=[TransactionDetailResponse.model_validate(r) for r in records],
    )


@router.get(
    "/{reference_or_id}",
    response_model=TransactionDetailResponse,
    summary="Get single transaction by ID or reference"
)
def get_transaction(
    reference_or_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Retrieve details of a single transaction by integer ID or string reference with tenant isolation."""
    query = db.query(Transaction)
    if reference_or_id.isdigit():
        tx = query.filter(Transaction.id == int(reference_or_id)).first()
    else:
        tx = query.filter(Transaction.transaction_reference == reference_or_id).first()

    if not tx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction '{reference_or_id}' not found"
        )

    # Multi-tenant access check
    user = current_user if isinstance(current_user, User) else None
    ctx = context if isinstance(context, TenantContext) else None

    org_id = ctx.organization.id if ctx and ctx.organization else (user.organization_id if user else None)
    role = ctx.role if ctx else (user.role if user else "analyst")
    if role != "superadmin" and org_id is not None and tx.organization_id is not None and tx.organization_id != org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: transaction belongs to another organization"
        )

    return TransactionDetailResponse.model_validate(tx)


# ============================================================================
# LEGACY ENDPOINTS (Backwards Compatibility)
# ============================================================================

@router.post("/predict", response_model=PredictionResponse, deprecated=True)
def predict_legacy_transaction(
    data: TransactionInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Legacy PCA prediction endpoint."""
    if legacy_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Legacy PCA ML model not available; please use /api/v1/transactions/score"
        )

    features = preprocess_input(data.amount, data.timestamp, data.features)
    prob = legacy_predictor.predict(features)
    prediction = 1 if prob > settings.fraud_threshold else 0

    tx = Transaction(
        amount=data.amount,
        timestamp=data.timestamp,
        features=str(data.features),
        fraud_prediction=prediction,
        fraud_probability=float(prob),
        created_by=current_user.id,
    )
    db.add(tx)
    db.commit()
    db.refresh(tx)

    return PredictionResponse(
        transaction_id=tx.id,
        fraud_prediction=prediction,
        probability=float(prob),
    )


@router.get("/history", deprecated=True)
def get_legacy_transaction_history(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Legacy transaction history endpoint."""
    return list_transactions(skip=skip, limit=limit, current_user=current_user, db=db)
