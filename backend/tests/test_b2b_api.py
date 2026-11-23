"""
Unit and integration tests for B2B Transaction Scoring API routes and persistence.
Tests score endpoint, idempotency, transaction retrieval, and model metadata.
"""

import asyncio
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.engine import Base
from app.models import User, Transaction, FraudPrediction, Alert
from app.api.routes.transactions import score_transaction, list_transactions, get_transaction
from app.main import get_model_info, health_check_v1
from ml.preprocessing.schema import B2BTransaction, RiskLevel, RecommendedAction
from fastapi import HTTPException


@pytest.fixture
def in_memory_db():
    """Isolated in-memory SQLite database for test execution."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def mock_user(in_memory_db):
    """Creates a mock officer user."""
    user = User(
        username="officer_test",
        email="officer@test.com",
        hashed_password="hash",
        role="fraud_officer",
        is_active=True,
    )
    in_memory_db.add(user)
    in_memory_db.commit()
    in_memory_db.refresh(user)
    return user


class TestB2BScoringAPI:
    """Test suite for /api/v1/transactions/score route."""

    def test_score_legitimate_transaction(self, in_memory_db, mock_user):
        tx = B2BTransaction(
            transaction_reference="tx_test_api_legit_001",
            amount=42.50,
            currency="USD",
            merchant_category="grocery",
            transaction_hour=15,
            country="US",
            customer_age_days=120,
            transactions_last_hour=0,
            average_transaction_amount=40.00,
            failed_transactions_last_day=0,
            device_trust_score=0.95,
        )

        response = score_transaction(tx=tx, current_user=mock_user, db=in_memory_db)
        assert response.transaction_reference == "tx_test_api_legit_001"
        assert response.risk_score < 0.30
        assert response.risk_level == RiskLevel.LOW
        assert response.recommended_action == RecommendedAction.APPROVE

        # Verify database record
        saved_tx = in_memory_db.query(Transaction).filter_by(transaction_reference="tx_test_api_legit_001").first()
        assert saved_tx is not None
        assert saved_tx.amount == 42.50
        assert saved_tx.risk_level == "LOW"

        # Verify prediction record
        pred = in_memory_db.query(FraudPrediction).filter_by(transaction_reference="tx_test_api_legit_001").first()
        assert pred is not None
        assert pred.risk_score == response.risk_score

    def test_score_fraud_creates_alert(self, in_memory_db, mock_user):
        tx = B2BTransaction(
            transaction_reference="tx_test_api_fraud_002",
            amount=12.00,
            currency="USD",
            merchant_category="digital_goods",
            transaction_hour=3,
            country="US",
            customer_age_days=5,
            transactions_last_hour=9,
            average_transaction_amount=30.00,
            failed_transactions_last_day=5,
            device_trust_score=0.08,
        )

        response = score_transaction(tx=tx, current_user=mock_user, db=in_memory_db)
        assert response.risk_score >= 0.70
        assert response.risk_level == RiskLevel.HIGH
        assert response.recommended_action == RecommendedAction.DECLINE

        # Verify alert was created in database
        alert = in_memory_db.query(Alert).filter_by(transaction_reference="tx_test_api_fraud_002").first()
        assert alert is not None
        assert alert.severity == "HIGH"
        assert alert.alert_status == "pending"

    def test_score_idempotency(self, in_memory_db, mock_user):
        tx = B2BTransaction(
            transaction_reference="tx_test_idempotent_003",
            amount=85.00,
            currency="USD",
            merchant_category="retail",
            transaction_hour=12,
            country="US",
            customer_age_days=90,
            transactions_last_hour=1,
            average_transaction_amount=80.00,
            failed_transactions_last_day=0,
            device_trust_score=0.88,
        )

        # First call
        res1 = score_transaction(tx=tx, current_user=mock_user, db=in_memory_db)
        # Second identical call
        res2 = score_transaction(tx=tx, current_user=mock_user, db=in_memory_db)

        assert res1.risk_score == res2.risk_score
        assert res1.recommended_action == res2.recommended_action

        # Ensure no duplicate transactions created in database
        tx_count = in_memory_db.query(Transaction).filter_by(transaction_reference="tx_test_idempotent_003").count()
        assert tx_count == 1

    def test_list_and_get_transaction(self, in_memory_db, mock_user):
        tx = B2BTransaction(
            transaction_reference="tx_test_list_004",
            amount=99.00,
            currency="USD",
            merchant_category="travel",
            transaction_hour=11,
            country="US",
            customer_age_days=200,
            transactions_last_hour=0,
            average_transaction_amount=95.00,
            failed_transactions_last_day=0,
            device_trust_score=0.90,
        )
        score_transaction(tx=tx, current_user=mock_user, db=in_memory_db)

        # List
        result = list_transactions(skip=0, limit=10, current_user=mock_user, db=in_memory_db)
        assert result.total >= 1
        assert any(t.transaction_reference == "tx_test_list_004" for t in result.transactions)

        # Get by reference
        detail = get_transaction(reference_or_id="tx_test_list_004", current_user=mock_user, db=in_memory_db)
        assert detail.transaction_reference == "tx_test_list_004"
        assert detail.amount == 99.00

    def test_get_nonexistent_transaction_raises_404(self, in_memory_db, mock_user):
        with pytest.raises(HTTPException) as exc_info:
            get_transaction(reference_or_id="tx_non_existent", current_user=mock_user, db=in_memory_db)
        assert exc_info.value.status_code == 404

    def test_model_info_and_health_endpoints(self):
        info = get_model_info()
        assert info["is_operational"] is True
        assert info["version"] == "1.0.0"
        assert len(info["registered_rules"]) >= 5

        health = health_check_v1()
        assert health["status"] == "healthy"
        assert health["product_model_operational"] is True
