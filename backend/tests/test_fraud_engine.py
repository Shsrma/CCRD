"""
Unit tests for CCRD Fraud Decision Engine.
"""

import pytest
from ml.preprocessing.schema import (
    B2BTransaction,
    RiskLevel,
    RecommendedAction,
)
from backend.app.fraud_engine import FraudDecisionEngine


@pytest.fixture
def fraud_engine():
    """Initializes FraudDecisionEngine loading the trained artifact."""
    return FraudDecisionEngine()


class TestFraudDecisionEngine:
    """Test suite for hybrid fraud scoring."""

    def test_engine_is_operational(self, fraud_engine):
        assert fraud_engine.is_operational() is True
        assert fraud_engine.version == "1.0.0"

    def test_normal_transaction_approved(self, fraud_engine):
        tx = B2BTransaction(
            transaction_reference="tx_test_legit_1001",
            amount=35.50,
            currency="USD",
            merchant_category="grocery",
            transaction_hour=14,
            country="US",
            customer_age_days=300,
            transactions_last_hour=0,
            average_transaction_amount=40.00,
            failed_transactions_last_day=0,
            device_trust_score=0.92,
        )
        decision = fraud_engine.evaluate_transaction(tx)

        assert decision.risk_score < 0.30
        assert decision.risk_level == RiskLevel.LOW
        assert decision.recommended_action == RecommendedAction.APPROVE
        assert decision.is_synthetic_evaluation is True

    def test_severe_velocity_fraud_declined(self, fraud_engine):
        tx = B2BTransaction(
            transaction_reference="tx_test_fraud_2002",
            amount=15.00,
            currency="USD",
            merchant_category="digital_goods",
            transaction_hour=3,
            country="US",
            customer_age_days=10,
            transactions_last_hour=8,
            average_transaction_amount=45.00,
            failed_transactions_last_day=4,
            device_trust_score=0.10,
        )
        decision = fraud_engine.evaluate_transaction(tx)

        assert decision.risk_score >= 0.70
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.recommended_action == RecommendedAction.DECLINE
        assert len(decision.triggered_rules) >= 2
        assert any("RULE_" in code for code in decision.reason_codes)

    def test_account_takeover_spike_declined(self, fraud_engine):
        tx = B2BTransaction(
            transaction_reference="tx_test_ato_3003",
            amount=4200.00,
            currency="USD",
            merchant_category="electronics",
            transaction_hour=2,
            country="MX",
            customer_age_days=400,
            transactions_last_hour=2,
            average_transaction_amount=50.00,
            failed_transactions_last_day=1,
            device_trust_score=0.08,
        )
        decision = fraud_engine.evaluate_transaction(tx)

        assert decision.risk_score >= 0.70
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.recommended_action == RecommendedAction.DECLINE
        assert "RULE_AMOUNT_SPIKE_VS_BASELINE" in decision.reason_codes or "RULE_LOW_DEVICE_TRUST" in decision.reason_codes

    def test_borderline_transaction_flagged_for_review(self, fraud_engine):
        tx = B2BTransaction(
            transaction_reference="tx_test_border_4004",
            amount=850.00,
            currency="USD",
            merchant_category="electronics",
            transaction_hour=16,
            country="US",
            customer_age_days=60,
            transactions_last_hour=3,
            average_transaction_amount=100.00,
            failed_transactions_last_day=1,
            device_trust_score=0.45,
        )
        decision = fraud_engine.evaluate_transaction(tx)

        assert 0.20 <= decision.risk_score <= 1.0
        assert decision.recommended_action in [RecommendedAction.REVIEW, RecommendedAction.DECLINE]
