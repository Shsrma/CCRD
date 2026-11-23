"""
Unit tests for CCRD Rules Engine and individual fraud rules.
"""

import pytest
from ml.preprocessing.schema import B2BTransaction, MerchantCategory
from backend.app.rules_engine import (
    RulesEngine,
    HighAmountRule,
    HighVelocityRule,
    UnusualNightHourRule,
    MultipleFailedAttemptsRule,
    UntrustedDeviceRule,
    NewAccountHighRiskCategoryRule,
)


@pytest.fixture
def baseline_transaction():
    """Returns a normal, low-risk transaction."""
    return B2BTransaction(
        transaction_reference="tx_test_normal_1234",
        amount=45.00,
        currency="USD",
        merchant_category="grocery",
        transaction_hour=14,
        country="US",
        customer_age_days=180,
        transactions_last_hour=1,
        average_transaction_amount=50.00,
        failed_transactions_last_day=0,
        device_trust_score=0.90,
    )


class TestIndividualRules:
    """Tests for each discrete fraud rule."""

    def test_high_amount_rule_triggers_on_absolute(self, baseline_transaction):
        rule = HighAmountRule(absolute_threshold=5000.0)
        baseline_transaction.amount = 5500.00
        result = rule.evaluate(baseline_transaction)
        assert result is not None
        assert result.reason_code == "RULE_ABSOLUTE_AMOUNT_EXCEEDED"
        assert result.severity == "HIGH"

    def test_high_amount_rule_triggers_on_ratio(self, baseline_transaction):
        rule = HighAmountRule(ratio_threshold=8.0)
        baseline_transaction.average_transaction_amount = 40.00
        baseline_transaction.amount = 400.00  # 10x
        result = rule.evaluate(baseline_transaction)
        assert result is not None
        assert result.reason_code == "RULE_AMOUNT_SPIKE_VS_BASELINE"
        assert result.severity == "MEDIUM"

    def test_high_velocity_rule(self, baseline_transaction):
        rule = HighVelocityRule(velocity_threshold=5)
        baseline_transaction.transactions_last_hour = 6
        result = rule.evaluate(baseline_transaction)
        assert result is not None
        assert result.reason_code == "RULE_VELOCITY_LIMIT_EXCEEDED"

    def test_unusual_night_hour_rule(self, baseline_transaction):
        rule = UnusualNightHourRule(amount_threshold=300.0)
        baseline_transaction.transaction_hour = 3
        baseline_transaction.amount = 450.00
        result = rule.evaluate(baseline_transaction)
        assert result is not None
        assert result.reason_code == "RULE_UNUSUAL_NIGHT_HOURS"

    def test_multiple_failed_attempts_rule(self, baseline_transaction):
        rule = MultipleFailedAttemptsRule(failure_threshold=3)
        baseline_transaction.failed_transactions_last_day = 4
        result = rule.evaluate(baseline_transaction)
        assert result is not None
        assert result.reason_code == "RULE_MULTIPLE_FAILED_ATTEMPTS"

    def test_untrusted_device_rule(self, baseline_transaction):
        rule = UntrustedDeviceRule(trust_threshold=0.25)
        baseline_transaction.device_trust_score = 0.12
        result = rule.evaluate(baseline_transaction)
        assert result is not None
        assert result.reason_code == "RULE_LOW_DEVICE_TRUST"

    def test_new_account_high_risk_category_rule(self, baseline_transaction):
        rule = NewAccountHighRiskCategoryRule(max_age_days=14, min_amount=200.0)
        baseline_transaction.customer_age_days = 4
        baseline_transaction.merchant_category = "crypto"
        baseline_transaction.amount = 350.00
        result = rule.evaluate(baseline_transaction)
        assert result is not None
        assert result.reason_code == "RULE_NEW_ACCOUNT_HIGH_RISK_SECTOR"


class TestRulesEngineCoordinator:
    """Tests for RulesEngine composite evaluation."""

    def test_normal_transaction_triggers_no_rules(self, baseline_transaction):
        engine = RulesEngine()
        results, modifier = engine.evaluate(baseline_transaction)
        assert len(results) == 0
        assert modifier == 0.0

    def test_multi_rule_trigger(self, baseline_transaction):
        engine = RulesEngine()
        baseline_transaction.transactions_last_hour = 6
        baseline_transaction.device_trust_score = 0.10
        baseline_transaction.failed_transactions_last_day = 4

        results, modifier = engine.evaluate(baseline_transaction)
        assert len(results) >= 3
        assert modifier > 0.40
