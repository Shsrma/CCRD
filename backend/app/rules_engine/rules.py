"""
Concrete deterministic fraud rules for CCRD Fraud Detection Platform.
"""

from typing import Optional
from backend.app.rules_engine.base import BaseRule, RuleResult


class HighAmountRule(BaseRule):
    """Flags transactions that exceed an absolute ceiling or spike far above historical baseline."""

    def __init__(
        self,
        absolute_threshold: float = 5000.0,
        ratio_threshold: float = 8.0,
        enabled: bool = True
    ):
        super().__init__(enabled=enabled)
        self.absolute_threshold = absolute_threshold
        self.ratio_threshold = ratio_threshold

    @property
    def name(self) -> str:
        return "HIGH_AMOUNT_ANOMALY"

    @property
    def description(self) -> str:
        return f"Transaction exceeds ${self.absolute_threshold:,.2f} or is {self.ratio_threshold}x higher than 30-day average."

    def evaluate(self, transaction) -> Optional[RuleResult]:
        if not self.enabled:
            return None

        amount = float(transaction.amount)
        avg = float(transaction.average_transaction_amount)
        ratio = amount / (avg + 1.0) if avg >= 0 else 0.0

        if amount >= self.absolute_threshold:
            return RuleResult(
                rule_name=self.name,
                reason_code="RULE_ABSOLUTE_AMOUNT_EXCEEDED",
                severity="HIGH",
                description=f"Transaction amount ${amount:.2f} exceeds absolute threshold of ${self.absolute_threshold:.2f}",
                risk_boost=0.35,
                metadata={"amount": amount, "threshold": self.absolute_threshold},
            )

        if ratio >= self.ratio_threshold and amount >= 300.0:
            return RuleResult(
                rule_name=self.name,
                reason_code="RULE_AMOUNT_SPIKE_VS_BASELINE",
                severity="MEDIUM",
                description=f"Transaction amount ${amount:.2f} is {ratio:.1f}x customer's average ${avg:.2f}",
                risk_boost=0.25,
                metadata={"amount": amount, "average": avg, "ratio": ratio},
            )

        return None


class HighVelocityRule(BaseRule):
    """Flags rapid bursts of transaction attempts on the same card or account."""

    def __init__(self, velocity_threshold: int = 5, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.velocity_threshold = velocity_threshold

    @property
    def name(self) -> str:
        return "HIGH_VELOCITY_BURST"

    @property
    def description(self) -> str:
        return f"More than {self.velocity_threshold} transactions attempted in trailing 60 minutes."

    def evaluate(self, transaction) -> Optional[RuleResult]:
        if not self.enabled:
            return None

        velocity = int(transaction.transactions_last_hour)
        if velocity >= self.velocity_threshold:
            return RuleResult(
                rule_name=self.name,
                reason_code="RULE_VELOCITY_LIMIT_EXCEEDED",
                severity="HIGH" if velocity >= 8 else "MEDIUM",
                description=f"High hourly velocity: {velocity} transactions attempted in past hour",
                risk_boost=0.30 if velocity >= 8 else 0.20,
                metadata={"transactions_last_hour": velocity, "threshold": self.velocity_threshold},
            )
        return None


class UnusualNightHourRule(BaseRule):
    """Flags large transactions occurring between 01:00 and 05:00 local time."""

    def __init__(self, amount_threshold: float = 300.0, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.amount_threshold = amount_threshold
        self.night_hours = {1, 2, 3, 4, 5}

    @property
    def name(self) -> str:
        return "UNUSUAL_NIGHT_TRANSACTION"

    @property
    def description(self) -> str:
        return "Elevated purchase occurring during high-risk off-peak night hours (01:00-05:00)."

    def evaluate(self, transaction) -> Optional[RuleResult]:
        if not self.enabled:
            return None

        hour = int(transaction.transaction_hour)
        amount = float(transaction.amount)

        if hour in self.night_hours and amount >= self.amount_threshold:
            return RuleResult(
                rule_name=self.name,
                reason_code="RULE_UNUSUAL_NIGHT_HOURS",
                severity="MEDIUM",
                description=f"Transaction of ${amount:.2f} executed at off-peak hour {hour:02d}:00",
                risk_boost=0.15,
                metadata={"transaction_hour": hour, "amount": amount},
            )
        return None


class MultipleFailedAttemptsRule(BaseRule):
    """Flags accounts with repeated recent authorization failures (card testing / credential stuffing)."""

    def __init__(self, failure_threshold: int = 3, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.failure_threshold = failure_threshold

    @property
    def name(self) -> str:
        return "REPEATED_FAILED_ATTEMPTS"

    @property
    def description(self) -> str:
        return f"{self.failure_threshold} or more authorization failures in past 24 hours."

    def evaluate(self, transaction) -> Optional[RuleResult]:
        if not self.enabled:
            return None

        fails = int(transaction.failed_transactions_last_day)
        if fails >= self.failure_threshold:
            return RuleResult(
                rule_name=self.name,
                reason_code="RULE_MULTIPLE_FAILED_ATTEMPTS",
                severity="HIGH",
                description=f"Account has {fails} failed transactions in trailing 24 hours",
                risk_boost=0.30,
                metadata={"failed_transactions_last_day": fails, "threshold": self.failure_threshold},
            )
        return None


class UntrustedDeviceRule(BaseRule):
    """Flags transactions originating from low-reputation or unrecognized devices."""

    def __init__(self, trust_threshold: float = 0.25, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.trust_threshold = trust_threshold

    @property
    def name(self) -> str:
        return "UNTRUSTED_DEVICE"

    @property
    def description(self) -> str:
        return f"Device trust score below {self.trust_threshold} (possible emulator or proxy)."

    def evaluate(self, transaction) -> Optional[RuleResult]:
        if not self.enabled:
            return None

        trust = float(transaction.device_trust_score)
        if trust <= self.trust_threshold:
            return RuleResult(
                rule_name=self.name,
                reason_code="RULE_LOW_DEVICE_TRUST",
                severity="HIGH" if trust <= 0.10 else "MEDIUM",
                description=f"Low device trust score: {trust:.2f} (threshold: {self.trust_threshold:.2f})",
                risk_boost=0.25 if trust <= 0.10 else 0.15,
                metadata={"device_trust_score": trust, "threshold": self.trust_threshold},
            )
        return None


class NewAccountHighRiskCategoryRule(BaseRule):
    """Flags newly created accounts (< 14 days) transacting in high-liquidity / high-risk categories."""

    def __init__(self, max_age_days: int = 14, min_amount: float = 200.0, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.max_age_days = max_age_days
        self.min_amount = min_amount
        self.high_risk_categories = {"crypto", "gaming", "digital_goods", "transfer"}

    @property
    def name(self) -> str:
        return "NEW_ACCOUNT_HIGH_RISK_CATEGORY"

    @property
    def description(self) -> str:
        return "Account younger than 14 days purchasing in high-liquidity or high-risk category."

    def evaluate(self, transaction) -> Optional[RuleResult]:
        if not self.enabled:
            return None

        age = int(transaction.customer_age_days)
        category = str(transaction.merchant_category).lower()
        amount = float(transaction.amount)

        if age <= self.max_age_days and category in self.high_risk_categories and amount >= self.min_amount:
            return RuleResult(
                rule_name=self.name,
                reason_code="RULE_NEW_ACCOUNT_HIGH_RISK_SECTOR",
                severity="HIGH",
                description=f"New account ({age} days old) transacting ${amount:.2f} in high-risk '{category}'",
                risk_boost=0.25,
                metadata={"customer_age_days": age, "category": category, "amount": amount},
            )
        return None
