"""
B2B Transaction Schema definitions using Pydantic v2.
Defines models for raw incoming merchant transactions, normalized vectors, and scoring outputs.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, ConfigDict


class MerchantCategory(str, Enum):
    """Supported merchant category business sectors."""
    RETAIL = "retail"
    GROCERY = "grocery"
    TRAVEL = "travel"
    ELECTRONICS = "electronics"
    DIGITAL_GOODS = "digital_goods"
    GAMING = "gaming"
    CRYPTO = "crypto"
    TRANSFER = "transfer"


class CurrencyCode(str, Enum):
    """Supported ISO 4217 currency codes."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"
    INR = "INR"
    JPY = "JPY"


class B2BTransaction(BaseModel):
    """
    Standard interpretable B2B transaction authorization request.
    Contains merchant, payment, behavioral, and device signals.
    """
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "transaction_reference": "tx_a1b2c3d4e5f6",
                "amount": 249.99,
                "currency": "USD",
                "merchant_category": "electronics",
                "transaction_hour": 14,
                "country": "US",
                "customer_age_days": 180,
                "transactions_last_hour": 1,
                "average_transaction_amount": 65.50,
                "failed_transactions_last_day": 0,
                "device_trust_score": 0.92,
            }
        }
    )

    transaction_reference: str = Field(
        ...,
        min_length=6,
        max_length=64,
        description="Unique merchant authorization reference identifier"
    )
    amount: float = Field(
        ...,
        gt=0.0,
        le=1_000_000.0,
        description="Transaction amount in major currency units"
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="ISO 4217 3-letter currency code"
    )
    merchant_category: MerchantCategory = Field(
        ...,
        description="Business sector of the merchant"
    )
    transaction_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Local hour of transaction (0-23)"
    )
    country: str = Field(
        default="US",
        min_length=2,
        max_length=2,
        description="2-letter ISO 3166-1 alpha-2 country code"
    )
    customer_age_days: int = Field(
        ...,
        ge=0,
        description="Account tenure in days"
    )
    transactions_last_hour: int = Field(
        ...,
        ge=0,
        le=1000,
        description="Count of transactions on account/card in the last hour"
    )
    average_transaction_amount: float = Field(
        ...,
        ge=0.0,
        description="Customer historical 30-day average transaction amount"
    )
    failed_transactions_last_day: int = Field(
        ...,
        ge=0,
        le=500,
        description="Count of failed/declined transactions on account in past 24 hours"
    )
    device_trust_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Device reputation and biometric confidence score (0.0 to 1.0)"
    )

    @field_validator("country")
    @classmethod
    def normalize_country(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("transaction_reference")
    @classmethod
    def validate_reference(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("transaction_reference cannot be empty")
        return v


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RecommendedAction(str, Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


class TransactionScoreResponse(BaseModel):
    """
    Response schema for transaction fraud scoring.
    """
    model_config = ConfigDict(use_enum_values=True)

    transaction_reference: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated risk index between 0.0 and 1.0")
    risk_level: RiskLevel
    recommended_action: RecommendedAction
    reason_codes: List[str] = Field(default_factory=list, description="Audit reason codes explaining decision")
    triggered_rules: List[str] = Field(default_factory=list, description="Names of triggered deterministic rules")
    model_version: str = Field(..., description="Version of the ML model used for scoring")
    is_synthetic_evaluation: bool = Field(default=True, description="Disclaimer flag indicating synthetic evaluation")
