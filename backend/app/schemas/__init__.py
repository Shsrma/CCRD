"""
Pydantic v2 schemas for request and response validation.
Supports both legacy PCA endpoints and the new B2B real-time fraud scoring API.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Re-export B2B schemas
from ml.preprocessing.schema import (
    B2BTransaction,
    TransactionScoreResponse,
    RiskLevel,
    RecommendedAction,
    MerchantCategory,
    CurrencyCode,
)


# ============================================================================
# LEGACY TRANSACTION SCHEMAS (Backwards Compatibility)
# ============================================================================

class TransactionInput(BaseModel):
    """Legacy schema for PCA fraud prediction request."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "amount": 123.45,
                "timestamp": 1704067200.0,
                "features": [0.5, -0.3, 0.1, 0.2]
            }
        }
    )

    amount: float = Field(..., gt=0, le=1_000_000, description="Transaction amount in currency units")
    timestamp: float = Field(..., description="Unix timestamp of transaction")
    features: List[float] = Field(..., min_length=1, description="Feature vector for ML model")


class PredictionResponse(BaseModel):
    """Legacy schema for fraud prediction response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": 123,
                "fraud_prediction": 1,
                "probability": 0.95
            }
        }
    )

    transaction_id: int
    fraud_prediction: int = Field(..., ge=0, le=1, description="0=Legitimate, 1=Fraudulent")
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


# ============================================================================
# B2B TRANSACTION & AUDIT SCHEMAS (Phase 2)
# ============================================================================

class TransactionDetailResponse(BaseModel):
    """Detailed record of a persisted transaction."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    transaction_reference: Optional[str] = None
    amount: float
    currency: str
    merchant_category: Optional[str] = None
    transaction_hour: Optional[int] = None
    country: Optional[str] = None
    customer_age_days: Optional[int] = None
    transactions_last_hour: Optional[int] = None
    average_transaction_amount: Optional[float] = None
    failed_transactions_last_day: Optional[int] = None
    device_trust_score: Optional[float] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    recommended_action: Optional[str] = None
    reason_codes: Optional[str] = None
    triggered_rules: Optional[str] = None
    model_version: Optional[str] = None
    created_at: datetime


class PaginatedTransactionsResponse(BaseModel):
    """Paginated list of transactions."""
    total: int
    skip: int
    limit: int
    transactions: List[TransactionDetailResponse]


# ============================================================================
# USER & AUTHENTICATION SCHEMAS
# ============================================================================

class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=100)
    email: Optional[str] = Field(None, max_length=255)


class UserCreate(UserBase):
    """Schema for user registration."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "fraud_officer_1",
                "email": "officer@bank.com",
                "password": "SecurePass123"
            }
        }
    )

    password: str = Field(..., min_length=8, max_length=255)

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if not any(char.isupper() for char in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one digit")
        return v


class User(UserBase):
    """Schema for user response."""
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "username": "fraud_officer_1",
                "email": "officer@bank.com",
                "role": "fraud_officer",
                "is_active": True,
                "created_at": "2025-01-01T00:00:00"
            }
        }
    )

    id: int
    role: str
    is_active: bool
    created_at: datetime


class LoginInput(BaseModel):
    """Schema for login request."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "fraud_officer_1",
                "password": "SecurePass123"
            }
        }
    )

    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class Token(BaseModel):
    """Schema for JWT token response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }
    )

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiration time in seconds")


# ============================================================================
# ALERT SCHEMAS
# ============================================================================

class AlertResponse(BaseModel):
    """Schema for fraud alert response."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    transaction_id: int
    transaction_reference: Optional[str] = None
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    threshold: float
    severity: str
    alert_status: str
    created_at: datetime
    updated_at: Optional[datetime] = None


# ============================================================================
# SYSTEM, MODEL & SETTINGS SCHEMAS
# ============================================================================

class ModelInfoResponse(BaseModel):
    """Information about active ML model and configuration."""
    model_name: str
    version: str
    is_operational: bool
    threshold_config: Dict[str, float]
    input_features: List[str]
    registered_rules: List[Dict[str, Any]]
    evaluation_metrics: Optional[Dict[str, Any]] = None
    synthetic_disclaimer: str


class GlobalSettings(BaseModel):
    """Schema for global application settings."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fraud_threshold": 0.5,
                "language": "en",
                "timezone": "UTC"
            }
        }
    )

    fraud_threshold: float = Field(..., ge=0.0, le=1.0)
    language: str = Field(default="en")
    timezone: str = Field(default="UTC")


class SettingsUpdate(BaseModel):
    """Schema for updating individual settings."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "fraud_threshold",
                "value": 0.6
            }
        }
    )

    setting_type: str = Field(..., alias="type")
    value: float | str = Field(...)


# ============================================================================
# MULTI-TENANT & API KEY SCHEMAS (Phase 3)
# ============================================================================

class OrganizationCreate(BaseModel):
    """Schema for tenant/merchant organization registration."""
    name: str = Field(..., min_length=2, max_length=100)
    slug: str = Field(..., min_length=2, max_length=100)
    plan: Optional[str] = Field(default="standard")


class OrganizationResponse(BaseModel):
    """Schema for organization details."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    plan: str
    is_active: bool
    created_at: datetime


class TenantConfigUpdate(BaseModel):
    """Schema for updating tenant-specific thresholds and rules."""
    low_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    high_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    rules_config: Optional[str] = None
    webhook_url: Optional[str] = None


class TenantConfigResponse(BaseModel):
    """Schema for tenant fraud risk configuration."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    organization_id: int
    low_threshold: float
    high_threshold: float
    rules_config: str
    webhook_url: Optional[str] = None
    updated_at: Optional[datetime] = None


class ApiKeyCreate(BaseModel):
    """Schema for requesting a new B2B API Key."""
    key_name: str = Field(default="Live Ingestion Key", max_length=100)
    environment: Optional[str] = Field(default="live", pattern="^(live|test)$")


class ApiKeyResponse(BaseModel):
    """Masked API Key metadata for UI management."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    key_name: str
    key_prefix: str
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None


class ApiKeyCreatedResponse(BaseModel):
    """Response returned once when an API Key is freshly minted."""
    id: int
    key_name: str
    key_prefix: str
    raw_key: str = Field(..., description="Full secret API key. Store safely, will not be shown again.")
    created_at: datetime

