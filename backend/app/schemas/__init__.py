"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


# ============================================================================
# TRANSACTION SCHEMAS
# ============================================================================

class TransactionInput(BaseModel):
    """Schema for fraud prediction request."""
    
    amount: float = Field(..., gt=0, description="Transaction amount in currency units")
    timestamp: float = Field(..., description="Unix timestamp of transaction")
    features: List[float] = Field(..., min_items=1, description="Feature vector for ML model")
    
    @validator("amount")
    def validate_amount(cls, v):
        if v > 1_000_000:
            raise ValueError("Transaction amount exceeds maximum threshold")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "amount": 123.45,
                "timestamp": 1704067200.0,
                "features": [0.5, -0.3, 0.1, 0.2]
            }
        }


class PredictionResponse(BaseModel):
    """Schema for fraud prediction response."""
    
    transaction_id: int
    fraud_prediction: int = Field(..., ge=0, le=1, description="0=Legitimate, 1=Fraudulent")
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": 123,
                "fraud_prediction": 1,
                "probability": 0.95
            }
        }


# ============================================================================
# USER SCHEMAS
# ============================================================================

class UserBase(BaseModel):
    """Base user schema."""
    
    username: str = Field(..., min_length=3, max_length=100)
    email: Optional[str] = Field(None, max_length=255)


class UserCreate(UserBase):
    """Schema for user registration."""
    
    password: str = Field(..., min_length=8, max_length=255)
    
    @validator("password")
    def validate_password_strength(cls, v):
        if not any(char.isupper() for char in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one digit")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "username": "fraud_officer_1",
                "email": "officer@bank.com",
                "password": "SecurePass123"
            }
        }


class User(UserBase):
    """Schema for user response."""
    
    id: int
    role: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True  # Support ORM mode
        schema_extra = {
            "example": {
                "id": 1,
                "username": "fraud_officer_1",
                "email": "officer@bank.com",
                "role": "fraud_officer",
                "is_active": True,
                "created_at": "2025-01-01T00:00:00"
            }
        }


# ============================================================================
# AUTHENTICATION SCHEMAS
# ============================================================================

class LoginInput(BaseModel):
    """Schema for login request."""
    
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)
    
    class Config:
        schema_extra = {
            "example": {
                "username": "fraud_officer_1",
                "password": "SecurePass123"
            }
        }


class Token(BaseModel):
    """Schema for JWT token response."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiration time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }


# ============================================================================
# ALERT SCHEMAS
# ============================================================================

class AlertResponse(BaseModel):
    """Schema for fraud alert response."""
    
    id: int
    transaction_id: int
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    threshold: float
    alert_status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# SETTINGS SCHEMAS
# ============================================================================

class GlobalSettings(BaseModel):
    """Schema for global application settings."""
    
    fraud_threshold: float = Field(..., ge=0.0, le=1.0)
    language: str = Field(default="en")
    timezone: str = Field(default="UTC")
    
    class Config:
        schema_extra = {
            "example": {
                "fraud_threshold": 0.5,
                "language": "en",
                "timezone": "UTC"
            }
        }


class SettingsUpdate(BaseModel):
    """Schema for updating individual settings."""
    
    setting_type: str = Field(..., alias="type")
    value: float | str = Field(...)
    
    class Config:
        schema_extra = {
            "example": {
                "type": "fraud_threshold",
                "value": 0.6
            }
        }
