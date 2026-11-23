"""
SQLAlchemy ORM models for User, Organization, TenantConfig, ApiKey, Transaction, FraudPrediction, and Alert entities.
Supports B2B multi-tenant SaaS architecture, data isolation, and legacy PCA transactions.
"""

from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.engine import Base


class Organization(Base):
    """
    Multi-tenant Organization (Merchant / Enterprise Customer).
    Provides strict logical data isolation for all transactions, alerts, and configurations.
    """
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    slug = Column(String(100), unique=True, index=True, nullable=False)
    plan = Column(String(50), default="standard", nullable=False)  # standard, enterprise
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    users = relationship("User", back_populates="organization")
    config = relationship("TenantConfig", back_populates="organization", uselist=False, cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="organization", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="organization")

    def __repr__(self):
        return f"<Organization(id={self.id}, slug={self.slug}, plan={self.plan})>"


class TenantConfig(Base):
    """
    Tenant-specific fraud rules configuration and decision thresholds.
    Allows organizations to customize risk boundaries and webhook alerts.
    """
    __tablename__ = "tenant_configs"

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), unique=True, nullable=False, index=True)
    low_threshold = Column(Float, default=0.30, nullable=False)
    high_threshold = Column(Float, default=0.70, nullable=False)
    rules_config = Column(Text, default="{}", nullable=False)  # JSON-encoded rule parameters
    webhook_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    organization = relationship("Organization", back_populates="config")


class ApiKey(Base):
    """
    Hashed API Key for programmatic B2B transaction ingestion via X-API-Key header.
    Stores only cryptographically secure SHA-256 hashes and key prefix for UI display.
    """
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    key_name = Column(String(100), nullable=False)
    key_prefix = Column(String(20), nullable=False)  # e.g. "ccrd_live_9a7f"
    hashed_key = Column(String(64), unique=True, index=True, nullable=False)  # SHA-256 hex
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)

    organization = relationship("Organization", back_populates="api_keys")


class User(Base):
    """User account model supporting multi-tenant RBAC (Superadmin, Tenant Admin, Analyst)."""
    
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True, index=True)
    role = Column(String(50), default="analyst", nullable=False)  # superadmin, tenant_admin, analyst, viewer
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # OTP secret for multi-factor authentication
    otp_secret = Column(String(255), nullable=True)
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    transactions = relationship("Transaction", back_populates="created_by_user")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, role={self.role}, org_id={self.organization_id})>"


class Transaction(Base):
    """Credit card transaction record (supports B2B multi-tenant schema & legacy PCA)."""
    
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_reference = Column(String(64), unique=True, index=True, nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    
    # Financial & contextual attributes
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    merchant_category = Column(String(50), nullable=True)
    transaction_hour = Column(Integer, nullable=True)
    country = Column(String(2), default="US", nullable=True)
    
    # Behavioral & velocity signals
    customer_age_days = Column(Integer, nullable=True)
    transactions_last_hour = Column(Integer, nullable=True)
    average_transaction_amount = Column(Float, nullable=True)
    failed_transactions_last_day = Column(Integer, nullable=True)
    device_trust_score = Column(Float, nullable=True)

    # Legacy fields for backward compatibility
    timestamp = Column(Float, nullable=True)  # Unix timestamp
    features = Column(String, nullable=True)  # Legacy feature string
    fraud_prediction = Column(Integer, nullable=True)  # 0=Legitimate, 1=Fraud
    fraud_probability = Column(Float, nullable=True)

    # Scoring results
    risk_score = Column(Float, nullable=True)
    risk_level = Column(String(20), nullable=True)  # LOW, MEDIUM, HIGH
    recommended_action = Column(String(20), nullable=True)  # APPROVE, REVIEW, DECLINE
    reason_codes = Column(String, nullable=True)  # JSON string
    triggered_rules = Column(String, nullable=True)  # JSON string
    model_version = Column(String(50), nullable=True)
    
    # Tracking
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    organization = relationship("Organization", back_populates="transactions")
    alerts = relationship("Alert", back_populates="transaction", cascade="all, delete-orphan")
    predictions = relationship("FraudPrediction", back_populates="transaction", cascade="all, delete-orphan")
    created_by_user = relationship("User", back_populates="transactions")
    
    def __repr__(self):
        ref = self.transaction_reference or f"id={self.id}"
        return f"<Transaction({ref}, amount={self.amount}, risk={self.risk_score}, org_id={self.organization_id})>"


class FraudPrediction(Base):
    """Immutable audit record of a transaction risk score calculation."""

    __tablename__ = "fraud_predictions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False, index=True)
    transaction_reference = Column(String(64), index=True, nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    recommended_action = Column(String(20), nullable=False)
    reason_codes = Column(String, nullable=False)  # JSON serialized list
    triggered_rules = Column(String, nullable=False)  # JSON serialized list
    model_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    transaction = relationship("Transaction", back_populates="predictions")


class Alert(Base):
    """Fraud alert generated when transaction is flagged as suspicious."""
    
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False, index=True)
    transaction_reference = Column(String(64), index=True, nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    
    # Alert details
    fraud_score = Column(Float, nullable=False)  # Risk score [0, 1]
    threshold = Column(Float, nullable=False)  # Threshold used for alert
    severity = Column(String(20), default="MEDIUM", nullable=False)  # MEDIUM, HIGH
    alert_status = Column(String(50), default="pending", nullable=False)  # pending, reviewed, resolved, false_positive
    
    # Tracking
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    transaction = relationship("Transaction", back_populates="alerts")
    
    def __repr__(self):
        return f"<Alert(id={self.id}, transaction_id={self.transaction_id}, score={self.fraud_score})>"

