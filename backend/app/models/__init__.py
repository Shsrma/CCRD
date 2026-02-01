"""
SQLAlchemy ORM models for User, Transaction, and Alert entities.
"""

from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.engine import Base


class User(Base):
    """User account model for fraud detection officers."""
    
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True, index=True)
    role = Column(String(50), default="fraud_officer", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # OTP secret for multi-factor authentication
    otp_secret = Column(String(255), nullable=True)
    
    # Relationships
    transactions = relationship("Transaction", back_populates="created_by")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"


class Transaction(Base):
    """Credit card transaction record."""
    
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    
    # Transaction data
    amount = Column(Float, nullable=False)
    timestamp = Column(Float, nullable=False)  # Unix timestamp
    features = Column(String, nullable=False)  # JSON string of feature vector
    
    # ML prediction result
    fraud_prediction = Column(Integer, nullable=True)  # 0=Legitimate, 1=Fraud
    fraud_probability = Column(Float, nullable=True)  # Confidence score [0, 1]
    
    # Tracking
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    alerts = relationship("Alert", back_populates="transaction", cascade="all, delete-orphan")
    created_by_user = relationship("User", back_populates="transactions")
    
    def __repr__(self):
        return f"<Transaction(id={self.id}, amount={self.amount}, fraud={self.fraud_prediction})>"


class Alert(Base):
    """Fraud alert generated when transaction is flagged as suspicious."""
    
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    
    # Reference to transaction
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False, index=True)
    
    # Alert details
    fraud_score = Column(Float, nullable=False)  # Fraud probability [0, 1]
    threshold = Column(Float, nullable=False)  # Threshold used for alert
    alert_status = Column(String(50), default="pending", nullable=False)  # pending, reviewed, resolved, false_positive
    
    # Tracking
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    transaction = relationship("Transaction", back_populates="alerts")
    
    def __repr__(self):
        return f"<Alert(id={self.id}, transaction_id={self.transaction_id}, score={self.fraud_score})>"
