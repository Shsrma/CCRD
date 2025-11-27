from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="Fraud Officer")
    is_active = Column(Boolean, default=True)
    otp_secret = Column(String, nullable=True)

    # Optional: Add relationship to transactions/alerts if needed
    # transactions = relationship("Transaction", back_populates="user")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float, nullable=False)
    time = Column(Float, nullable=False)
    features = Column(String, nullable=False)

    alerts = relationship("Alert", back_populates="transaction")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    score = Column(Float, nullable=False)
    timestamp = Column(Float, nullable=False)

    transaction = relationship("Transaction", back_populates="alerts")
