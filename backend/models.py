from sqlalchemy import Column, Integer, Float, String, Boolean
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="Fraud Officer")
    is_active = Column(Boolean, default=True)
    otp_secret = Column(String, nullable=True) # Placeholder for 2FA/OTP

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float)
    time = Column(Float)
    features = Column(String)

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer)
    score = Column(Float)
    timestamp = Column(Float) # ADDED: Timestamp for display purposes
    timestamp = Column(Float) # ADDED: Timestamp for display