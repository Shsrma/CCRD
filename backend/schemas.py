from pydantic import BaseModel
from typing import List


# ---------------------------
# Transaction & Prediction
# ---------------------------
class TransactionInput(BaseModel):
    time: float
    amount: float
    features: List[float]


class PredictionResponse(BaseModel):
    transaction_id: int
    fraud_prediction: int
    probability: float


# ---------------------------
# User Schemas
# ---------------------------
class UserBase(BaseModel):
    username: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    role: str
    is_active: bool

    class Config:
        orm_mode = True


# ---------------------------
# Authentication Schemas
# ---------------------------
class Token(BaseModel):
    access_token: str
    token_type: str


class LoginInput(BaseModel):
    username: str
    password: str
