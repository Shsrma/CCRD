# ------------------------------------------------------
# app.py (FIXED & CLEANED VERSION)
# ------------------------------------------------------

from typing import Annotated
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
import os
import time
import random
import jwt
from passlib.context import CryptContext
import uvicorn

# Load environment variables
load_dotenv()

# API Keys
GROQ_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

# SECURITY CONFIG
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# PROJECT IMPORTS
from models import Transaction, Alert, User, Base
from schemas import (
    TransactionInput,
    PredictionResponse,
    UserCreate,
    LoginInput,
    Token,
    User as UserSchema
)
from database import SessionLocal, engine
from utils import load_model, preprocess_input


# ------------------------------------------------------
# 🔧 DATABASE INITIALIZATION FIX
# ------------------------------------------------------
Base.metadata.create_all(bind=engine)


# ------------------------------------------------------
# 🔧 PASSWORD / JWT UTILITIES
# ------------------------------------------------------
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: float):
    to_encode = data.copy()
    expire = time.time() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user_by_username(db, username: str):
    return db.query(User).filter(User.username == username).first()

# CORRECTED DB DEPENDENCY
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------
# 🔐 GET CURRENT USER — FIXED DEPENDENCY
# ------------------------------------------------------
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db=Depends(get_db)
):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")

        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = get_user_by_username(db, username)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")

        return user

    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate token")


# ------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------
app = FastAPI(title="Credit Card Fraud Detection System")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # simplify for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model
model, scaler = load_model()


# ------------------------------------------------------
# GLOBAL SETTINGS
# ------------------------------------------------------
GLOBAL_SETTINGS = {
    "language": "en",
    "timezone": "UTC",
    "fraud_threshold": 0.5
}

@app.get("/global-settings")
def get_global_settings(current_user: Annotated[UserSchema, Depends(get_current_user)]):
    return GLOBAL_SETTINGS

@app.post("/global-settings")
def set_global_settings(payload: dict, current_user: Annotated[UserSchema, Depends(get_current_user)]):

    t = payload.get("type")
    v = payload.get("value")

    if not t or v is None:
        raise HTTPException(status_code=400, detail="type & value required")

    if t == "fraud_threshold":
        try:
            v = float(v)
            if not (0 <= v <= 1):
                raise ValueError
        except:
            raise HTTPException(status_code=400, detail="Threshold must be between 0–1")

    GLOBAL_SETTINGS[t] = v
    return {"status": "updated", "settings": GLOBAL_SETTINGS}


# ------------------------------------------------------
# AUTH ROUTES (NO AUTH REQUIRED)
# ------------------------------------------------------
@app.post("/signup", response_model=UserSchema)
def register_user(user_data: UserCreate, db=Depends(get_db)):

    if get_user_by_username(db, user_data.username):
        raise HTTPException(status_code=400, detail="Username already exists")

    otp_code = "".join([str(random.randint(0, 9)) for _ in range(6)])
    hashed_pw = hash_password(user_data.password)

    new_user = User(
        username=user_data.username,
        hashed_password=hashed_pw,
        role="Fraud Officer",
        otp_secret=otp_code
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    print("\n--- SIMULATED OTP:", otp_code, "---\n")

    return new_user


@app.post("/login", response_model=Token)
def login(form_data: LoginInput, db=Depends(get_db)):

    user = get_user_by_username(db, form_data.username)

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_access_token(
        data={"sub": user.username},
        expires_delta=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

    return {"access_token": token, "token_type": "bearer"}


# ------------------------------------------------------
# 🔐 SECURED ENDPOINTS
# ------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(
    data: TransactionInput,
    current_user: Annotated[UserSchema, Depends(get_current_user)],
    db=Depends(get_db)
):

    tx = Transaction(amount=data.amount, time=data.time, features=str(data.features))
    db.add(tx)
    db.commit()
    db.refresh(tx)

    X = preprocess_input(data, scaler)
    score = model.predict_proba([X])[0][1]

    threshold = GLOBAL_SETTINGS["fraud_threshold"]
    prediction = int(score > threshold)

    if prediction == 1:
        alert = Alert(transaction_id=tx.id, score=score, timestamp=time.time())
        db.add(alert)
        db.commit()

    return PredictionResponse(
        transaction_id=tx.id,
        fraud_prediction=prediction,
        probability=float(score)
    )


@app.get("/alerts")
def get_alerts(current_user: Annotated[UserSchema, Depends(get_current_user)], db=Depends(get_db)):
    return db.query(Alert).all()


@app.get("/profile", response_model=UserSchema)
def get_profile(current_user: Annotated[UserSchema, Depends(get_current_user)]):
    return current_user


@app.post("/translate")
def translate(payload: dict, current_user: Annotated[UserSchema, Depends(get_current_user)]):
    text = payload.get("text", "")
    target = payload.get("target", "en")
    return {"translated": f"(Mock) '{text}' → '{target}'"}


@app.post("/convert-timezone")
def convert_timezone(payload: dict, current_user: Annotated[UserSchema, Depends(get_current_user)]):
    return {"local_time": "2025-11-22 10:00:00", "timezone": "Europe/London"}


# ------------------------------------------------------
# START SERVER
# ------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
