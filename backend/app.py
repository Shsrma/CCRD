# ------------------------------------------------------
# app.py (FINAL MERGED VERSION with Authentication and Features)
# ------------------------------------------------------

# NEW SECURITY IMPORTS
from typing import Annotated
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
import os
import requests
import time
import random
import jwt
from passlib.context import CryptContext

# Load .env variables
load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

# SECURITY CONFIG
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key") # IMPORTANT: Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# EXISTING IMPORTS
from models import Transaction, Alert, User
from schemas import TransactionInput, PredictionResponse, UserCreate, LoginInput, Token, User as UserSchema
from database import SessionLocal, engine
from utils import load_model, preprocess_input
import uvicorn

# --- UTILITY FUNCTIONS FOR SECURITY ---
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: float):
    to_encode = data.copy()
    expire = time.time() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_by_username(db, username: str):
    return db.query(User).filter(User.username == username).first()

def get_current_user(db: SessionLocal = Depends(SessionLocal), token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = get_user_by_username(db, username=username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
# ---------------------------------------------------------------------------------

# Create database tables (now includes User)
User.metadata.create_all(bind=engine)
Transaction.metadata.create_all(bind=engine)
Alert.metadata.create_all(bind=engine)

# FastAPI App
app = FastAPI(title="Credit Card Fraud Detection System")

# CORS (IMPORTANT for frontend)
origins = [
    "http://localhost",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8000",
    "*" # Using * for deployment simplicity, but restrict in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model
model, scaler = load_model()


# ------------------------------------------------------
#  GLOBAL SETTINGS STORAGE (language + timezone + THRESHOLD)
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

    if t not in ("language", "timezone", "fraud_threshold"):
        raise HTTPException(status_code=400, detail="Invalid setting type")

    if t == "fraud_threshold":
        try:
            v = float(v)
            if not (0.0 <= v <= 1.0):
                raise ValueError
        except ValueError:
            raise HTTPException(status_code=400, detail="Threshold must be a float between 0.0 and 1.0")
        
    GLOBAL_SETTINGS[t] = v
    return {"status": "updated", "settings": GLOBAL_SETTINGS}

# ------------------------------------------------------
#  AUTHENTICATION ROUTES (UNSECURED)
# ------------------------------------------------------

@app.post("/signup", response_model=UserSchema)
def register_user(user_data: UserCreate, db: SessionLocal = Depends(SessionLocal)):
    db_user = get_user_by_username(db, username=user_data.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = hash_password(user_data.password)
    
    # Simulate OTP generation/setup during signup
    otp_code = "".join([str(random.randint(0, 9)) for _ in range(6)])
    
    new_user = User(
        username=user_data.username,
        hashed_password=hashed_password,
        role="Fraud Officer",
        otp_secret=otp_code
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    print(f"\n--- SIMULATED OTP for {new_user.username}: {otp_code} (Use this for 2FA) ---\n")
    
    return new_user


@app.post("/login", response_model=Token)
def login_for_access_token(form_data: LoginInput, db: SessionLocal = Depends(SessionLocal)):
    user = get_user_by_username(db, username=form_data.username)
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


# ------------------------------------------------------
#  SECURED ROUTE TEMPLATE (Requires JWT Token)
# ------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(data: TransactionInput, current_user: Annotated[UserSchema, Depends(get_current_user)]):
    db = SessionLocal()

    db_tx = Transaction(amount=data.amount, time=data.time, features=str(data.features))
    db.add(db_tx)
    db.commit()
    db.refresh(db_tx)

    X = preprocess_input(data, scaler)
    score = model.predict_proba([X])[0][1]
    
    current_threshold = GLOBAL_SETTINGS["fraud_threshold"]
    prediction = int(score > current_threshold)

    if prediction == 1:
        alert = Alert(transaction_id=db_tx.id, score=score, timestamp=time.time())
        db.add(alert)
        db.commit()

    return PredictionResponse(transaction_id=db_tx.id, fraud_prediction=prediction, probability=float(score))

@app.get("/alerts")
def get_alerts(current_user: Annotated[UserSchema, Depends(get_current_user)]):
    db = SessionLocal()
    return db.query(Alert).all()

@app.get("/profile", response_model=UserSchema)
def get_user_profile(current_user: Annotated[UserSchema, Depends(get_current_user)]):
    """Returns the details of the currently logged-in user."""
    return current_user

# Translation and Timezone conversion endpoints also now require authentication
@app.post("/translate")
def translate_text(payload: dict, current_user: Annotated[UserSchema, Depends(get_current_user)]):
    # This is a placeholder for the full Groq implementation
    text = payload.get("text", "")
    target = payload.get("target", "en")
    return {"translated": f"Translated '{text}' to '{target}' (API logic omitted for brevity)."}

@app.post("/convert-timezone")
def convert_timezone(payload: dict, current_user: Annotated[UserSchema, Depends(get_current_user)]):
    # This is a placeholder for the full Google Timezone implementation
    return {"local_time": "2025-11-22 10:00:00", "timezone": "Europe/London (API logic omitted for brevity)."}


# ------------------------------------------------------
#  START SERVER
# ------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)