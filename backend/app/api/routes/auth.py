"""
Authentication routes: signup, login, logout.
"""

import random
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import hash_password, verify_password, create_access_token
from app.core.logger import logger
from app.database.engine import get_db
from app.models import User
from app.schemas import UserCreate, Token, LoginInput, User as UserSchema

settings = get_settings()
router = APIRouter()


def get_user_by_username(db: Session, username: str) -> User | None:
    """Retrieve user by username."""
    return db.query(User).filter(User.username == username).first()


@router.post("/signup", response_model=UserSchema, status_code=status.HTTP_201_CREATED)
def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user account.
    
    - **username**: Must be 3-100 characters, unique
    - **email**: Optional email address
    - **password**: Minimum 8 characters, must include uppercase and digit
    """
    
    # Check if username already exists
    existing_user = get_user_by_username(db, user_data.username)
    if existing_user:
        logger.warning(f"Signup attempt with existing username: {user_data.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Hash password
    hashed_password = hash_password(user_data.password)
    
    # Generate OTP (for future MFA implementation)
    otp_secret = "".join([str(random.randint(0, 9)) for _ in range(6)])
    
    # Create user
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        role="fraud_officer",
        is_active=True,
        otp_secret=otp_secret,
    )
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        logger.info(f"New user registered: {user_data.username}")
        
        # Log OTP only in development
        if settings.debug:
            logger.debug(f"Generated OTP (dev only): {otp_secret}")
        
        return new_user
    
    except Exception as e:
        db.rollback()
        logger.error(f"User registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        ) from e


@router.post("/login", response_model=Token)
def login(
    form_data: LoginInput,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return JWT access token.
    
    - **username**: Registered username
    - **password**: Account password
    
    Returns JWT token valid for 30 minutes.
    """
    
    # Get user
    user = get_user_by_username(db, form_data.username)
    
    if not user:
        logger.warning(f"Login attempt with non-existent username: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Verify password
    if not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    if not user.is_active:
        logger.warning(f"Login attempt by inactive user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    # Create JWT token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username},
        secret_key=settings.secret_key,
        algorithm=settings.algorithm,
        expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {form_data.username}")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds())
    )


@router.post("/logout")
def logout():
    """
    Logout endpoint (client-side token deletion).
    
    Client should delete the JWT token from localStorage.
    """
    return {"message": "Successfully logged out. Please delete your token."}
