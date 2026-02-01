"""
Shared route dependencies (middleware-like functions for routes).
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from datetime import timedelta

from app.core.config import get_settings
from app.core.security import decode_token
from app.database.engine import get_db
from app.models import User
from app.core.logger import logger

settings = get_settings()

# OAuth2 scheme for Swagger UI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Extract and validate current user from JWT token.
    
    Used as a dependency in protected routes:
        @app.get("/protected")
        def protected_route(current_user: User = Depends(get_current_user)):
            return {"user": current_user.username}
    """
    try:
        # Decode token
        payload = decode_token(token, settings.secret_key, settings.algorithm)
        username = payload.get("sub")
        
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing username",
            )
        
        # Get user from database
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            logger.warning(f"Token for non-existent user: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        
        if not user.is_active:
            logger.warning(f"Inactive user attempted login: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled",
            )
        
        return user
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate token",
        ) from e
