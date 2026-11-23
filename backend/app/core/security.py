"""
Security utilities: JWT token handling, password hashing, etc.
"""

import time
from datetime import datetime, timedelta
from typing import Optional
import bcrypt
import jwt
from fastapi import HTTPException, status


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt directly."""
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        return bcrypt.checkpw(pwd_bytes, hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(
    data: dict, 
    secret_key: str, 
    algorithm: str = "HS256",
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Claims to include in token (e.g., {"sub": "username"})
        secret_key: Secret key for signing
        algorithm: JWT algorithm (default: HS256)
        expires_delta: Token expiration time (default: 30 minutes)
    
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)
    
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
        return encoded_jwt
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create access token"
        ) from e


def decode_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256"
) -> dict:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
        secret_key: Secret key for verification
        algorithm: JWT algorithm (default: HS256)
    
    Returns:
        Decoded token claims
    
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject"
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def hash_api_key(raw_key: str) -> str:
    """Cryptographically hash an API key with SHA-256 for secure database storage."""
    import hashlib
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


def generate_api_key(name: str = "default", environment: str = "live") -> tuple[str, str, str]:
    """
    Generate a secure B2B API Key.
    Returns:
        (raw_api_key, key_prefix, hashed_key)
    Only the raw_api_key is returned to the client once.
    The hashed_key is persisted in the database.
    """
    import secrets
    env = "live" if environment == "live" else "test"
    random_part = secrets.token_hex(20)
    raw_key = f"ccrd_{env}_{random_part}"
    prefix = raw_key[:14]  # e.g. "ccrd_live_3f9a"
    hashed = hash_api_key(raw_key)
    return raw_key, prefix, hashed

