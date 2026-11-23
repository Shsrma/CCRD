"""
Shared route dependencies (middleware-like functions for routes).
Supports JWT session auth, B2B API Key auth (X-API-Key), and Multi-Tenant Isolation.
"""

from typing import Optional, List
from dataclasses import dataclass
import json
from datetime import datetime
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import decode_token, hash_api_key
from app.database.engine import get_db
from app.models import User, Organization, TenantConfig, ApiKey
from app.core.logger import logger

settings = get_settings()

# OAuth2 scheme for Swagger UI (auto_error=False allows X-API-Key fallback)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


@dataclass
class TenantContext:
    """Holds tenant isolation context, user, active credentials, and tenant config."""
    organization: Organization
    user: Optional[User] = None
    api_key: Optional[ApiKey] = None
    role: str = "analyst"
    low_threshold: float = 0.30
    high_threshold: float = 0.70
    rules_config: dict = None

    @property
    def tenant_id(self) -> int:
        return self.organization.id


def get_or_create_default_org(db: Session) -> Organization:
    """Retrieves or provisions the default primary organization for seamless backward compatibility."""
    org = db.query(Organization).filter(Organization.slug == "default").first()
    if not org:
        org = Organization(
            name="Primary Merchant Default",
            slug="default",
            plan="enterprise",
            is_active=True,
        )
        db.add(org)
        db.flush()

        config = TenantConfig(
            organization_id=org.id,
            low_threshold=0.30,
            high_threshold=0.70,
            rules_config="{}",
        )
        db.add(config)
        db.commit()
        db.refresh(org)
    return org


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Extract and validate current user from JWT token."""
    if not token:
        # Fallback: check if a default user exists for unauthenticated local endpoints
        user = db.query(User).first()
        if user:
            return user
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        payload = decode_token(token, settings.secret_key, settings.algorithm)
        username = payload.get("sub")
        
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing username",
            )
        
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


def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Optional user extractor for endpoints allowing anonymous or API key callers."""
    if not token:
        return None
    try:
        return get_current_user(token=token, db=db)
    except Exception:
        return None



def get_tenant_context(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> TenantContext:
    """
    Resolves the tenant context using either:
    1. Programmatic API Key (X-API-Key: ccrd_live_...)
    2. JWT Session Token (Authorization: Bearer <jwt>)
    3. Default tenant fallback for graceful degradation in local demo mode.
    """
    # 1. API Key Auth
    if x_api_key:
        hashed = hash_api_key(x_api_key)
        api_key = db.query(ApiKey).filter(
            ApiKey.hashed_key == hashed,
            ApiKey.is_active == True,
        ).first()

        if not api_key:
            logger.warning("Invalid or revoked API key attempt")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked API Key",
            )

        api_key.last_used_at = datetime.utcnow()
        db.commit()

        org = api_key.organization
        config = org.config if org else None
        low_th = config.low_threshold if config else 0.30
        high_th = config.high_threshold if config else 0.70
        rules_cfg = json.loads(config.rules_config) if config and config.rules_config else {}

        return TenantContext(
            organization=org,
            user=None,
            api_key=api_key,
            role="api_service",
            low_threshold=low_th,
            high_threshold=high_th,
            rules_config=rules_cfg,
        )

    # 2. JWT Session Auth
    if token:
        user = get_current_user(token=token, db=db)
        org = user.organization
        if not org:
            org = get_or_create_default_org(db)
            if user.organization_id is None:
                user.organization_id = org.id
                db.commit()

        config = org.config
        low_th = config.low_threshold if config else 0.30
        high_th = config.high_threshold if config else 0.70
        rules_cfg = json.loads(config.rules_config) if config and config.rules_config else {}

        return TenantContext(
            organization=org,
            user=user,
            api_key=None,
            role=user.role,
            low_threshold=low_th,
            high_threshold=high_th,
            rules_config=rules_cfg,
        )

    # 3. Default demo fallback
    default_org = get_or_create_default_org(db)
    config = default_org.config
    low_th = config.low_threshold if config else 0.30
    high_th = config.high_threshold if config else 0.70
    rules_cfg = json.loads(config.rules_config) if config and config.rules_config else {}

    return TenantContext(
        organization=default_org,
        user=None,
        api_key=None,
        role="analyst",
        low_threshold=low_th,
        high_threshold=high_th,
        rules_config=rules_cfg,
    )


def require_roles(allowed_roles: List[str]):
    """Enforce Role-Based Access Control (RBAC)."""
    def role_checker(context: TenantContext = Depends(get_tenant_context)):
        if context.role.lower() not in [r.lower() for r in allowed_roles] and context.role.lower() != "superadmin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires one of roles: {allowed_roles}",
            )
        return context
    return role_checker

