"""
Organization and API Key management routes for B2B Multi-Tenant SaaS.
Supports tenant provisioning, threshold customization, and programmatic API key generation.
"""

from typing import List, Optional
import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.engine import get_db
from app.models import Organization, TenantConfig, ApiKey, User
from app.api.dependencies import get_tenant_context, TenantContext, require_roles
from app.core.security import generate_api_key
from app.schemas import (
    OrganizationCreate,
    OrganizationResponse,
    TenantConfigUpdate,
    TenantConfigResponse,
    ApiKeyCreate,
    ApiKeyResponse,
    ApiKeyCreatedResponse,
)
from app.core.logger import logger

router = APIRouter()


@router.post(
    "",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new B2B Merchant/Organization",
)
def create_organization(
    org_in: OrganizationCreate,
    db: Session = Depends(get_db),
):
    """Provisions a new tenant organization with isolated schema and default config."""
    existing = db.query(Organization).filter(Organization.slug == org_in.slug).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Organization slug '{org_in.slug}' already exists",
        )

    org = Organization(
        name=org_in.name,
        slug=org_in.slug,
        plan=org_in.plan or "standard",
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

    logger.info(f"Provisioned new organization: {org.name} ({org.slug})")
    return org


@router.get(
    "/me",
    response_model=OrganizationResponse,
    summary="Get current organization details",
)
def get_current_organization(
    context: TenantContext = Depends(get_tenant_context),
):
    """Returns the caller's active tenant organization profile."""
    return context.organization


@router.get(
    "/me/config",
    response_model=TenantConfigResponse,
    summary="Get tenant fraud risk thresholds and rules config",
)
def get_tenant_configuration(
    context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Returns active thresholds (low/high) and rule settings for the caller's organization."""
    config = db.query(TenantConfig).filter(TenantConfig.organization_id == context.organization.id).first()
    if not config:
        config = TenantConfig(
            organization_id=context.organization.id,
            low_threshold=0.30,
            high_threshold=0.70,
            rules_config="{}",
        )
        db.add(config)
        db.commit()
        db.refresh(config)
    return config


@router.put(
    "/me/config",
    response_model=TenantConfigResponse,
    summary="Update tenant fraud risk thresholds and rules config",
)
def update_tenant_configuration(
    cfg_in: TenantConfigUpdate,
    context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Allows tenant administrators to customize fraud detection thresholds and rules."""
    config = db.query(TenantConfig).filter(TenantConfig.organization_id == context.organization.id).first()
    if not config:
        config = TenantConfig(organization_id=context.organization.id)
        db.add(config)

    if cfg_in.low_threshold is not None:
        config.low_threshold = cfg_in.low_threshold
    if cfg_in.high_threshold is not None:
        config.high_threshold = cfg_in.high_threshold
    if cfg_in.rules_config is not None:
        config.rules_config = cfg_in.rules_config
    if cfg_in.webhook_url is not None:
        config.webhook_url = cfg_in.webhook_url

    db.commit()
    db.refresh(config)
    logger.info(f"Updated fraud config for organization {context.organization.slug}")
    return config


@router.post(
    "/me/api-keys",
    response_model=ApiKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a new B2B API Key for programmatic scoring",
)
def create_tenant_api_key(
    key_in: ApiKeyCreate,
    context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """
    Generates a secure API Key for B2B backend integrations.
    Returns the secret key once.
    """
    raw_key, prefix, hashed = generate_api_key(
        name=key_in.key_name,
        environment=key_in.environment or "live"
    )

    api_key_record = ApiKey(
        organization_id=context.organization.id,
        key_name=key_in.key_name,
        key_prefix=prefix,
        hashed_key=hashed,
        is_active=True,
    )
    db.add(api_key_record)
    db.commit()
    db.refresh(api_key_record)

    logger.info(f"Created API Key '{key_in.key_name}' for org {context.organization.slug}")

    return ApiKeyCreatedResponse(
        id=api_key_record.id,
        key_name=api_key_record.key_name,
        key_prefix=prefix,
        raw_key=raw_key,
        created_at=api_key_record.created_at,
    )


@router.get(
    "/me/api-keys",
    response_model=List[ApiKeyResponse],
    summary="List active API Keys for this organization",
)
def list_tenant_api_keys(
    context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Lists masked API key identifiers and usage metadata for dashboard management."""
    keys = db.query(ApiKey).filter(
        ApiKey.organization_id == context.organization.id,
        ApiKey.is_active == True,
    ).order_by(ApiKey.created_at.desc()).all()
    return keys


@router.delete(
    "/me/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke an API Key",
)
def revoke_tenant_api_key(
    key_id: int,
    context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db),
):
    """Immediately invalidates an API Key, rejecting future requests using it."""
    key = db.query(ApiKey).filter(
        ApiKey.id == key_id,
        ApiKey.organization_id == context.organization.id,
    ).first()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API Key not found",
        )

    key.is_active = False
    db.commit()
    logger.info(f"Revoked API Key {key_id} for org {context.organization.slug}")
    return None
