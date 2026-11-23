"""
Unit and integration tests for Phase 3 Multi-Tenant SaaS Architecture.
Tests Organization provisioning, API key authentication, tenant data isolation, and custom threshold override.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi import HTTPException

from app.database.engine import Base
from app.models import User, Organization, TenantConfig, ApiKey, Transaction
from app.api.dependencies import TenantContext, get_tenant_context
from app.api.routes.organizations import (
    create_organization,
    update_tenant_configuration,
    create_tenant_api_key,
    list_tenant_api_keys,
    revoke_tenant_api_key,
)
from app.api.routes.transactions import score_transaction, list_transactions, get_transaction
from app.schemas import (
    OrganizationCreate,
    TenantConfigUpdate,
    ApiKeyCreate,
    B2BTransaction,
    RiskLevel,
    RecommendedAction,
)


@pytest.fixture
def multi_tenant_db():
    """In-memory isolated SQLite database for multi-tenancy tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def org_alpha(multi_tenant_db):
    """Provisions Tenant Organization Alpha (Fintech)."""
    return create_organization(
        org_in=OrganizationCreate(name="Alpha Payments", slug="alpha-pay", plan="enterprise"),
        db=multi_tenant_db,
    )


@pytest.fixture
def org_beta(multi_tenant_db):
    """Provisions Tenant Organization Beta (E-Commerce)."""
    return create_organization(
        org_in=OrganizationCreate(name="Beta Market", slug="beta-market", plan="standard"),
        db=multi_tenant_db,
    )


class TestMultiTenancyAndApiKeys:
    """Test suite verifying tenant isolation and programmatic API key access."""

    def test_organization_provisioning(self, org_alpha, multi_tenant_db):
        assert org_alpha.id is not None
        assert org_alpha.name == "Alpha Payments"
        assert org_alpha.slug == "alpha-pay"

        # Verify automatic default config creation
        config = multi_tenant_db.query(TenantConfig).filter_by(organization_id=org_alpha.id).first()
        assert config is not None
        assert config.low_threshold == 0.30
        assert config.high_threshold == 0.70

    def test_duplicate_slug_raises_400(self, org_alpha, multi_tenant_db):
        with pytest.raises(HTTPException) as exc_info:
            create_organization(
                org_in=OrganizationCreate(name="Alpha Clone", slug="alpha-pay"),
                db=multi_tenant_db,
            )
        assert exc_info.value.status_code == 400

    def test_api_key_lifecycle(self, org_alpha, multi_tenant_db):
        context = TenantContext(organization=org_alpha, role="tenant_admin")

        # 1. Create API key
        key_resp = create_tenant_api_key(
            key_in=ApiKeyCreate(key_name="Ingestion Key 1", environment="live"),
            context=context,
            db=multi_tenant_db,
        )
        assert key_resp.raw_key.startswith("ccrd_live_")
        assert key_resp.key_prefix in key_resp.raw_key

        # 2. List API keys
        keys = list_tenant_api_keys(context=context, db=multi_tenant_db)
        assert len(keys) == 1
        assert keys[0].is_active is True

        # 3. Authenticate with X-API-Key
        resolved_context = get_tenant_context(
            x_api_key=key_resp.raw_key,
            token=None,
            db=multi_tenant_db,
        )
        assert resolved_context.organization.id == org_alpha.id
        assert resolved_context.role == "api_service"

        # 4. Revoke API key
        revoke_tenant_api_key(key_id=key_resp.id, context=context, db=multi_tenant_db)
        with pytest.raises(HTTPException) as exc_info:
            get_tenant_context(x_api_key=key_resp.raw_key, token=None, db=multi_tenant_db)
        assert exc_info.value.status_code == 401

    def test_tenant_data_isolation(self, org_alpha, org_beta, multi_tenant_db):
        """
        Verify that Tenant Beta cannot view, query, or fetch transactions
        belonging to Tenant Alpha.
        """
        context_alpha = TenantContext(organization=org_alpha, role="analyst")
        context_beta = TenantContext(organization=org_beta, role="analyst")

        # Tenant Alpha scores a transaction
        tx = B2BTransaction(
            transaction_reference="tx_alpha_private_001",
            amount=50.00,
            currency="USD",
            merchant_category="grocery",
            transaction_hour=14,
            country="US",
            customer_age_days=100,
            transactions_last_hour=0,
            average_transaction_amount=45.00,
            failed_transactions_last_day=0,
            device_trust_score=0.95,
        )
        score_transaction(tx=tx, context=context_alpha, db=multi_tenant_db)

        # Tenant Alpha lists transactions -> sees 1 transaction
        alpha_list = list_transactions(context=context_alpha, db=multi_tenant_db)
        assert alpha_list.total == 1
        assert alpha_list.transactions[0].transaction_reference == "tx_alpha_private_001"

        # Tenant Beta lists transactions -> sees 0 transactions (data isolation!)
        beta_list = list_transactions(context=context_beta, db=multi_tenant_db)
        assert beta_list.total == 0
        assert len(beta_list.transactions) == 0

        # Tenant Beta attempts to get Tenant Alpha's transaction directly -> 403 Forbidden!
        with pytest.raises(HTTPException) as exc_info:
            get_transaction(reference_or_id="tx_alpha_private_001", context=context_beta, db=multi_tenant_db)
        assert exc_info.value.status_code == 403

    def test_tenant_custom_threshold_override(self, org_alpha, multi_tenant_db):
        """
        Verify that organizations can customize risk boundaries.
        When Alpha sets high_threshold = 0.50, a moderate transaction (score ~0.55)
        is DECLINED instead of REVIEW.
        """
        context_alpha = TenantContext(organization=org_alpha, role="tenant_admin")

        # Update Alpha's thresholds: make DECLINE threshold stricter (0.45 instead of 0.70)
        updated_cfg = update_tenant_configuration(
            cfg_in=TenantConfigUpdate(low_threshold=0.20, high_threshold=0.45),
            context=context_alpha,
            db=multi_tenant_db,
        )
        assert updated_cfg.high_threshold == 0.45

        # Update context with new thresholds
        context_alpha.low_threshold = 0.20
        context_alpha.high_threshold = 0.45

        # Borderline transaction with moderate rule boost
        tx = B2BTransaction(
            transaction_reference="tx_alpha_strict_002",
            amount=850.00,
            currency="USD",
            merchant_category="electronics",
            transaction_hour=15,
            country="US",
            customer_age_days=40,
            transactions_last_hour=2,
            average_transaction_amount=100.00,
            failed_transactions_last_day=1,
            device_trust_score=0.45,
        )
        response = score_transaction(tx=tx, context=context_alpha, db=multi_tenant_db)

        # Under default threshold (0.70), this might be REVIEW; under 0.45 it is DECLINED!
        assert response.risk_score >= 0.45
        assert response.risk_level == RiskLevel.HIGH
        assert response.recommended_action == RecommendedAction.DECLINE
