"""
Demo data seeder for CCRD Real-Time Fraud Intelligence Platform.
Seeds default merchant organization, users, API keys, diverse transactions, and alerts.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import random

# Ensure backend and project root are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database.engine import SessionLocal, create_all_tables
from app.models import Organization, TenantConfig, ApiKey, User, Transaction, FraudPrediction, Alert
from app.core.security import hash_password, generate_api_key, hash_api_key
from backend.app.fraud_engine import FraudDecisionEngine
from ml.preprocessing.schema import B2BTransaction


def seed_database():
    print("🌱 Initializing and seeding CCRD database...")
    create_all_tables()
    db = SessionLocal()

    try:
        # 1. Organization
        org = db.query(Organization).filter(Organization.slug == "default").first()
        if not org:
            org = Organization(
                name="Global Merchant Corp",
                slug="default",
                plan="enterprise",
                is_active=True,
            )
            db.add(org)
            db.flush()

            cfg = TenantConfig(
                organization_id=org.id,
                low_threshold=0.30,
                high_threshold=0.70,
                rules_config="{}",
            )
            db.add(cfg)
            db.commit()
            db.refresh(org)
            print(f"✅ Created default Organization: {org.name} (id: {org.id})")

        # 2. Demo User
        user = db.query(User).filter(User.username == "fraud_analyst").first()
        if not user:
            user = User(
                username="fraud_analyst",
                hashed_password=hash_password("AnalystPass123!"),
                email="analyst@ccrd-demo.internal",
                role="analyst",
                organization_id=org.id,
                is_active=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"✅ Created demo User: {user.username} / AnalystPass123!")

        # 3. Demo API Key
        raw_demo_key = "ccrd_live_demo_9a8f7e6d5c4b3a2019"
        demo_hash = hash_api_key(raw_demo_key)
        existing_key = db.query(ApiKey).filter(ApiKey.hashed_key == demo_hash).first()
        if not existing_key:
            api_key = ApiKey(
                organization_id=org.id,
                key_name="Stripe Ingestion Production Key",
                key_prefix=raw_demo_key[:14],
                hashed_key=demo_hash,
                is_active=True,
            )
            db.add(api_key)
            db.commit()
            print(f"✅ Created demo API Key: {raw_demo_key}")

        # 4. Initialize Fraud Engine for realistic scoring
        fraud_engine = FraudDecisionEngine()

        # Check existing transactions
        tx_count = db.query(Transaction).count()
        if tx_count < 10:
            print("🚀 Generating initial realistic transaction stream...")
            seed_samples = [
                # Normal Legitimate Transactions
                {"amount": 34.50, "merchant_category": "grocery", "hour": 11, "age": 240, "vel": 0, "avg": 35.0, "failed": 0, "trust": 0.96},
                {"amount": 12.80, "merchant_category": "retail", "hour": 13, "age": 120, "vel": 1, "avg": 15.0, "failed": 0, "trust": 0.92},
                {"amount": 89.90, "merchant_category": "retail", "hour": 16, "age": 90, "vel": 0, "avg": 80.0, "failed": 0, "trust": 0.88},
                {"amount": 4.50, "merchant_category": "retail", "hour": 8, "age": 400, "vel": 0, "avg": 5.0, "failed": 0, "trust": 0.99},
                {"amount": 62.00, "merchant_category": "travel", "hour": 17, "age": 310, "vel": 1, "avg": 60.0, "failed": 0, "trust": 0.94},
                {"amount": 115.00, "merchant_category": "grocery", "hour": 19, "age": 180, "vel": 1, "avg": 90.0, "failed": 0, "trust": 0.91},
                {"amount": 28.00, "merchant_category": "digital_goods", "hour": 21, "age": 75, "vel": 0, "avg": 30.0, "failed": 0, "trust": 0.85},
                {"amount": 49.99, "merchant_category": "digital_goods", "hour": 1, "age": 520, "vel": 0, "avg": 50.0, "failed": 0, "trust": 0.98},

                # Borderline / Elevated Review Transactions
                {"amount": 480.00, "merchant_category": "travel", "hour": 15, "age": 45, "vel": 2, "avg": 90.0, "failed": 1, "trust": 0.52},
                {"amount": 650.00, "merchant_category": "electronics", "hour": 14, "age": 35, "vel": 1, "avg": 120.0, "failed": 1, "trust": 0.48},
                {"amount": 320.00, "merchant_category": "retail", "hour": 18, "age": 20, "vel": 2, "avg": 80.0, "failed": 0, "trust": 0.45},
                {"amount": 550.00, "merchant_category": "gaming", "hour": 23, "age": 15, "vel": 2, "avg": 70.0, "failed": 1, "trust": 0.42},

                # Fraud / Declined Attacks
                {"amount": 4200.00, "merchant_category": "electronics", "hour": 2, "age": 300, "vel": 2, "avg": 50.0, "failed": 2, "trust": 0.08},
                {"amount": 15.00, "merchant_category": "digital_goods", "hour": 3, "age": 8, "vel": 9, "avg": 20.0, "failed": 5, "trust": 0.11},
                {"amount": 890.00, "merchant_category": "crypto", "hour": 22, "age": 3, "vel": 1, "avg": 100.0, "failed": 1, "trust": 0.22},
                {"amount": 2900.00, "merchant_category": "transfer", "hour": 4, "age": 6, "vel": 4, "avg": 40.0, "failed": 3, "trust": 0.14},
                {"amount": 3500.00, "merchant_category": "transfer", "hour": 1, "age": 14, "vel": 3, "avg": 65.0, "failed": 2, "trust": 0.09},
            ]

            now = datetime.utcnow()
            for idx, sample in enumerate(seed_samples):
                ref = f"tx_live_seed_{idx+1001}"
                tx_in = B2BTransaction(
                    transaction_reference=ref,
                    amount=sample["amount"],
                    currency="USD",
                    merchant_category=sample["merchant_category"],
                    transaction_hour=sample["hour"],
                    country="US" if sample["trust"] > 0.3 else "MX",
                    customer_age_days=sample["age"],
                    transactions_last_hour=sample["vel"],
                    average_transaction_amount=sample["avg"],
                    failed_transactions_last_day=sample["failed"],
                    device_trust_score=sample["trust"],
                )
                decision = fraud_engine.evaluate_transaction(tx_in)
                risk_level_str = decision.risk_level.value if hasattr(decision.risk_level, "value") else str(decision.risk_level)
                action_str = decision.recommended_action.value if hasattr(decision.recommended_action, "value") else str(decision.recommended_action)

                tx_time = now - timedelta(minutes=(len(seed_samples) - idx) * 12)
                tx_record = Transaction(
                    transaction_reference=ref,
                    organization_id=org.id,
                    amount=sample["amount"],
                    currency="USD",
                    merchant_category=sample["merchant_category"],
                    transaction_hour=sample["hour"],
                    country="US" if sample["trust"] > 0.3 else "MX",
                    customer_age_days=sample["age"],
                    transactions_last_hour=sample["vel"],
                    average_transaction_amount=sample["avg"],
                    failed_transactions_last_day=sample["failed"],
                    device_trust_score=sample["trust"],
                    risk_score=decision.risk_score,
                    risk_level=risk_level_str,
                    recommended_action=action_str,
                    reason_codes=json.dumps(decision.reason_codes),
                    triggered_rules=json.dumps(decision.triggered_rules),
                    model_version=decision.model_version,
                    created_at=tx_time,
                )
                db.add(tx_record)
                db.flush()

                pred = FraudPrediction(
                    transaction_id=tx_record.id,
                    transaction_reference=ref,
                    organization_id=org.id,
                    risk_score=decision.risk_score,
                    risk_level=risk_level_str,
                    recommended_action=action_str,
                    reason_codes=json.dumps(decision.reason_codes),
                    triggered_rules=json.dumps(decision.triggered_rules),
                    model_version=decision.model_version,
                    created_at=tx_time,
                )
                db.add(pred)

                if action_str in ["REVIEW", "DECLINE"]:
                    alert = Alert(
                        transaction_id=tx_record.id,
                        transaction_reference=ref,
                        organization_id=org.id,
                        fraud_score=decision.risk_score,
                        threshold=0.30,
                        severity="HIGH" if action_str == "DECLINE" else "MEDIUM",
                        alert_status="pending",
                        created_at=tx_time,
                    )
                    db.add(alert)

            db.commit()
            print(f"✅ Successfully seeded {len(seed_samples)} realistic transactions and alerts.")

    except Exception as e:
        db.rollback()
        print(f"❌ Error during database seeding: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
