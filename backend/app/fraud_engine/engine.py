"""
Fraud Decision Engine for CCRD Real-Time Fraud Intelligence Platform.

Coordinates:
1. Input Normalization
2. Deterministic Rules Engine Execution
3. Calibrated ML Pipeline Inference
4. Hybrid Risk Score Synthesis
5. Threshold Arbitration (APPROVE / REVIEW / DECLINE)
6. Reason Code Generation
"""

import os
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from ml.preprocessing.schema import (
    B2BTransaction,
    RiskLevel,
    RecommendedAction,
    TransactionScoreResponse,
)
from backend.app.rules_engine import RulesEngine, RuleResult
from app.core.logger import logger


class FraudDecisionEngine:
    """
    Central decision engine arbitrating between deterministic business rules
    and machine learning probability estimates.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        rules_engine: Optional[RulesEngine] = None,
        low_threshold: float = 0.30,
        high_threshold: float = 0.70,
    ):
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.model_path = Path(model_path) if model_path else (self.project_root / "ml" / "artifacts" / "product_fraud_model.joblib")
        self.metadata_path = Path(metadata_path) if metadata_path else (self.project_root / "ml" / "artifacts" / "model_metadata.json")

        self.rules_engine = rules_engine or RulesEngine()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self.pipeline = None
        self.metadata = {}
        self.version = "1.0.0"
        self._load_model_artifacts()

    def _load_model_artifacts(self):
        """Loads serialized pipeline and metadata safely."""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
                    self.version = self.metadata.get("version", "1.0.0")
                    thresholds = self.metadata.get("threshold_config", {})
                    self.low_threshold = thresholds.get("low_risk_upper_bound", self.low_threshold)
                    self.high_threshold = thresholds.get("medium_risk_upper_bound", self.high_threshold)

            if self.model_path.exists():
                self.pipeline = joblib.load(self.model_path)
                logger.info(f"✅ Loaded product fraud model from {self.model_path} (version: {self.version})")
            else:
                logger.warning(f"⚠️ Product fraud model artifact not found at {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load fraud model artifact: {e}")
            self.pipeline = None

    def is_operational(self) -> bool:
        """Returns True if the ML model pipeline is loaded and ready for scoring."""
        return self.pipeline is not None

    def evaluate_transaction(
        self,
        tx: B2BTransaction,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
    ) -> TransactionScoreResponse:
        """
        Executes the full 10-step fraud evaluation sequence for an incoming transaction.
        Supports tenant-specific custom risk boundaries (low_threshold & high_threshold).
        """
        effective_low = low_threshold if low_threshold is not None else self.low_threshold
        effective_high = high_threshold if high_threshold is not None else self.high_threshold

        # Step 1 & 2: Input is already validated and normalized via Pydantic v2
        # Step 3: Execute Rules Engine
        rule_results, rule_boost = self.rules_engine.evaluate(tx)
        triggered_rule_names = [r.rule_name for r in rule_results]
        rule_reason_codes = [r.reason_code for r in rule_results]

        # Step 4 & 5: Run ML inference if model is loaded
        if self.pipeline is not None:
            # Construct DataFrame with expected features
            input_df = pd.DataFrame([
                {
                    "amount": tx.amount,
                    "currency": tx.currency,
                    "merchant_category": tx.merchant_category,
                    "transaction_hour": tx.transaction_hour,
                    "country": tx.country,
                    "customer_age_days": tx.customer_age_days,
                    "transactions_last_hour": tx.transactions_last_hour,
                    "average_transaction_amount": tx.average_transaction_amount,
                    "failed_transactions_last_day": tx.failed_transactions_last_day,
                    "device_trust_score": tx.device_trust_score,
                }
            ])
            # Predict calibrated probability
            ml_prob = float(self.pipeline.predict_proba(input_df)[0][1])
        else:
            # Safe fallback if ML pipeline is offline: base solely on rule signals
            logger.warning("ML pipeline offline; using heuristic fallback score.")
            ml_prob = rule_boost

        # Step 6: Hybrid risk score synthesis
        # Combine ML probability and rules
        has_critical_rule = any(r.severity == "HIGH" for r in rule_results)
        if has_critical_rule:
            composite_score = max(0.75, min(1.0, ml_prob + rule_boost))
        else:
            composite_score = min(1.0, max(ml_prob, ml_prob * 0.70 + rule_boost * 0.30))

        composite_score = round(float(composite_score), 4)

        # Step 7: Apply decision thresholds (incorporating tenant customization)
        if composite_score >= effective_high:
            risk_level = RiskLevel.HIGH
            recommended_action = RecommendedAction.DECLINE
        elif composite_score >= effective_low:
            risk_level = RiskLevel.MEDIUM
            recommended_action = RecommendedAction.REVIEW
        else:
            risk_level = RiskLevel.LOW
            recommended_action = RecommendedAction.APPROVE

        # Step 8: Reason codes generation
        reason_codes = list(rule_reason_codes)
        if ml_prob >= 0.70:
            reason_codes.append("ML_HIGH_RISK_PATTERN")
        elif ml_prob >= 0.30:
            reason_codes.append("ML_MODERATE_RISK_ELEVATION")
        elif not reason_codes:
            reason_codes.append("NORMAL_TRANSACTION_PATTERN")

        return TransactionScoreResponse(
            transaction_reference=tx.transaction_reference,
            risk_score=composite_score,
            risk_level=risk_level,
            recommended_action=recommended_action,
            reason_codes=reason_codes,
            triggered_rules=triggered_rule_names,
            model_version=self.version,
            is_synthetic_evaluation=True,
        )
