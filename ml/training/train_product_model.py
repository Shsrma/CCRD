"""
Product ML Pipeline Training Script for CCRD Real-Time Fraud Intelligence Platform.

Trains, calibrates, compares, and serializes fraud detection models
(Logistic Regression baseline, Random Forest, XGBoost) using the interpretable B2B schema.

Encapsulates feature engineering, scaling, one-hot encoding, and probability calibration
in a single self-contained Scikit-Learn Pipeline artifact.
"""

import os
import sys
import json
import joblib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.synthetic_generator import (
    SyntheticTransactionGenerator,
    SYNTHETIC_DATA_DISCLAIMER,
)
from ml.feature_engineering.transformer import (
    TransactionFeatureEngineer,
    build_preprocessor,
)
from ml.evaluation.metrics import evaluate_fraud_model


ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_evaluate_models(
    total_samples: int = 30000,
    fraud_ratio: float = 0.025,
    random_seed: int = 42,
):
    """
    Executes end-to-end model training, cross-model comparison, calibration,
    and serialization.
    """
    print(f"[*] Generating {total_samples} synthetic transactions (fraud ratio: {fraud_ratio:.1%})...")
    generator = SyntheticTransactionGenerator(random_seed=random_seed)
    train_df, test_df = generator.generate_train_test_split(
        total_samples=total_samples,
        fraud_ratio=fraud_ratio,
        test_size=0.2,
    )

    feature_cols = [
        "amount",
        "currency",
        "merchant_category",
        "transaction_hour",
        "country",
        "customer_age_days",
        "transactions_last_hour",
        "average_transaction_amount",
        "failed_transactions_last_day",
        "device_trust_score",
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["is_fraud"].values

    X_test = test_df[feature_cols]
    y_test = test_df["is_fraud"].values

    print(f"[*] Train set shape: {X_train.shape}, positive fraud labels: {np.sum(y_train)}")
    print(f"[*] Test set shape: {X_test.shape}, positive fraud labels: {np.sum(y_test)}")

    # Candidate models
    candidates = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=6,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=15,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_seed,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, base_clf in candidates.items():
        print(f"\n[*] Training and calibrating: {name}...")

        # Build pipeline: Feature Engineering -> ColumnTransformer -> Classifier
        pipe = Pipeline(
            steps=[
                ("engineer", TransactionFeatureEngineer()),
                ("preprocessor", build_preprocessor()),
                ("classifier", base_clf),
            ]
        )

        pipe.fit(X_train, y_train)

        # Calibrate probabilities using CalibratedClassifierCV
        print(f"[*] Calibrating probabilities for {name}...")
        calibrated_pipe = CalibratedClassifierCV(
            estimator=pipe,
            method="sigmoid",
            cv=3,
        )
        calibrated_pipe.fit(X_train, y_train)

        # Inference on test partition
        y_prob = calibrated_pipe.predict_proba(X_test)[:, 1]
        # Using default operational decision threshold 0.50
        y_pred = (y_prob >= 0.50).astype(int)

        metrics = evaluate_fraud_model(y_test, y_pred, y_prob)
        results[name] = metrics
        fitted_pipelines[name] = calibrated_pipe

        print(
            f"    -> ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"PR-AUC: {metrics['pr_auc']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"FPR: {metrics['false_positive_rate']:.4f} | "
            f"Brier: {metrics['brier_score']:.4f}"
        )

    # Champion selection based on PR-AUC (key metric for severe fraud imbalance)
    champion_name = max(results, key=lambda k: results[k]["pr_auc"])
    champion_pipeline = fitted_pipelines[champion_name]
    champion_metrics = results[champion_name]

    print(f"\n==================================================")
    print(f"🏆 CHAMPION MODEL SELECTED: {champion_name.upper()}")
    print(f"    PR-AUC:   {champion_metrics['pr_auc']:.4f}")
    print(f"    ROC-AUC:  {champion_metrics['roc_auc']:.4f}")
    print(f"    F1-Score: {champion_metrics['f1_score']:.4f}")
    print(f"    Recall:   {champion_metrics['recall']:.4f}")
    print(f"==================================================")

    # Save champion model artifact
    model_artifact_path = ARTIFACTS_DIR / "product_fraud_model.joblib"
    joblib.dump(champion_pipeline, model_artifact_path)
    print(f"[+] Saved serialized pipeline to: {model_artifact_path}")

    # Save model metadata
    metadata = {
        "model_name": champion_name,
        "model_type": str(type(champion_pipeline).__name__),
        "version": "1.0.0",
        "training_timestamp": datetime.utcnow().isoformat() + "Z",
        "input_features": feature_cols,
        "dataset_metadata": {
            "total_samples": total_samples,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "fraud_ratio": fraud_ratio,
            "random_seed": random_seed,
            "synthetic_disclaimer": SYNTHETIC_DATA_DISCLAIMER,
        },
        "threshold_config": {
            "low_risk_upper_bound": 0.30,
            "medium_risk_upper_bound": 0.70,
            "default_action_threshold": 0.50,
        },
        "evaluation_metrics": champion_metrics,
        "candidate_comparison": results,
    }

    metadata_path = ARTIFACTS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved model metadata to: {metadata_path}")

    feature_names_path = ARTIFACTS_DIR / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"[+] Saved feature names to: {feature_names_path}")

    return champion_pipeline, metadata


if __name__ == "__main__":
    train_and_evaluate_models()
