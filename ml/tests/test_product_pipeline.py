"""
Unit tests for CCRD Product ML Pipeline and Synthetic Generator.
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from ml.datasets.synthetic_generator import SyntheticTransactionGenerator
from ml.feature_engineering.transformer import TransactionFeatureEngineer, build_preprocessor


@pytest.fixture
def generator():
    return SyntheticTransactionGenerator(random_seed=42)


class TestSyntheticGenerator:
    """Tests for synthetic data generation."""

    def test_generate_single_legit_transaction(self, generator):
        tx = generator.generate_legitimate_transaction(1)
        assert tx["is_fraud"] == 0
        assert tx["amount"] > 0
        assert 0 <= tx["transaction_hour"] <= 23
        assert 0.0 <= tx["device_trust_score"] <= 1.0

    def test_generate_single_fraud_transaction(self, generator):
        tx = generator.generate_fraudulent_transaction(1)
        assert tx["is_fraud"] == 1
        assert tx["amount"] > 0
        assert 0.0 <= tx["device_trust_score"] <= 1.0

    def test_generate_dataset_proportions(self, generator):
        df = generator.generate_dataset(total_samples=1000, fraud_ratio=0.03)
        assert len(df) == 1000
        fraud_count = df["is_fraud"].sum()
        assert 25 <= fraud_count <= 35  # ~30 fraud cases


class TestFeatureEngineer:
    """Tests for feature engineering transformations."""

    def test_derived_features(self, generator):
        df = generator.generate_dataset(total_samples=20)
        engineer = TransactionFeatureEngineer()
        transformed = engineer.transform(df)

        assert "amount_to_avg_ratio" in transformed.columns
        assert "log_amount" in transformed.columns
        assert "is_night_transaction" in transformed.columns
        assert "velocity_burst_flag" in transformed.columns
        assert "low_trust_device_flag" in transformed.columns

        # Verify logic
        for _, row in transformed.iterrows():
            expected_ratio = row["amount"] / (row["average_transaction_amount"] + 1.0)
            assert np.isclose(row["amount_to_avg_ratio"], expected_ratio, atol=1e-3)


class TestModelArtifactInference:
    """Tests loading the serialized product ML artifact."""

    def test_artifact_loads_and_predicts(self, generator):
        artifact_path = Path(__file__).resolve().parent.parent / "artifacts" / "product_fraud_model.joblib"
        assert artifact_path.exists(), "Product fraud model artifact does not exist"

        pipeline = joblib.load(artifact_path)
        assert pipeline is not None

        # Sample transactions
        df = generator.generate_dataset(total_samples=10)
        feature_cols = [
            "amount", "currency", "merchant_category", "transaction_hour", "country",
            "customer_age_days", "transactions_last_hour", "average_transaction_amount",
            "failed_transactions_last_day", "device_trust_score"
        ]
        X = df[feature_cols]

        probs = pipeline.predict_proba(X)
        assert probs.shape == (10, 2)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert np.allclose(probs[:, 0] + probs[:, 1], 1.0)
