"""
Synthetic Transaction Data Generator for CCRD Fraud Intelligence Platform.

Generates realistic legitimate and fraudulent transaction records across multiple
fraud archetypes (velocity attacks, account takeover, high-risk new account, transfer funnel).
All generated records adhere to the B2B interpretable transaction schema.

DISCLAIMER: All generated records are strictly synthetic and for demonstration / testing purposes.
"""

import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


SYNTHETIC_DATA_DISCLAIMER = (
    "NOTICE: This dataset consists entirely of synthetically generated transactions "
    "for machine learning research, demonstration, and benchmarking. "
    "It does not contain real cardholder, payment, or banking information."
)


class SyntheticTransactionGenerator:
    """
    Parametric synthetic transaction generator producing realistic legitimate
    and fraudulent financial transactions.
    """

    CATEGORIES = [
        "retail", "grocery", "travel", "electronics",
        "digital_goods", "gaming", "crypto", "transfer"
    ]
    CATEGORY_PROBS_LEGIT = [0.32, 0.28, 0.10, 0.10, 0.08, 0.04, 0.04, 0.04]
    CATEGORY_PROBS_FRAUD = [0.08, 0.04, 0.12, 0.28, 0.20, 0.12, 0.10, 0.06]

    COUNTRIES = ["US", "GB", "DE", "CA", "FR", "IN", "AU", "BR", "JP", "MX"]
    COUNTRY_PROBS_LEGIT = [0.75, 0.06, 0.04, 0.04, 0.03, 0.03, 0.02, 0.01, 0.01, 0.01]
    COUNTRY_PROBS_FRAUD = [0.40, 0.10, 0.08, 0.08, 0.07, 0.08, 0.05, 0.06, 0.04, 0.04]

    def __init__(self, random_seed: int = 42):
        """Initialize generator with reproducible seed."""
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def generate_legitimate_transaction(self, tx_id: int) -> Dict[str, Any]:
        """Generate a single legitimate transaction adhering to normal consumer patterns."""
        rng = self.rng

        # Circadian distribution for hour: peaks around 12:00-19:00, trough at 01:00-05:00
        hour_weights = [
            1, 0.5, 0.3, 0.2, 0.2, 0.5, 1, 2, 4, 6, 7, 8,
            8.5, 8, 7.5, 8, 8.5, 9, 8, 6, 5, 4, 2.5, 1.5
        ]
        hour_probs = np.array(hour_weights) / sum(hour_weights)
        transaction_hour = int(rng.choice(24, p=hour_probs))

        # Baseline average spend: log-normal with median ~$45
        avg_amount = float(np.round(rng.lognormal(mean=3.7, sigma=0.55), 2))
        avg_amount = max(5.0, min(avg_amount, 2500.0))

        # Current transaction amount varies reasonably around average
        amount_ratio = float(rng.lognormal(mean=0.0, sigma=0.35))
        amount = float(np.round(avg_amount * amount_ratio, 2))
        amount = max(1.50, min(amount, 8000.0))

        # Merchant category & country
        merchant_category = str(rng.choice(self.CATEGORIES, p=self.CATEGORY_PROBS_LEGIT))
        country = str(rng.choice(self.COUNTRIES, p=self.COUNTRY_PROBS_LEGIT))

        # Account age: established account (30 to 1800 days)
        customer_age_days = int(rng.integers(15, 1800))

        # Velocity: mostly 0, occasionally 1 or 2
        transactions_last_hour = int(rng.choice([0, 1, 2, 3], p=[0.82, 0.14, 0.035, 0.005]))

        # Failed transactions last day: mostly 0
        failed_transactions = int(rng.choice([0, 1, 2], p=[0.94, 0.055, 0.005]))

        # Device trust: high (Beta distribution Beta(7, 2), median ~0.82)
        device_trust = float(np.round(rng.beta(7.0, 2.0), 3))
        device_trust = max(0.20, min(device_trust, 0.99))

        return {
            "transaction_reference": f"tx_{uuid.uuid4().hex[:12]}",
            "amount": amount,
            "currency": "USD",
            "merchant_category": merchant_category,
            "transaction_hour": transaction_hour,
            "country": country,
            "customer_age_days": customer_age_days,
            "transactions_last_hour": transactions_last_hour,
            "average_transaction_amount": avg_amount,
            "failed_transactions_last_day": failed_transactions,
            "device_trust_score": device_trust,
            "is_fraud": 0,
        }

    def generate_fraudulent_transaction(self, tx_id: int) -> Dict[str, Any]:
        """
        Generate a single fraudulent transaction based on one of 4 realistic fraud archetypes:
        1. Velocity / Card Testing Attack
        2. Account Takeover (ATO) Bustout
        3. High-Risk Category New Account Fraud
        4. Rapid Transfer / Funnel
        """
        rng = self.rng
        archetype = int(rng.choice([1, 2, 3, 4], p=[0.30, 0.30, 0.25, 0.15]))

        if archetype == 1:
            # --- Archetype 1: Velocity / Card Testing ---
            # Low amount, high rapid attempts, multiple failures, low device trust
            amount = float(np.round(rng.uniform(1.0, 25.0), 2))
            avg_amount = float(np.round(rng.uniform(20.0, 150.0), 2))
            transactions_last_hour = int(rng.integers(4, 15))
            failed_transactions = int(rng.integers(2, 7))
            device_trust = float(np.round(rng.uniform(0.05, 0.35), 3))
            transaction_hour = int(rng.integers(0, 24))
            customer_age_days = int(rng.integers(1, 400))
            category = str(rng.choice(["digital_goods", "retail", "gaming"]))
            country = str(rng.choice(self.COUNTRIES, p=self.COUNTRY_PROBS_FRAUD))

        elif archetype == 2:
            # --- Archetype 2: Account Takeover (ATO) / Bustout ---
            # Normal historical customer baseline, but sudden large transaction at unusual hour
            avg_amount = float(np.round(rng.lognormal(mean=3.5, sigma=0.4), 2))
            # Massive spike ratio: 5x to 25x typical spend
            spike_multiplier = float(rng.uniform(5.0, 20.0))
            amount = float(np.round(avg_amount * spike_multiplier, 2))
            amount = max(350.0, min(amount, 12000.0))
            # Unusual night hours: 01:00 - 05:00
            transaction_hour = int(rng.choice([0, 1, 2, 3, 4, 5, 23]))
            # New/unrecognized device
            device_trust = float(np.round(rng.uniform(0.02, 0.28), 3))
            # Established account being taken over
            customer_age_days = int(rng.integers(90, 1500))
            transactions_last_hour = int(rng.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1]))
            failed_transactions = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
            category = str(rng.choice(["electronics", "travel", "crypto", "digital_goods"]))
            country = str(rng.choice(["MX", "BR", "IN", "DE", "FR", "US"]))

        elif archetype == 3:
            # --- Archetype 3: High-Risk New Account Fraud ---
            # Brand new account purchasing high-liquidity goods
            customer_age_days = int(rng.integers(0, 7))
            amount = float(np.round(rng.uniform(250.0, 2500.0), 2))
            avg_amount = float(np.round(amount * rng.uniform(0.7, 1.1), 2))
            category = str(rng.choice(["crypto", "electronics", "digital_goods", "gaming"]))
            device_trust = float(np.round(rng.uniform(0.10, 0.45), 3))
            transactions_last_hour = int(rng.integers(1, 5))
            failed_transactions = int(rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
            transaction_hour = int(rng.integers(0, 24))
            country = str(rng.choice(self.COUNTRIES, p=self.COUNTRY_PROBS_FRAUD))

        else:
            # --- Archetype 4: Rapid Transfer / Funnel ---
            category = str(rng.choice(["transfer", "crypto"]))
            amount = float(np.round(rng.uniform(150.0, 3000.0), 2))
            avg_amount = float(np.round(rng.uniform(40.0, 200.0), 2))
            transactions_last_hour = int(rng.integers(3, 9))
            failed_transactions = int(rng.integers(0, 3))
            device_trust = float(np.round(rng.uniform(0.08, 0.35), 3))
            customer_age_days = int(rng.integers(5, 300))
            transaction_hour = int(rng.integers(0, 24))
            country = str(rng.choice(self.COUNTRIES, p=self.COUNTRY_PROBS_FRAUD))

        return {
            "transaction_reference": f"tx_{uuid.uuid4().hex[:12]}",
            "amount": amount,
            "currency": "USD",
            "merchant_category": category,
            "transaction_hour": transaction_hour,
            "country": country,
            "customer_age_days": customer_age_days,
            "transactions_last_hour": transactions_last_hour,
            "average_transaction_amount": avg_amount,
            "failed_transactions_last_day": failed_transactions,
            "device_trust_score": device_trust,
            "is_fraud": 1,
        }

    def generate_dataset(
        self,
        total_samples: int = 50000,
        fraud_ratio: float = 0.025
    ) -> pd.DataFrame:
        """
        Generate a full DataFrame of synthetic transactions with a specified fraud ratio.

        Args:
            total_samples: Total number of records to generate.
            fraud_ratio: Proportion of fraudulent records (default: 2.5%).

        Returns:
            pd.DataFrame with generated records and synthetic disclaimer in attrs.
        """
        num_fraud = int(total_samples * fraud_ratio)
        num_legit = total_samples - num_fraud

        records = []
        for i in range(num_legit):
            records.append(self.generate_legitimate_transaction(i))

        for i in range(num_fraud):
            records.append(self.generate_fraudulent_transaction(num_legit + i))

        df = pd.DataFrame(records)
        # Shuffle records to mix legitimate and fraudulent rows
        df = df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
        df.attrs["disclaimer"] = SYNTHETIC_DATA_DISCLAIMER
        df.attrs["synthetic"] = True
        return df

    def generate_train_test_split(
        self,
        total_samples: int = 50000,
        fraud_ratio: float = 0.025,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate stratified train and test synthetic datasets.

        Returns:
            Tuple of (train_df, test_df)
        """
        train_samples = int(total_samples * (1.0 - test_size))
        test_samples = total_samples - train_samples

        train_gen = SyntheticTransactionGenerator(random_seed=self.random_seed)
        test_gen = SyntheticTransactionGenerator(random_seed=self.random_seed + 999)

        train_df = train_gen.generate_dataset(train_samples, fraud_ratio)
        test_df = test_gen.generate_dataset(test_samples, fraud_ratio)

        return train_df, test_df
