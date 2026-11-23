"""
Feature engineering module for CCRD product ML pipeline.
Transforms raw B2B transaction attributes into numerical and one-hot encoded feature representations.
Computes derived risk features (velocity ratios, night flags, risk category flags).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


HIGH_RISK_CATEGORIES = {"crypto", "gaming", "digital_goods", "transfer"}
NIGHT_HOURS = {0, 1, 2, 3, 4, 5, 23}


class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that derives interpretable financial risk features
    from raw B2B transaction attributes.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Spend acceleration: ratio of current amount to 30-day average
        avg_amt = df["average_transaction_amount"].replace(0, 1.0)
        df["amount_to_avg_ratio"] = df["amount"] / (avg_amt + 1.0)

        # 2. Log amount (tames heavy-tailed distributions)
        df["log_amount"] = np.log1p(df["amount"].clip(lower=0.01))

        # 3. Off-peak night transaction flag
        df["is_night_transaction"] = df["transaction_hour"].isin(NIGHT_HOURS).astype(int)

        # 4. High-risk merchant sector flag
        df["is_high_risk_category"] = df["merchant_category"].isin(HIGH_RISK_CATEGORIES).astype(int)

        # 5. Velocity burst flag (>= 4 attempts in trailing hour)
        df["velocity_burst_flag"] = (df["transactions_last_hour"] >= 4).astype(int)

        # 6. High recent failure count flag (>= 2 failed attempts in 24h)
        df["high_failure_flag"] = (df["failed_transactions_last_day"] >= 2).astype(int)

        # 7. Low device trust flag (trust <= 0.30)
        df["low_trust_device_flag"] = (df["device_trust_score"] <= 0.30).astype(int)

        # 8. High-risk new account flag (age < 14 days and high-risk category)
        df["new_account_risk_flag"] = (
            (df["customer_age_days"] < 14) & (df["is_high_risk_category"] == 1)
        ).astype(int)

        return df


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that scales continuous features
    and one-hot encodes categorical merchant categories and countries.
    """
    numeric_features = [
        "amount",
        "log_amount",
        "amount_to_avg_ratio",
        "transaction_hour",
        "customer_age_days",
        "transactions_last_hour",
        "average_transaction_amount",
        "failed_transactions_last_day",
        "device_trust_score",
        "is_night_transaction",
        "is_high_risk_category",
        "velocity_burst_flag",
        "high_failure_flag",
        "low_trust_device_flag",
        "new_account_risk_flag",
    ]

    categorical_features = [
        "merchant_category",
        "country",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    return preprocessor
