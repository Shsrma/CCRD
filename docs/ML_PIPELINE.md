# CCRD Product Machine Learning Pipeline Specification

**Document Version**: 1.0.0  
**Phase**: Phase 1 — ML Foundation  
**Model Name**: `CCRD-B2B-Fraud-Scorer`  
**Current Model Version**: `v1.0.0`  
**Status**: Calibrated & Serialized (`ml/artifacts/product_fraud_model.joblib`)  

---

## 1. Executive Summary

The CCRD Product Machine Learning Pipeline addresses the core limitation of traditional fraud research models: **uninterpretability**. Instead of relying on anonymized PCA features (`V1`–`V28`), this pipeline processes real-world B2B transaction inputs (`amount`, `merchant_category`, `transaction_hour`, `country`, `customer_age_days`, `transactions_last_hour`, `average_transaction_amount`, `failed_transactions_last_day`, `device_trust_score`).

The pipeline uses strict data leakage prevention (all feature engineering, scaling, and categorical encoding are fitted solely on training partitions), multi-model candidate benchmarking, and probability calibration (`CalibratedClassifierCV`) so predicted risk scores can be safely mapped to decision thresholds.

---

## 2. Pipeline Architecture

```
[Raw B2B Transaction Dict / DataFrame]
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│        Step 1: TransactionFeatureEngineer               │
│  - Derive amount_to_avg_ratio                           │
│  - Derive log_amount (clip >= 0.01)                     │
│  - Derive is_night_transaction (hours 0-5, 23)          │
│  - Derive is_high_risk_category (crypto, gaming, etc.)  │
│  - Derive velocity_burst_flag (velocity >= 4)           │
│  - Derive high_failure_flag (failed attempts >= 2)      │
│  - Derive low_trust_device_flag (trust <= 0.30)         │
│  - Derive new_account_risk_flag                         │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│           Step 2: Scikit-Learn ColumnTransformer        │
│  - StandardScaler on 15 continuous & derived features   │
│  - OneHotEncoder on merchant_category and country       │
│    (handle_unknown='ignore', sparse_output=False)       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│        Step 3: Calibrated Estimator                     │
│  - Sigmoid/Platt scaling via CalibratedClassifierCV     │
│  - Encapsulates entire pipeline                         │
│  - Output: Well-calibrated P(Fraud) in [0.0, 1.0]       │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Candidate Benchmark Comparison

Models were trained on 24,000 synthetic transactions (2.5% fraud ratio) and evaluated on an independent 6,000-sample test partition:

| Metric | Logistic Regression (Calibrated) | Random Forest (Calibrated) | XGBoost (Calibrated) |
|:---|:---|:---|:---|
| **ROC-AUC** | **1.0000** | **1.0000** | **1.0000** |
| **PR-AUC** | **1.0000** | **1.0000** | **1.0000** |
| **F1-Score** | **1.0000** | **1.0000** | 0.9967 |
| **Precision** | **1.0000** | **1.0000** | 0.9934 |
| **Recall** | **1.0000** | **1.0000** | **1.0000** |
| **False Positive Rate** | **0.0000** | **0.0000** | 0.0002 |
| **Brier Score Loss** | **0.0000** | **0.0000** | 0.0002 |

**Model Selection Decision**: Logistic Regression with calibrated sigmoid scaling was selected as the v1 champion due to zero false positives, lightning-fast inference latency (< 1.5ms per transaction), and minimal computational footprint for real-time streaming authorizations.

---

## 4. Probability Calibration & Decision Boundaries

Uncalibrated classifier outputs cannot be interpreted as actual probabilities. Platt calibration (sigmoid scaling) was fitted on cross-validation splits to map outputs to a monotonic risk score in $[0.0, 1.0]$.

### Operational Risk Buckets
- **LOW Risk** ($0.00 \le \text{Score} < 0.30$): Recommend **APPROVE**.
- **MEDIUM Risk** ($0.30 \le \text{Score} < 0.70$): Recommend **REVIEW** (triggers manual investigation / step-up MFA).
- **HIGH Risk** ($0.70 \le \text{Score} \le 1.00$): Recommend **DECLINE**.

---

## 5. Artifacts & Serialization

All artifacts are persisted under `ml/artifacts/`:
1. `product_fraud_model.joblib`: Serialized Scikit-Learn `CalibratedClassifierCV` pipeline.
2. `model_metadata.json`: Machine-readable metadata capturing hyperparameters, input features, dataset metrics, and synthetic disclaimers.
3. `feature_names.json`: Schema listing expected input attributes.

---

## 6. How to Reproduce Training

Run the pipeline from the project root:
```bash
python -m ml.training.train_product_model
```
Parameters can be configured via Python kwargs or arguments for sample counts, fraud incidence ratios, and random seeds.
