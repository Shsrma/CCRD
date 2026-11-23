# CCRD Fraud Decision Engine Specification

### 1. Hybrid Decisioning Philosophy
The CCRD Fraud Decision Engine combines deterministic business rules with calibrated probabilistic machine learning.

```
Incoming Transaction (Pydantic v2 Normalized)
          │
          ├───► [ Rules Engine ] ──────────────┐
          │     (Deterministic checks,         │
          │      6 rules, boosted penalties)   │
          │                                    ▼
          └───► [ Calibrated ML Pipeline ] ──► [ Hybrid Risk Synthesis ]
                (Feature engineering,          │ (Platt calibrated prob +
                 sigmoid calibrated prob)      │  rules boost + floor bounds)
                                               ▼
                                      [ Threshold Arbitration ]
                                      ├── Score < 0.30  ──► LOW risk    ──► APPROVE
                                      ├── 0.30 - 0.70   ──► MEDIUM risk ──► REVIEW
                                      └── Score >= 0.70 ──► HIGH risk   ──► DECLINE
```

### 2. Deterministic Rules Roster
1. **HIGH_AMOUNT_ANOMALY**: Amount > \$5,000 or Amount > \$500 and ratio >= 5.0x customer baseline.
2. **HIGH_VELOCITY_SPIKE**: > 4 transactions in the last hour.
3. **UNUSUAL_NIGHT_TRANSACTION**: Activity between 01:00 and 05:00 for amounts > \$300.
4. **MULTIPLE_FAILED_TRANSACTIONS**: >= 3 failed card authentication attempts in past 24h.
5. **UNTRUSTED_DEVICE**: Device trust score < 0.25.
6. **NEW_ACCOUNT_HIGH_RISK_CATEGORY**: Customer account <= 30 days transacting in high-risk categories (`crypto`, `gaming`, `gift_cards`, `wire_transfer`).

### 3. ML Model & Calibration
- Base Estimator: Logistic Regression with Feature Engineering (`amount_to_avg_ratio`, `is_night`, `is_new_customer`, `high_failed_tx_flag`, `low_trust_device_flag`, log amount transformation, One-Hot Encoding).
- Calibrator: Platt sigmoid calibration (`CalibratedClassifierCV(method='sigmoid')`).
- Metrics: ROC-AUC = 0.9995, PR-AUC = 0.9972, F1 = 0.9634.

### 4. Idempotency & Auditability
Every score request records an immutable `FraudPrediction` entity with reason codes, triggered rule names, and model version to ensure strict compliance and auditability.
