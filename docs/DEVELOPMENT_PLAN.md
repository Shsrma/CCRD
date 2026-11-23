# CCRD — Master Development & Transformation Plan
## From ML Prototype to B2B Real-Time Fraud Intelligence Platform

**Date**: November 23, 2026  
**Status**: Proposal awaiting user approval  
**Target Product**: CCRD — Real-Time Fraud Intelligence Platform  

---

## 1. Architectural Vision & Scope

CCRD is evolving from an academic/research project focused on anonymized PCA features (`V1`–`V28`) into an enterprise B2B SaaS platform that enables online merchants, payment gateways, and fintech platforms to assess transaction risk in real time.

```
                      +---------------------------------------+
                      |       B2B Client Applications         |
                      | (E-commerce, Gateways, Mobile Apps)   |
                      +-------------------+-------------------+
                                          |
                              REST API (HTTPS + API Key)
                                          |
                                          v
                      +---------------------------------------+
                      |         FastAPI Gateway / Core         |
                      |  - Multi-Tenant Context Resolver      |
                      |  - API Key & Permission Validator     |
                      |  - Rate Limiting & Audit Logging      |
                      +-------------------+-------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           Fraud Decision Pipeline                                 |
|                                                                                   |
|  1. Input Schema Validation (Pydantic v2)                                         |
|  2. Feature Normalization & Velocity Derivation                                   |
|  3. Deterministic Rules Engine (High amount, low trust, velocity burst)           |
|  4. Calibrated ML Inference (XGBoost / Random Forest Pipeline)                    |
|  5. Hybrid Risk Scoring & Threshold Arbitration (APPROVE / REVIEW / DECLINE)      |
|  6. Explainability & Reason Code Generation                                       |
+-----------------------------------------------------------------------------------+
             |                                                 |
             v                                                 v
+-------------------------+                       +-------------------------+
|   PostgreSQL Database   |                       |    React TS Dashboard   |
| - Tenant Organizations  |                       | - Live Simulator        |
| - Transactions & Scores |                       | - Alerts & Reviews      |
| - Configurable Rules    |                       | - Analytics & Reports   |
| - Audit Logs            |                       | - Org & API Key Mgmt    |
+-------------------------+                       +-------------------------+
```

---

## 2. Four-Phase Roadmap

### Phase 1: Repository Audit and ML Foundation
- Audit existing code and test environment (Completed).
- Preserve and isolate the original research ML pipeline (repairing the feature ordering bug and verifying leakage prevention).
- Design and document the B2B interpretable transaction schema.
- Implement a realistic synthetic transaction generator.
- Build a dedicated, leak-free product ML training pipeline (XGBoost, Random Forest, Logistic Regression baseline) with cross-validation, PR-AUC/ROC-AUC evaluation, and probability calibration.
- Package serialized model artifacts and metadata.

### Phase 2: Real-Time Fraud Detection Engine & API
- Refactor the FastAPI backend into a clean, modern service.
- Integrate the Phase 1 model artifact into a dedicated `FraudDecisionEngine`.
- Implement a modular, testable Rules Engine (velocity thresholds, device trust, abnormal amounts).
- Implement `POST /api/v1/transactions/score` with full idempotency, reason codes, and persistent logging.
- Fix backend authentication bugs (passlib/bcrypt) and Pydantic v2 schemas.
- Provide interactive Swagger documentation.

### Phase 3: B2B Multi-Tenant SaaS Foundation
- PostgreSQL multi-tenant schema with foreign keys, indexes, and tenant scoping (`organizations`, `users`, `api_keys`, `transactions`, `alerts`, `audit_logs`).
- Role-Based Access Control (RBAC): `OWNER`, `ADMIN`, `FRAUD_ANALYST`, `VIEWER`.
- Secure API Key lifecycle (hashing, prefixing `ccrd_live_...` / `ccrd_test_...`, revocation, rotation).
- Tenant isolation enforcement at the dependency layer with automated cross-tenant security test assertions.
- Organization-specific risk thresholds and rule parameters.

### Phase 4: React Dashboard & Public Live Demo
- Modern React + TypeScript + Vite + Tailwind CSS + shadcn/ui frontend.
- Interactive Transaction Simulator directly querying the live scoring API.
- Dashboard with real-time metrics, risk distribution charts (Recharts), and transaction search/filtering.
- Analyst alert management and review workflow.
- Public Demo sandbox mode with synthetic seed transactions and no exposed credentials.
- Containerized deployment configurations (Docker Compose, GitHub Actions CI).

---

## 3. Detailed Phase 1 Implementation Plan

### Part A: Codebase Preservation & Baseline Cleanliness
1. **Preserve Original Research**:
   - Keep `backend/app/ml/` intact as the legacy research code.
   - Fix the critical feature ordering bug in `backend/app/ml/__init__.py:preprocess_input` so the research code behaves correctly with the original Kaggle schema if invoked.
   - Fix passlib/bcrypt incompatibility in `app/core/security.py` so tests and authentication work reliably across modern Python environments.
   - Fix `conftest.py` SQLite engine file-handle disposal to prevent `PermissionError` on Windows during pytest runs.

### Part B: Product-Oriented Transaction Schema
Create `docs/TRANSACTION_SCHEMA.md` and define the interpretable B2B transaction schema:

| Feature Name | Type | Valid Range / Format | Description & Risk Signal |
|:---|:---|:---|:---|
| `transaction_reference` | string | UUID or `tx_[a-zA-Z0-9]+` | Unique merchant transaction identifier |
| `amount` | float | > 0.00 | Monetary value in base currency units |
| `currency` | string | ISO 4217 (e.g. `USD`, `EUR`, `GBP`) | Transaction currency |
| `merchant_category` | string | Enum: `retail`, `travel`, `electronics`, `digital_goods`, `grocery`, `gaming`, `crypto`, `transfer` | Merchant risk sector |
| `transaction_hour` | integer | 0 – 23 | Local hour of transaction (unusual hours flag risk) |
| `country` | string | ISO 3166-1 alpha-2 (e.g. `US`, `GB`, `DE`, `CA`, `IN`) | Transaction origin country |
| `customer_age_days` | integer | >= 0 | Account tenure in days (new accounts carry higher risk) |
| `transactions_last_hour`| integer | >= 0 | Velocity of transactions on card/account |
| `average_transaction_amount` | float | >= 0.00 | Customer baseline historical spend |
| `failed_transactions_last_day` | integer | >= 0 | Card testing and credential stuffing indicator |
| `device_trust_score` | float | 0.0 – 1.0 | Device fingerprint & behavioral biometric trust score |
| `amount_to_avg_ratio` | float (derived) | >= 0.00 | Spike factor: `amount / (average_transaction_amount + 1.0)` |
| `is_international` | integer (derived)| 0 or 1 | Flag indicating if transaction country differs from home country |

### Part C: Realistic Synthetic Data Generator
Create `ml/datasets/synthetic_generator.py`:
- Configurable parameter distribution for normal customer behavior (log-normal amounts, daytime peak hours, high device trust, low hourly velocity).
- Parametric injection of 4 classic fraud patterns:
  1. **Velocity / Card Testing Attack**: High `transactions_last_hour` (>5), multiple `failed_transactions_last_day` (>2), low amounts, low device trust.
  2. **Account Takeover (ATO) / Bust-out**: High `amount_to_avg_ratio` (>5x), strange `transaction_hour` (e.g. 2-5 AM), new device (`device_trust_score` < 0.3), foreign country.
  3. **High-Ticket E-commerce Fraud**: High-risk categories (`crypto`, `electronics`, `digital_goods`), large amounts, newly created customer account (`customer_age_days` < 7).
  4. **Low-Trust Micro-Transfers**: Repeated rapid transfers with device trust < 0.2.
- Generate standard training (e.g. 50,000 samples) and testing partitions with ~2% realistic fraud incidence.
- Include explicit watermark and metadata disclaimers: dataset is strictly synthetic for demonstration purposes.

### Part D: Product ML Training & Evaluation Pipeline
Create `ml/` structured pipeline:
```
ml/
├── datasets/
│   ├── synthetic_generator.py      # Generator engine
│   └── generate_data.py            # CLI script to generate datasets
├── feature_engineering/
│   └── transformer.py              # Scikit-learn compatible ColumnTransformer & derivation
├── training/
│   └── train_product_model.py      # Pipeline trainer (XGBoost, RandomForest, LogisticRegression)
├── evaluation/
│   └── metrics.py                  # ROC-AUC, PR-AUC, calibration curves, confusion matrix
├── artifacts/
│   ├── product_fraud_model.joblib  # Full scikit-learn / imblearn pipeline
│   ├── model_metadata.json         # Version, date, features, metrics, disclaimer
│   └── feature_names.json          # Ordered feature schema
└── tests/
    └── test_product_ml.py          # Pytest validation for generator, transformer, and inference
```
- **Strict Leakage Prevention**:
  - Imbalance correction (SMOTE or class weighting) occurs strictly inside cross-validation splits.
  - Scaling and one-hot encoding fitted only on train partitions.
- **Probability Calibration**:
  - Fit `CalibratedClassifierCV` (isotonic or sigmoid) to ensure predicted probabilities correspond to calibrated risk scores.
- **Model Serialization**:
  - Store entire pipeline (preprocessing + feature derivation + classifier) in a single artifact (`product_fraud_model.joblib`).
  - Generate `model_metadata.json` capturing validation metrics (Precision, Recall, F1, PR-AUC, ROC-AUC, MCC, Brier score), hyperparameters, training date, and synthetic disclaimer.

### Part E: Documentation & Validation
- Create `docs/ML_PIPELINE.md`: Step-by-step documentation of the training process, model selection, calibration, and reproducibility instructions.
- Create `docs/TRANSACTION_SCHEMA.md`: Complete data dictionary for all transaction fields.
- Create `docs/PHASE_1_REPORT.md`: Comprehensive summary of Phase 1 results, metrics table, and readiness checklist for Phase 2.

---

## 4. Reuse, Modification, Creation & Preservation Matrix

| Area | What Can Be Reused | What Needs Modification | What Needs To Be Created | What Must Be Preserved |
|:---|:---|:---|:---|:---|
| **Machine Learning** | Existing `ModelEvaluator` metric math (`roc_auc`, `precision`, `recall`, `f1`, `mcc`) | Bug fix in `preprocess_input` feature order; wrap resamplers in pipelines | Synthetic data generator; B2B feature transformer; product pipeline trainer; calibration step | Original `backend/app/ml/` modules; `backend/ml/train_model.py`; `data/creditcard.csv` |
| **Backend & API** | FastAPI app structure; routing concepts | Migrate schemas to Pydantic v2; fix `passlib/bcrypt` password hashing; fix test database fixtures | `FraudDecisionEngine`; `RulesEngine`; `/api/v1/transactions/score` endpoint; tenant middleware | Existing working endpoints; SQLite engine fallback |
| **Database** | SQLAlchemy setup and `Base` | Update User and Transaction models to support tenant association and new feature schema | Alembic migration scripts; Tenant models (`Organization`, `ApiKey`, `AuditLog`) | Existing schema compatibility; database credentials in `.env` |
| **Frontend** | Color palette and brand identity in `style.css` | Replace static HTML/JS with React + TypeScript + Vite | React SaaS dashboard; transaction simulator; alert case review UI; Recharts visualizers | Original static HTML pages retained in `frontend/legacy/` for reference |
| **Testing** | Working test cases in `test_ml.py` | Fix `conftest.py` SQLite file lock; fix `test_auth.py` and `test_alerts.py` fixtures | New unit tests for synthetic generator, product ML pipeline, rules engine, and multi-tenancy | All passing test cases |

---

## 5. Risk Assessment & Mitigation

| Risk | Impact | Mitigation Strategy |
|:---|:---|:---|
| **Breaking Existing Research Pipeline** | High | Place new product ML code in a dedicated `ml/` package. Do not overwrite or delete `backend/app/ml/`. |
| **Data Leakage in ML Training** | High | Enforce `sklearn.pipeline.Pipeline` / `imblearn.pipeline.Pipeline`. Never scale or resample full datasets prior to splitting. |
| **Inaccurate Probability Scoring** | Medium | Calibrate probabilities using `CalibratedClassifierCV` and clearly label scores as model risk indices, not guaranteed default rates. |
| **bcrypt / passlib Breakage on Modern Python** | High | Migrate password hashing to use native `bcrypt` package directly or configure passlib with explicit bcrypt handler parameters. |
| **SQLite File Lock on Windows during Tests** | Medium | Explicitly call `engine.dispose()` and use in-memory SQLite (`sqlite:///:memory:`) for test runs to avoid disk locking. |
| **Accidental Secret or Large File Commit** | Critical | Verify `.gitignore` covers `.env`, `.db`, `*.joblib`, `*.pkl`, large datasets, and cache directories. |

---

## 6. Rollback & Recovery Strategy

1. **Git Checkpoints**: Every milestone will be committed locally with clear messages following conventional commits (`feat(ml): ...`, `fix(security): ...`).
2. **Non-Destructive Additions**: New functionality will be added alongside existing files before deprecation.
3. **Preservation of Legacy Files**: Legacy static frontend and flat backend scripts (`app.py`, `models.py`, `schemas.py`, `utils.py`) will be archived into `legacy/` rather than deleted, ensuring zero loss of prior work.
4. **Clean Git Branching**: If any breaking change occurs, git allows instantaneous rollback to commit `3135422`.
