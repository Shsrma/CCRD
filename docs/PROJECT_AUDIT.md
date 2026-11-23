# CCRD Project Audit & Codebase Assessment

**Date**: November 23, 2026  
**Auditor**: Senior Software Architect & AI/ML Engineer (Antigravity)  
**Project**: CCRD (Credit Card Fraud Detection System)  
**Proposed SaaS Product**: CCRD — Real-Time Fraud Intelligence Platform  
**Git Branch**: `main` (commit `3135422`, working tree clean)  

---

## 1. Executive Summary

A comprehensive, ground-truth audit of the entire CCRD repository was conducted. The repository originated as an ML research/demonstration project predicting fraud using the Kaggle European Credit Card dataset (features `Time`, `V1`–`V28`, `Amount`). A subsequent modernization attempt added a modular FastAPI backend, unit test suite, and Docker configurations. 

However, rigorous inspection of the code, dependencies, and test executions reveals a significant gap between the documentation claims (e.g. in `IMPLEMENTATION_SUMMARY.md`) and actual operational status:
- The backend contains two divergent implementations: a legacy flat script (`backend/app.py`) and a modular package (`backend/app/main.py`).
- The machine learning model artifacts (`model.pkl`, `scaler.pkl`) **do not exist** on disk, causing real-time prediction endpoints to fail with HTTP 503.
- The dataset `data/creditcard.csv` is not the 284,807-row Kaggle dataset, but a truncated 20-row sample.
- The unit test suite fails due to a library incompatibility between `passlib 1.7.4` and `bcrypt >= 4.0` in Python 3.14, along with SQLite Windows file locking in `conftest.py`.
- A critical **feature order misalignment bug** exists between data preprocessing in training and real-time inference.
- The frontend consists of legacy static HTML/JavaScript files with endpoints that mismatch the FastAPI router paths (`/alerts` vs `/api/v1/alerts`).

---

## 2. Repository Layout & Artifacts

```
CCRD/
├── .github/workflows/          # CI workflows (tests.yml, docker.yml, codeql.yml)
├── backend/
│   ├── app/                    # Modernized FastAPI application
│   │   ├── api/                # Router definitions (auth, transactions, alerts, settings)
│   │   ├── core/               # Configuration (config.py), logging, security
│   │   ├── database/           # SQLAlchemy session & Base engine
│   │   ├── ml/                 # DataPreprocessor, ModelEvaluator, FraudDetectionModels, Trainer
│   │   ├── models/             # SQLAlchemy ORM models (User, Transaction, Alert)
│   │   └── schemas/            # Pydantic v1-style schemas
│   ├── ml/
│   │   ├── train_model.py      # CLI training entry point
│   │   └── models/             # [EMPTY] Missing model.pkl, scaler.pkl
│   ├── tests/                  # Pytest test suite (29 tests)
│   ├── app.py                  # Legacy flat FastAPI backend (redundant)
│   ├── database.py             # Legacy database script (redundant)
│   ├── models.py               # Legacy models script (redundant)
│   ├── schemas.py              # Legacy schemas script (redundant)
│   ├── utils.py                # Legacy utils with feature ordering bug
│   ├── main.py                 # Application launcher
│   └── fraud.db                # SQLite database file
├── data/
│   └── creditcard.csv          # Truncated 20-row CSV file (5,266 bytes)
├── docs/                       # Existing docs (API.md, DEPLOYMENT.md, etc.)
├── frontend/                   # Static HTML/JS (index.html, alerts.html, login.html, etc.)
├── scripts/                    # Deployment shell/PowerShell scripts
├── Dockerfile                  # Multi-stage Python 3.11 Dockerfile
├── docker-compose.yml          # PostgreSQL 16 + FastAPI service definition
├── package.json                # Root package.json with remnant Angular/Firebase deps
├── next.config.js              # Next.js config remnant
└── vercel.json                 # Vercel deployment config pointing to missing build
```

---

## 3. Detailed Component Audit

### 3.1 Machine Learning Pipeline & Artifacts

1. **Missing Model Artifacts**:
   - `backend/app/core/config.py` defaults to `MODEL_PATH="backend/ml/models/model.pkl"` and `SCALER_PATH="backend/ml/models/scaler.pkl"`.
   - Neither file exists in the repository. On backend startup, `app/main.py` logs `⚠️ ML model files not found` and `app/api/routes/transactions.py` sets `predictor = None`, causing `/api/v1/transactions/predict` to return `503 Service Unavailable`.
2. **Dataset Status**:
   - `data/creditcard.csv` has only 20 rows (10 legitimate, 10 fraud) totaling 5,266 bytes. Training scripts on this file run with an 80/20 split (16 train rows, 4 test rows), making any reported metrics (e.g. 0.9992 accuracy) non-representative.
3. **Feature Ordering Inconsistency (Critical Bug)**:
   - In `data/creditcard.csv`, columns are: `Time, V1, ..., V28, Amount, Class`.
   - In `backend/app/ml/__init__.py:preprocess_input`:
     ```python
     features = np.array([amount, timestamp] + feature_vector, dtype=float)
     ```
   - In `backend/utils.py:preprocess_input`:
     ```python
     X = np.array([data.time, data.amount] + data.features, dtype=float)
     ```
   - If a model is trained on `[Time, V1..V28, Amount]`, feeding `[Amount, Time, V1..V28]` results in total feature distortion.
4. **Imbalance Handling**:
   - `backend/app/ml/imbalancer.py` implements SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, SMOTE-Tomek, SMOTE-ENN, and balanced class weights.
   - However, in `backend/app/ml/trainer.py`, class imbalance techniques are applied before scaling or inside the training loop without full pipeline encapsulation (`imblearn.pipeline.Pipeline`), creating risk of scaling leakage.

### 3.2 Backend Architecture & API

1. **Divergent Codebases**:
   - There are two parallel backends in the same directory:
     - `backend/app.py`: monolithic 258-line file with hardcoded secret fallback, Groq and Google API keys, and routes mapped to `/login`, `/alerts`, `/predict`, `/global-settings`.
     - `backend/app/main.py`: modular FastAPI app routing through `/api/v1/auth`, `/api/v1/transactions`, `/api/v1/alerts`, `/api/v1/settings`.
2. **Pydantic V1 vs V2 Deprecation**:
   - The environment runs Pydantic v2.10+, but schemas in `backend/app/schemas/__init__.py` use v1 syntax (`class Config`, `schema_extra`, `@validator`, `min_items`), generating numerous deprecation warnings on every request.
3. **Database Architecture**:
   - The current DB is SQLite (`fraud.db`) using SQLAlchemy ORM.
   - Models include `User`, `Transaction`, and `Alert`.
   - **No multi-tenancy**: There is no concept of `organizations`, `tenants`, or `api_keys`. Every user shares the same global pool of transactions and alerts.
   - Password hashing in `app/core/security.py` uses `passlib.context.CryptContext(schemes=["bcrypt"])`. Under Python 3.14 with `bcrypt >= 4.0.0`, passlib's internal `detect_wrap_bug` raises `ValueError: password cannot be longer than 72 bytes`, breaking user creation and authentication tests.
   - Plaintext OTP secret: `User.otp_secret` stores a raw 6-digit number in the database.

### 3.3 Frontend

1. **Implementation Status**:
   - The frontend is **not** React, TypeScript, Vite, or Tailwind CSS. It is a set of static HTML files (`index.html`, `login.html`, `alerts.html`, `profile.html`, `setting.html`) with vanilla JS and CSS.
2. **Broken Endpoints**:
   - `frontend/alerts.html` fetches `${API_BASE_URL}/alerts` (404 in modular backend).
   - `frontend/login.html` fetches `${API_BASE_URL}/login` (404 in modular backend).
   - `frontend/setting.html` fetches `${API_BASE_URL}/global-settings` (404 in modular backend).
   - `frontend/profile.html` fetches `${API_BASE_URL}/profile` (non-existent route).
3. **Remnant Configs**:
   - Root has `package.json` with `@angular/fire` and `firebase`.
   - Root has `next.config.js` and `vercel.json` pointing to `cd frontend && npm install && npm run build`, which fails because `frontend/` has no `package.json`.

### 3.4 Test Suite & Quality Verification

1. **Pytest Run Results**:
   - Command: `python -m pytest tests/test_ml.py -p no:langsmith -o addopts=""` -> **6 passed** (feature preprocessing and invalid model path checks).
   - Command on `tests/test_alerts.py`, `tests/test_auth.py`, `tests/test_transactions.py` -> **Fails with 8+ errors**:
     - `ValueError: password cannot be longer than 72 bytes` caused by passlib + bcrypt.
     - `PermissionError: [WinError 32]` during SQLite temp file unlinking on Windows in `conftest.py`.
2. **Missing Coverage Tool**:
   - `pytest.ini` specifies `--cov=app --cov-report=term-missing`, but `pytest-cov` is not registered in the test runner, blocking default `pytest` calls unless overridden.

### 3.5 Security Analysis

1. **Credentials & Secrets**:
   - `.env` in root contains active/dummy API keys (`GROQ_API_KEY`, `GOOGLE_API_KEY`) and a default `SECRET_KEY`. `.gitignore` correctly ignores `.env`.
   - `backend/app.py` has fallback default `"your-super-secret-key"`.
2. **Data Model**:
   - Sensitive financial fields: Currently stores `features` as a raw string without tokenization. No card number or CVV is stored, which is correct.
3. **Tenant Boundary**:
   - There is currently zero tenant isolation or role-based access control beyond a superficial `"admin"` / `"fraud_officer"` string column on `User`.

---

## 4. Documentation vs Reality Comparison

| Claim in Docs / Summary | Verified Source Code Reality | Status |
|:---|:---|:---|
| "Test Coverage: 80%+" | Tests fail to run cleanly due to bcrypt & conftest lock bugs; only 6 ML unit tests pass | ❌ False / Broken |
| "Production Ready SaaS" | Missing model artifacts; SQLite only; no multi-tenancy; 503 on predictions | ❌ Incomplete Prototype |
| "Random Forest AUC = 0.9912" | Evaluated on 20-row sample or unversioned Kaggle run; no serialized pipeline | ⚠️ Unverified |
| "React Frontend Dashboard" | Plain HTML/CSS files with hardcoded outdated API endpoints | ❌ Missing React Stack |
| "Modular Architecture" | Duplicate backends exist (`app.py` vs `app/main.py`) | ⚠️ Partial / Unconsolidated |

---

## 5. Summary of Key Risks & Action Items

1. **Risk of Data Loss / Overwrite**:
   - We must preserve existing research code (`backend/app/ml/`, `backend/ml/train_model.py`) and keep it separate from the new B2B product pipeline.
2. **Technical Debt to Clean Safely**:
   - Resolve passlib/bcrypt hashing in `app/core/security.py`.
   - Fix SQLite test teardown in `backend/tests/conftest.py`.
   - Retire/archive redundant legacy flat files (`backend/app.py`, `utils.py`, `database.py`, `models.py`, `schemas.py`) without deleting them until approved.
3. **Phase 1 Priority**:
   - Build a clean, reproducible, leak-free ML pipeline with an interpretable B2B transaction schema and synthetic data generator.
