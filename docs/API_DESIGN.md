# CCRD API Design Specification
## Real-Time B2B Fraud Intelligence Platform

### 1. Architectural Overview
The CCRD Fraud Intelligence API provides synchronous, low-latency transaction risk evaluation, reason code attribution, and fraud decisioning for B2B financial clients and merchants.

- **Protocol**: HTTP/1.1 REST + JSON
- **Authentication**: Bearer JWT (Session users) and `X-API-Key` (Programmatic B2B integrations)
- **Base URL**: `/api/v1`

---

### 2. Endpoints Reference

#### `POST /api/v1/transactions/score`
Synchronously evaluates risk for an incoming transaction and persists the evaluation.

**Headers:**
- `Content-Type: application/json`
- `Authorization: Bearer <token>` or `X-API-Key: <api_key>`

**Request Body Schema:**
```json
{
  "transaction_reference": "tx_20261123_88921",
  "amount": 2500.00,
  "currency": "USD",
  "merchant_category": "electronics",
  "transaction_hour": 3,
  "country": "US",
  "customer_age_days": 14,
  "transactions_last_hour": 5,
  "average_transaction_amount": 120.00,
  "failed_transactions_last_day": 3,
  "device_trust_score": 0.12
}
```

**Response Body (200 OK):**
```json
{
  "transaction_reference": "tx_20261123_88921",
  "risk_score": 0.8845,
  "risk_level": "HIGH",
  "recommended_action": "DECLINE",
  "reason_codes": [
    "RULE_AMOUNT_SPIKE_VS_BASELINE",
    "RULE_UNUSUAL_NIGHT_TRANSACTION",
    "RULE_MULTIPLE_FAILED_ATTEMPTS",
    "RULE_LOW_DEVICE_TRUST",
    "ML_HIGH_RISK_PATTERN"
  ],
  "triggered_rules": [
    "HIGH_AMOUNT_ANOMALY",
    "UNUSUAL_NIGHT_TRANSACTION",
    "MULTIPLE_FAILED_TRANSACTIONS",
    "UNTRUSTED_DEVICE"
  ],
  "model_version": "1.0.0",
  "is_synthetic_evaluation": true
}
```

**Status Codes:**
- `200 OK`: Evaluation completed and persisted (or idempotent response returned).
- `400 Bad Request`: Payload validation error.
- `401 Unauthorized`: Missing or invalid credentials / API key.
- `422 Unprocessable Entity`: Field range constraint violations.
- `500 Internal Server Error`: Unexpected scoring failure.

---

#### `GET /api/v1/transactions`
Retrieves a paginated list of scored transactions with filtering capabilities.

**Query Parameters:**
- `skip` (int, default: 0): Records to offset.
- `limit` (int, default: 50, max: 500): Page size.
- `risk_level` (string, optional): Filter by `LOW`, `MEDIUM`, or `HIGH`.
- `action` (string, optional): Filter by `APPROVE`, `REVIEW`, or `DECLINE`.

**Response (200 OK):**
```json
{
  "total": 1420,
  "skip": 0,
  "limit": 50,
  "transactions": [ ... ]
}
```

---

#### `GET /api/v1/transactions/{reference_or_id}`
Retrieves detailed information for a single transaction by ID or reference.

---

#### `GET /api/v1/model/info`
Returns metadata on the active fraud model, feature set, registered deterministic rules, and benchmark metrics.

---

#### `GET /api/v1/health`
Liveness and readiness health check probe reporting service status and model operational state.
