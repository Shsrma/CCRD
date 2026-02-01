# API Documentation

## Base URL

```text
http://localhost:8000/api/v1
```

## Authentication

All protected endpoints require JWT token in header:

```text
Authorization: Bearer <access_token>
```

Tokens expire after 30 minutes.

---

## Authentication Endpoints

### POST /auth/signup

Register a new user account.

**Request:**
```json
{
  "username": "fraud_officer_1",
  "email": "officer@bank.com",
  "password": "SecurePass123"
}
```

**Requirements:**
- Username: 3-100 characters, unique
- Email: Optional, valid email format
- Password: 8+ characters, uppercase + digit required

**Response (201):**
```json
{
  "id": 1,
  "username": "fraud_officer_1",
  "email": "officer@bank.com",
  "role": "fraud_officer",
  "is_active": true,
  "created_at": "2025-01-01T00:00:00"
}
```

**Errors:**
- 400: Username already exists
- 422: Invalid input

---

### POST /auth/login

Authenticate and get JWT token.

**Request:**
```json
{
  "username": "fraud_officer_1",
  "password": "SecurePass123"
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Errors:**
- 401: Invalid credentials
- 403: Account disabled

---

### POST /auth/logout

Logout current user (client-side token deletion).

**Response (200):**
```json
{
  "message": "Successfully logged out. Please delete your token."
}
```

---

## Transaction Endpoints

### POST /transactions/predict

Predict if a transaction is fraudulent.

**Request:**
```json
{
  "amount": 123.45,
  "timestamp": 1704067200.0,
  "features": [0.5, -0.3, 0.1, 0.2, -0.1]
}
```

**Parameters:**
- `amount` (float): Transaction amount, must be > 0
- `timestamp` (float): Unix timestamp
- `features` (array): ML feature vector

**Response (200):**
```json
{
  "transaction_id": 42,
  "fraud_prediction": 1,
  "probability": 0.94
}
```

**Values:**
- `fraud_prediction`: 0 (legitimate) or 1 (fraudulent)
- `probability`: Confidence [0.0, 1.0]

**Errors:**
- 401: Not authenticated
- 422: Invalid input
- 503: ML model unavailable

---

### GET /transactions/history

Get transaction history for current user.

**Query Parameters:**
- `skip` (int, default=0): Pagination offset
- `limit` (int, default=100): Max results (max 1000)

**Response (200):**
```json
{
  "total": 42,
  "transactions": [
    {
      "id": 1,
      "amount": 100.00,
      "timestamp": 1704067200.0,
      "fraud_prediction": 0,
      "fraud_probability": 0.12,
      "created_at": "2025-01-01T00:00:00"
    }
  ]
}
```

---

### GET /transactions/{transaction_id}

Get details of a specific transaction.

**Response (200):**
```json
{
  "id": 42,
  "amount": 123.45,
  "timestamp": 1704067200.0,
  "fraud_prediction": 1,
  "fraud_probability": 0.94,
  "created_at": "2025-01-01T12:34:56"
}
```

**Errors:**
- 404: Transaction not found

---

## Alert Endpoints

### GET /alerts/

Get all fraud alerts.

**Query Parameters:**
- `alert_status` (string): Filter by status
  - `pending` (default)
  - `reviewed`
  - `resolved`
  - `false_positive`
- `skip` (int, default=0): Pagination offset
- `limit` (int, default=100): Max results

**Response (200):**
```json
[
  {
    "id": 1,
    "transaction_id": 42,
    "fraud_score": 0.94,
    "threshold": 0.5,
    "alert_status": "pending",
    "created_at": "2025-01-01T12:34:56",
    "updated_at": "2025-01-01T12:34:56"
  }
]
```

---

### GET /alerts/pending/count

Get count of pending alerts.

**Response (200):**
```json
{
  "pending_count": 5
}
```

---

### GET /alerts/{alert_id}

Get details of a specific alert.

**Response (200):**
```json
{
  "id": 1,
  "transaction_id": 42,
  "fraud_score": 0.94,
  "threshold": 0.5,
  "alert_status": "pending",
  "created_at": "2025-01-01T12:34:56",
  "updated_at": "2025-01-01T12:34:56"
}
```

---

### PATCH /alerts/{alert_id}/status

Update alert status.

**Request:**
```json
{
  "new_status": "resolved"
}
```

**Valid Statuses:**
- `pending`
- `reviewed`
- `resolved`
- `false_positive`

**Response (200):**
```json
{
  "id": 1,
  "transaction_id": 42,
  "fraud_score": 0.94,
  "alert_status": "resolved",
  "updated_at": "2025-01-01T13:00:00"
}
```

---

## Settings Endpoints

### GET /settings/

Get current system settings.

**Response (200):**
```json
{
  "fraud_threshold": 0.5,
  "language": "en",
  "timezone": "UTC"
}
```

---

### PATCH /settings/

Update system settings (admin only).

**Request:**
```json
{
  "fraud_threshold": 0.6,
  "language": "en",
  "timezone": "America/New_York"
}
```

**Response (200):**
```json
{
  "fraud_threshold": 0.6,
  "language": "en",
  "timezone": "America/New_York"
}
```

**Errors:**
- 403: Not an administrator

---

### PATCH /settings/single

Update a single setting.

**Request:**
```json
{
  "type": "fraud_threshold",
  "value": 0.7
}
```

**Response (200):**
```json
{
  "status": "updated",
  "setting": "fraud_threshold",
  "value": 0.7,
  "all_settings": {...}
}
```

---

## System Endpoints

### GET /health

Health check endpoint.

**Response (200):**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### GET /

API root endpoint with metadata.

**Response (200):**
```json
{
  "name": "Credit Card Fraud Detection API",
  "version": "1.0.0",
  "docs": "/docs",
  "openapi": "/openapi.json"
}
```

---

## Error Responses

All errors follow standard HTTP status codes:

### 400 Bad Request
```json
{
  "error": "Invalid request parameters"
}
```

### 401 Unauthorized
```json
{
  "error": "Invalid token"
}
```

### 403 Forbidden
```json
{
  "error": "Insufficient permissions"
}
```

### 404 Not Found
```json
{
  "error": "Resource not found"
}
```

### 422 Unprocessable Entity
```json
{
  "error": "Invalid input data"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error"
}
```

### 503 Service Unavailable
```json
{
  "error": "ML model not available"
}
```

---

## Rate Limiting

- **Current**: No built-in rate limiting
- **Recommended**: Implement via reverse proxy (Nginx, API Gateway)
- **Suggested limits**:
  - 100 requests/minute per user
  - 1000 requests/minute per API key
  - 10000 requests/minute per IP

---

## Pagination

For endpoints returning lists, use:
- `skip`: Offset (0-indexed)
- `limit`: Max results (default 100, max 1000)

```
GET /alerts/?skip=100&limit=50
```

---

## Interactive Documentation

When running the API:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json
