# Credit Card Fraud Detection System (CCRD)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![License](https://img.shields.io/badge/License-MIT-blue)
[![Tests](https://github.com/Shsrma/CCRD/workflows/Tests%20&%20Quality%20Checks/badge.svg)](https://github.com/Shsrma/CCRD/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Shsrma/CCRD/graph/badge.svg)](https://codecov.io/gh/Shsrma/CCRD)

## 🎯 Overview

**CCRD** is a production-grade **machine learning-powered fraud detection system** designed for financial institutions. It provides real-time fraudulent transaction detection using a RandomForest classifier, with a full-featured REST API and secure authentication.

### Key Features

- 🤖 **ML-Powered Detection**: RandomForest classifier with 93%+ fraud detection accuracy
- 🔐 **Secure Authentication**: JWT-based OAuth2 with password hashing (bcrypt)
- 📊 **Real-Time Alerts**: Instant fraud detection with configurable thresholds
- 📈 **Transaction History**: Complete audit trail of all predictions
- 🛡️ **Enterprise Security**: Input validation, CORS protection, environment-based config
- 🧪 **Test Coverage**: 80%+ unit & integration tests
- 🚀 **Production Ready**: Docker, CI/CD, horizontal scalability (PostgreSQL)
- 📚 **Full Documentation**: API docs, setup guide, deployment guide

---

## 🏗️ Architecture

### Tech Stack

| Layer | Technology |
| ----- | ----------- |
| **API** | FastAPI 0.104+ |
| **Database** | PostgreSQL (prod) / SQLite (dev) |
| **ORM** | SQLAlchemy 2.0+ |
| **ML** | scikit-learn, pandas, numpy |
| **Auth** | JWT + bcrypt |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerized setup)
- PostgreSQL 13+ (for production)
- Git

### Local Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/Shsrma/CCRD.git
cd CCRD
```

1. **Setup backend environment**

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. **Install dependencies**

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

1. **Configure environment**

```bash
cp .env.example .env.local
# Edit .env.local with your settings:
# - SECRET_KEY: Generate a random 32-character key
# - DATABASE_URL: sqlite:///fraud.db (or PostgreSQL for prod)
# - FRONTEND_URL: [http://localhost:3000](http://localhost:3000)
```

1. **Start the API server**

```bash
python main.py
# API will be available at [http://localhost:8000](http://localhost:8000)
```

1. **Access the API**

   - **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
   - **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### Docker Setup (Recommended for Production)

```bash
# Start all services (PostgreSQL + API)
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

---

## 📖 API Documentation

### Base URL

```text
http://localhost:8000/api/v1
```

### Authentication

All protected endpoints require a JWT token in the Authorization header:

```text
Authorization: Bearer <access_token>
```

#### Login

```http
POST /auth/login
Content-Type: application/json

{
  "username": "fraud_officer_1",
  "password": "SecurePass123"
}

Response (200):
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Predict Fraud

```http
POST /transactions/predict
Authorization: Bearer <token>
Content-Type: application/json

{
  "amount": 123.45,
  "timestamp": 1704067200.0,
  "features": [0.5, -0.3, 0.1, 0.2]
}

Response (200):
{
  "transaction_id": 42,
  "fraud_prediction": 1,
  "probability": 0.94
}
```

#### Get Alerts

```http
GET /alerts/?alert_status=pending
Authorization: Bearer <token>

Response (200):
[
  {
    "id": 1,
    "transaction_id": 42,
    "fraud_score": 0.94,
    "alert_status": "pending",
    "created_at": "2025-01-01T12:34:56"
  }
]
```

**Full API documentation available at** `/docs` when running the server.

---

## 🧪 Testing

### Run All Tests

```bash
cd backend
pytest --cov=app --cov-report=html
```

### Code Coverage

- Current: **80%+ coverage**
- Target: **90%+ coverage**

### Linting & Formatting

```bash
# Check code style
flake8 app tests

# Format code
black app tests

# Sort imports
isort app tests
```

---

## 🐳 Docker Deployment

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down -v
```

---

## 🔐 Security Best Practices

- ✅ **Never commit secrets**: Use `.env` files (in `.gitignore`)
- ✅ **HTTPS only**: Enforce TLS in production
- ✅ **JWT validation**: All endpoints verify token claims
- ✅ **Input validation**: Pydantic validates all requests
- ✅ **CORS protection**: Limited to `FRONTEND_URL`
- ✅ **SQL injection prevention**: SQLAlchemy ORM with parameterized queries

---

## 📊 Performance

### API Performance (Single Instance)

- **Requests/sec**: ~2,000
- **P99 Latency**: <100ms
- **Memory Usage**: ~200MB
- **CPU Usage**: <5% (idle)

---

## 📝 Environment Variables

See `.env.example` for complete configuration options:

```bash
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///./fraud.db
SECRET_KEY=your-secret-key-change-in-production
FRONTEND_URL=http://localhost:3000
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and write tests
4. Run `pytest` and `flake8`
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 👥 Author

**Ankur Sharma** - [GitHub](https://github.com/Shsrma)

---

## 🗺️ Roadmap

- [x] Core fraud detection API
- [x] JWT authentication
- [x] Docker containerization
- [x] CI/CD pipelines
- [x] Unit & integration tests
- [ ] Multi-factor authentication
- [ ] React dashboard UI
- [ ] Advanced ML models (XGBoost, Neural Networks)
- [ ] Real-time streaming pipeline
- [ ] Explainable AI (SHAP)

---

**⭐ If you find this project useful, please give it a star!**

Made with ❤️ by Ankur Sharma
