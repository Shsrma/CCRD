# Deployment Guide

This guide covers deploying CCRD to various cloud platforms.

## Prerequisites

- Docker & Docker Compose installed
- GitHub account and repository
- Cloud platform account (Render, AWS, Heroku, etc.)
- PostgreSQL database

## Local Testing

Before deploying, test locally:

```bash
# Build Docker image
docker build -t ccrd-api:latest ./backend

# Run with docker-compose
docker-compose up -d

# Test endpoints
curl http://localhost:8000/health

# Stop
docker-compose down
```

---

## Render.com (Easiest - Recommended for Beginners)

### Step 1: Connect GitHub Repository

1. Sign up at [Render.com](https://render.com)
2. Create new Web Service
3. Connect GitHub account
4. Select CCRD repository

### Step 2: Configure Build

**Build Command:**
```bash
cd backend && pip install -r requirements.txt
```

**Start Command:**
```bash
python main.py
```

### Step 3: Environment Variables

Set in Render dashboard:
```
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=<PostgreSQL connection string>
SECRET_KEY=<generate random 32-char string>
FRONTEND_URL=https://your-frontend.com
```

### Step 4: Create PostgreSQL Database

1. In Render dashboard: New → PostgreSQL
2. Create database
3. Copy connection string to `DATABASE_URL`

### Step 5: Deploy

Click "Deploy" - Render will build and deploy automatically!

**Estimated deployment time**: 3-5 minutes

---

## Heroku (Simple)

### Step 1: Install Heroku CLI

```bash
# macOS
brew tap heroku/brew && brew install heroku

# Windows
# Download from: https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Login & Create App

```bash
heroku login
heroku create ccrd-api
```

### Step 3: Add PostgreSQL Add-on

```bash
heroku addons:create heroku-postgresql:standard-0
```

### Step 4: Set Environment Variables

```bash
heroku config:set DEBUG=false
heroku config:set SECRET_KEY=your-secret-key
heroku config:set LOG_LEVEL=INFO
```

### Step 5: Create Procfile

```
web: cd backend && python main.py
```

### Step 6: Deploy

```bash
git push heroku main
```

---

## AWS ECS (Scalable)

### Step 1: Create ECR Repository

```bash
aws ecr create-repository --repository-name ccrd-api

# Get URI
aws ecr describe-repositories --repository-names ccrd-api
```

### Step 2: Build & Push Image

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <ECR_URI>

docker build -t ccrd-api:latest ./backend
docker tag ccrd-api:latest <ECR_URI>/ccrd-api:latest
docker push <ECR_URI>/ccrd-api:latest
```

### Step 3: Create RDS Database

```bash
aws rds create-db-instance \
  --db-instance-identifier ccrd-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --allocated-storage 20 \
  --master-username ccrd_user \
  --master-user-password <strong_password>
```

### Step 4: Create ECS Cluster

Use AWS Console or CLI:
```bash
aws ecs create-cluster --cluster-name ccrd-cluster
```

### Step 5: Create Task Definition

Create `ecs-task-def.json`:
```json
{
  "family": "ccrd-api",
  "containerDefinitions": [
    {
      "name": "ccrd-api",
      "image": "<ECR_URI>/ccrd-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@<RDS_ENDPOINT>:5432/ccrd_db"
        },
        {
          "name": "SECRET_KEY",
          "value": "your-secret-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ccrd-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "cpu": "256",
  "memory": "512"
}
```

Register task definition:
```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-def.json
```

### Step 6: Create Service

```bash
aws ecs create-service \
  --cluster ccrd-cluster \
  --service-name ccrd-api \
  --task-definition ccrd-api \
  --desired-count 2 \
  --launch-type EC2
```

---

## Kubernetes (Enterprise)

### Step 1: Create K8s Manifests

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ccrd-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ccrd-api
  template:
    metadata:
      labels:
        app: ccrd-api
    spec:
      containers:
      - name: ccrd-api
        image: registry.example.com/ccrd-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ccrd-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: ccrd-secrets
              key: secret-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
```

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ccrd-api-service
spec:
  selector:
    app: ccrd-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Step 2: Create Secrets

```bash
kubectl create secret generic ccrd-secrets \
  --from-literal=database-url=postgresql://... \
  --from-literal=secret-key=your-secret-key
```

### Step 3: Deploy

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check status
kubectl get pods
kubectl get services
```

---

## Environment Variables for Production

**Critical - Must Change:**
- `SECRET_KEY`: Use `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- `DATABASE_URL`: Use production PostgreSQL
- `DEBUG`: Set to `false`

**Recommended:**
- `LOG_LEVEL`: Set to `WARNING`
- `FRONTEND_URL`: Set to actual frontend domain
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Consider reducing (default: 30)

---

## Database Migrations

### With Alembic

```bash
# In backend directory
alembic init migrations
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head
```

### Docker Compose

```bash
docker-compose exec backend alembic upgrade head
```

---

## Monitoring & Logging

### CloudWatch (AWS)

```bash
docker run --log-driver awslogs \
  --log-opt awslogs-group=/ecs/ccrd-api \
  ccrd-api:latest
```

### Sentry (Error Tracking)

Add to requirements:
```
sentry-sdk==1.38.0
```

Initialize in app:
```python
import sentry_sdk

sentry_sdk.init(
    dsn="your-sentry-dsn",
    traces_sample_rate=1.0
)
```

---

## SSL/TLS Certificates

### Let's Encrypt (Free)

```bash
# Using Certbot
certbot certonly --standalone -d yourdomain.com

# Renew automatically
sudo systemctl enable certbot.timer
```

### AWS Certificate Manager

Free certificates for AWS resources.

---

## Performance Optimization

### Enable Caching

```python
from fastapi.responses import Response

@app.get("/alerts/pending/count")
async def get_pending_count(response: Response):
    response.headers["Cache-Control"] = "public, max-age=300"
    # ...
```

### Database Connection Pooling

Already configured in SQLAlchemy for production.

### CDN for Static Assets

Use CloudFront (AWS) or Cloudflare for frontend assets.

---

## Scaling Strategies

### Vertical Scaling
- Increase instance size
- More CPU/memory

### Horizontal Scaling
- Multiple API instances
- Load balancer
- Shared database

### Database Scaling
- Read replicas
- Connection pooling
- Caching layer (Redis)

---

## Health Checks

The API provides `/health` endpoint:

```bash
curl http://your-api.com/health
# {"status": "healthy", "version": "1.0.0"}
```

Configure health checks in deployment:
- ECS: Set health check path to `/health`
- K8s: Set liveness probe to `/health`
- Render: Will auto-detect

---

## Rollback Strategy

### Docker
```bash
docker pull old-image-tag
docker run old-image-tag
```

### Kubernetes
```bash
kubectl rollout history deployment/ccrd-api
kubectl rollout undo deployment/ccrd-api --to-revision=2
```

### Render
Automatic rollback on deployment failure.

---

## Support

- Check logs with: `docker-compose logs backend`
- API health: `curl http://localhost:8000/health`
- Database connection: `psql <database_url>`

For issues, create GitHub issue with:
- Platform (Render, AWS, Heroku, etc.)
- Error logs
- Environment configuration (without secrets)
