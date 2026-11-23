# CCRD Multi-Tenancy & SaaS Architecture Specification

## 1. Architectural Model
The CCRD Real-Time Fraud Intelligence Platform implements a pooled-schema, row-level logical multi-tenancy model with foreign key tenancy attribution (`organization_id`).

```
                    ┌────────────────────────────┐
                    │      Tenant Ingestion      │
                    │   (X-API-Key / Bearer)     │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │   TenantContext Resolver   │
                    │ (Organization, Config, Key)│
                    └─────────────┬──────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌──────────────────┐                             ┌──────────────────┐
│  Organization A  │                             │  Organization B  │
│  - Custom Low:   │                             │  - Custom Low:   │
│    0.20          │                             │    0.35          │
│  - Custom High:  │                             │  - Custom High:  │
│    0.45          │                             │    0.75          │
│  - Isolated Txs  │                             │  - Isolated Txs  │
│  - Isolated Keys │                             │  - Isolated Keys │
└──────────────────┘                             └──────────────────┘
```

## 2. Security & Data Isolation
- **Authentication**: Supports cryptographically secure API keys (prefix `ccrd_live_` or `ccrd_test_`, SHA-256 hashed in database) and session Bearer JWT tokens.
- **Tenant Isolation**: Every `Transaction`, `FraudPrediction`, and `Alert` entity includes an indexed `organization_id`. Database queries strictly scope results to caller's tenant ID, returning HTTP 403 Forbidden on cross-tenant access attempts.
- **Configurable Boundaries**: Tenants can override standard risk thresholds (`low_threshold`, `high_threshold`) and toggle rules dynamically.
