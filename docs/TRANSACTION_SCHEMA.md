# CCRD — B2B Interpretable Transaction Schema
## Product-Oriented Transaction Data Specification

**Document Version**: 1.0.0  
**Phase**: Phase 1 — ML Foundation  
**Audience**: Data Engineers, ML Engineers, Backend Developers, B2B Integration Clients  

---

## 1. Schema Overview & Design Philosophy

The legacy Kaggle credit card dataset relies entirely on anonymized Principal Component Analysis features (`V1` through `V28`). While suitable for academic benchmarking, these features are **completely uninterpretable** to real merchants, payment gateways, fraud analysts, and compliance regulators. A merchant cannot provide "V14", nor can an analyst explain to an auditor why a transaction was declined based on an abstract PCA projection.

The **CCRD B2B Transaction Schema** is designed around concrete, high-signal, interpretable financial transaction attributes available at the moment of payment authorization.

### Key Principles
1. **Zero Raw Cardholder Data**: No Primary Account Numbers (PAN / full 16-digit card numbers), expiration dates, or CVV/CVC codes are ever accepted or stored.
2. **Deterministic Fraud Signals**: Captures transaction velocity, behavioral divergence from customer baseline, device telemetry, and geographical risk.
3. **Auditable Reason Codes**: Every feature maps directly to clear human-readable explanations (e.g. `VELOCITY_SPIKE`, `DEVICE_UNTRUSTED`, `ABNORMAL_AMOUNT`).

---

## 2. Core Feature Dictionary

| Field Name | Type | Constraints / Valid Range | Example | Description & Risk Signal |
|:---|:---|:---|:---|:---|
| `transaction_reference` | String | Format: `tx_[a-zA-Z0-9_-]{8,32}` or UUID | `tx_8f9a2b1c4e7d` | Unique client-provided transaction identifier for deduplication and tracking. |
| `amount` | Float | `> 0.00`, Max `1,000,000.00` | `149.50` | Monetary value of the authorization request in major currency units. |
| `currency` | String | 3-letter ISO 4217 code (`[A-Z]{3}`) | `USD` | Transaction currency. Normalized during multi-currency scoring. |
| `merchant_category` | String | Enum (8 categories) | `electronics` | High-risk categories (`crypto`, `gaming`, `digital_goods`, `transfer`) exhibit elevated fraud baselines. |
| `transaction_hour` | Integer | `0 <= hour <= 23` | `3` | Local hour of transaction (24-hour clock). Fraud incidence spikes during off-peak hours (02:00–05:00). |
| `country` | String | 2-letter ISO 3166-1 alpha-2 (`[A-Z]{2}`) | `US` | Origin country of transaction or card issuing institution. |
| `customer_age_days` | Integer | `>= 0` | `365` | Age of customer account in days. New accounts (< 7 days) exhibit high bustout risk. |
| `transactions_last_hour` | Integer | `>= 0` | `6` | Number of transactions attempted on the card/account in the trailing 60 minutes (velocity signal). |
| `average_transaction_amount` | Float | `>= 0.00` | `45.20` | Customer's historical 30-day mean transaction value (behavioral baseline). |
| `failed_transactions_last_day` | Integer | `>= 0` | `3` | Count of failed authorization attempts in the trailing 24 hours (card testing / brute force indicator). |
| `device_trust_score` | Float | `0.0 <= score <= 1.0` | `0.15` | Composite device fingerprint and behavioral biometric trust score (0 = untrusted/emulator, 1 = verified). |

---

## 3. Merchant Categories

| Category Code | Description | Typical Fraud Vector | Default Risk Weight |
|:---|:---|:---|:---|
| `retail` | Physical brick-and-mortar / general merchandise | Stolen card testing, chargebacks | Low (1.0x) |
| `grocery` | Supermarkets, food staples, convenience | Friendly fraud, low-dollar testing | Low (0.8x) |
| `travel` | Airlines, hotels, car rentals | Identity theft, synthetic IDs, late chargebacks | Medium (1.5x) |
| `electronics` | High-value consumer tech, smartphones | Fast resale / pawn arbitrage | High (2.2x) |
| `digital_goods` | In-app purchases, digital gift cards, subscriptions | Immediate liquidation, impossible clawback | High (2.5x) |
| `gaming` | Online gambling, gaming currency | Card stuffing, money laundering | High (2.6x) |
| `crypto` | Cryptocurrency purchases and wallet funding | Non-reversible transactions | Critical (3.0x) |
| `transfer` | Peer-to-peer (P2P) transfers, remittance | Account takeover, funnel accounts | Critical (3.0x) |

---

## 4. Derived Features & Feature Engineering

During feature transformation (inside the ML pipeline and rules engine), the following features are deterministically derived:

### 1. `amount_to_avg_ratio`
$$\text{amount\_to\_avg\_ratio} = \frac{\text{amount}}{\text{average\_transaction\_amount} + 1.0}$$
- **Signal**: Captures spend acceleration. A ratio $> 5.0$ indicates that the current transaction is over 5x higher than the customer's typical purchase.

### 2. `is_night_transaction`
$$\text{is\_night\_transaction} = \begin{cases} 1 & \text{if } \text{transaction\_hour} \in \{0, 1, 2, 3, 4, 5\} \\ 0 & \text{otherwise} \end{cases}$$
- **Signal**: Flags authorizations occurring between midnight and 6:00 AM local time.

### 3. `is_high_risk_category`
$$\text{is\_high\_risk\_category} = \begin{cases} 1 & \text{if } \text{merchant\_category} \in \{\text{crypto}, \text{gaming}, \text{digital\_goods}, \text{transfer}\} \\ 0 & \text{otherwise} \end{cases}$$

### 4. `velocity_burst_flag`
$$\text{velocity\_burst\_flag} = \begin{cases} 1 & \text{if } \text{transactions\_last\_hour} \ge 4 \\ 0 & \text{otherwise} \end{cases}$$

### 5. `high_failed_attempts_flag`
$$\text{high\_failed\_attempts\_flag} = \begin{cases} 1 & \text{if } \text{failed\_transactions\_last\_day} \ge 2 \\ 0 & \text{otherwise} \end{cases}$$

### 6. `low_trust_device_flag`
$$\text{low\_trust\_device\_flag} = \begin{cases} 1 & \text{if } \text{device\_trust\_score} \le 0.30 \\ 0 & \text{otherwise} \end{cases}$$

---

## 5. Input Schema Representation (JSON)

```json
{
  "transaction_reference": "tx_9a8b7c6d5e4f",
  "amount": 499.99,
  "currency": "USD",
  "merchant_category": "electronics",
  "transaction_hour": 3,
  "country": "US",
  "customer_age_days": 12,
  "transactions_last_hour": 4,
  "average_transaction_amount": 55.00,
  "failed_transactions_last_day": 2,
  "device_trust_score": 0.18
}
```

---

## 6. Regulatory & Privacy Compliance

- **PCI-DSS Compliance**: The schema does not touch Cardholder Data (CHD) or Sensitive Authentication Data (SAD). Full PAN, expiration date, and CVV are never passed to CCRD.
- **GDPR / CCPA Minimization**: Identifiers are pseudonymized merchant transaction references. No names, personal email addresses, phone numbers, or physical street addresses are required for scoring.
- **Synthetic Data Flag**: In public demo environments, all transactions evaluated under this schema are synthetic or generated via automated test suites.
