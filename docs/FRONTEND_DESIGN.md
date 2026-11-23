# CCRD Frontend & Dashboard Architecture Specification

## 1. Overview
The CCRD user interface is built with React 18, TypeScript, Tailwind CSS, Lucide Icons, and Recharts. It is designed as a mission-control console for fraud risk officers, compliance analysts, and merchant operations teams.

## 2. Core Functional Modules
1. **Live Transaction Simulator**:
   - Presets: Normal Grocery Purchase, High-Velocity Attack, Account Takeover, New Account Risk.
   - Custom Transaction Form: Interactive sliders for amounts, merchant categories, velocity, failed attempts, and device trust.
   - Real-time scoring display with latency measurement in milliseconds.
2. **Scored Transactions Stream**:
   - Filterable stream (`ALL`, `APPROVE`, `REVIEW`, `DECLINE`).
   - Visual risk score bars and triggered rule pills.
   - Detailed transaction inspector drawer.
3. **Alerts Investigation Queue**:
   - Live queue of authorizations flagged for review or blocked.
   - Actionable resolution buttons (`Investigate`, `Confirm Fraud`, `False Positive`).
4. **Analytics & Performance Visualizations**:
   - Recharts Bar Chart: Calibrated Risk Score Distribution histogram.
   - Recharts Donut Chart: Decision Arbitration Ratio split.
5. **Multi-Tenant API Key Manager**:
   - Interface for merchants to generate, view masked prefixes, and revoke `X-API-Key` credentials with cURL integration code snippets.
6. **Model Info & Rules Inspector**:
   - Transparent display of model version, ROC-AUC (0.9995), PR-AUC (0.9972), F1 (0.9634), Platt calibration, and the 6 registered deterministic rules.
