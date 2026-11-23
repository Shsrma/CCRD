export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH';
export type RecommendedAction = 'APPROVE' | 'REVIEW' | 'DECLINE';
export type AlertStatus = 'pending' | 'reviewed' | 'resolved' | 'false_positive';

export interface B2BTransactionPayload {
  transaction_reference: string;
  amount: number;
  currency: string;
  merchant_category: string;
  transaction_hour: number;
  country: string;
  customer_age_days: number;
  transactions_last_hour: number;
  average_transaction_amount: number;
  failed_transactions_last_day: number;
  device_trust_score: number;
}

export interface TransactionScoreResponse {
  transaction_reference: string;
  risk_score: number;
  risk_level: RiskLevel;
  recommended_action: RecommendedAction;
  reason_codes: string[];
  triggered_rules: string[];
  model_version: string;
  is_synthetic_evaluation: boolean;
  latency_ms?: number;
}

export interface TransactionDetail {
  id: number;
  transaction_reference: string;
  amount: number;
  currency: string;
  merchant_category: string;
  transaction_hour: number;
  country: string;
  customer_age_days: number;
  transactions_last_hour: number;
  average_transaction_amount: number;
  failed_transactions_last_day: number;
  device_trust_score: number;
  risk_score: number;
  risk_level: RiskLevel;
  recommended_action: RecommendedAction;
  reason_codes: string;
  triggered_rules: string;
  model_version: string;
  created_at: string;
}

export interface AlertItem {
  id: number;
  transaction_id: number;
  transaction_reference?: string;
  fraud_score: number;
  threshold: number;
  severity: 'MEDIUM' | 'HIGH';
  alert_status: AlertStatus;
  created_at: string;
}

export interface ModelMetadata {
  model_name: string;
  version: string;
  is_operational: boolean;
  threshold_config: {
    low_risk_upper_bound: number;
    medium_risk_upper_bound: number;
    default_action_threshold: number;
  };
  input_features: string[];
  registered_rules: Array<{
    name: string;
    description: string;
    severity: string;
    score_boost: number;
  }>;
  evaluation_metrics: {
    test_roc_auc?: number;
    test_pr_auc?: number;
    test_f1?: number;
  };
  synthetic_disclaimer: string;
}

export interface OrganizationInfo {
  id: number;
  name: string;
  slug: string;
  plan: string;
  is_active: boolean;
}

export interface ApiKeyItem {
  id: number;
  key_name: string;
  key_prefix: string;
  is_active: boolean;
  created_at: string;
  last_used_at?: string;
}
