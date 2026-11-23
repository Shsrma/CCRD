import {
  B2BTransactionPayload,
  TransactionScoreResponse,
  TransactionDetail,
  AlertItem,
  ModelMetadata,
  OrganizationInfo,
  ApiKeyItem,
} from '../types';

const API_BASE = '/api/v1';

export const api = {
  /**
   * Score an incoming transaction in real-time.
   */
  async scoreTransaction(
    payload: B2BTransactionPayload,
    apiKey?: string
  ): Promise<TransactionScoreResponse> {
    const start = performance.now();
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (apiKey) {
      headers['X-API-Key'] = apiKey;
    }

    const res = await fetch(`${API_BASE}/transactions/score`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Scoring request failed' }));
      throw new Error(err.error || err.detail || 'Scoring failed');
    }

    const data: TransactionScoreResponse = await res.json();
    data.latency_ms = Math.round(performance.now() - start);
    return data;
  },

  /**
   * Retrieve scored transactions with pagination and optional filters.
   */
  async getTransactions(
    skip = 0,
    limit = 50,
    riskLevel?: string,
    action?: string
  ): Promise<{ total: number; transactions: TransactionDetail[] }> {
    const params = new URLSearchParams({
      skip: String(skip),
      limit: String(limit),
    });
    if (riskLevel && riskLevel !== 'ALL') params.append('risk_level', riskLevel);
    if (action && action !== 'ALL') params.append('action', action);

    const res = await fetch(`${API_BASE}/transactions?${params.toString()}`);
    if (!res.ok) {
      throw new Error('Failed to fetch transactions');
    }
    return res.json();
  },

  /**
   * Retrieve active alerts queue.
   */
  async getAlerts(): Promise<AlertItem[]> {
    const res = await fetch(`${API_BASE}/alerts`);
    if (!res.ok) {
      return [];
    }
    const data = await res.json();
    return Array.isArray(data) ? data : data.alerts || [];
  },

  /**
   * Update the status of an alert (investigating, resolved, false_positive).
   */
  async updateAlertStatus(id: number, status: string): Promise<void> {
    await fetch(`${API_BASE}/alerts/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    }).catch(() => null);
  },

  /**
   * Fetch model metadata, version, registered rules, and evaluation metrics.
   */
  async getModelInfo(): Promise<ModelMetadata> {
    const res = await fetch(`${API_BASE}/model/info`);
    if (!res.ok) {
      throw new Error('Failed to fetch model info');
    }
    return res.json();
  },

  /**
   * Fetch active tenant organization profile.
   */
  async getOrganization(): Promise<OrganizationInfo> {
    const res = await fetch(`${API_BASE}/organizations/me`);
    if (!res.ok) {
      return {
        id: 1,
        name: 'Primary Enterprise Merchant',
        slug: 'default',
        plan: 'enterprise',
        is_active: true,
      };
    }
    return res.json();
  },

  /**
   * List active API keys for caller tenant.
   */
  async listApiKeys(): Promise<ApiKeyItem[]> {
    const res = await fetch(`${API_BASE}/organizations/me/api-keys`);
    if (!res.ok) return [];
    return res.json();
  },

  /**
   * Generate a new B2B API Key.
   */
  async createApiKey(key_name: string): Promise<{ raw_key: string; key_prefix: string; id: number }> {
    const res = await fetch(`${API_BASE}/organizations/me/api-keys`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key_name }),
    });
    if (!res.ok) throw new Error('Failed to generate API Key');
    return res.json();
  },

  /**
   * Revoke an API Key.
   */
  async revokeApiKey(id: number): Promise<void> {
    await fetch(`${API_BASE}/organizations/me/api-keys/${id}`, {
      method: 'DELETE',
    });
  },
};
