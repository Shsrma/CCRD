import React, { useState } from 'react';
import { Play, Sparkles, AlertOctagon, Zap, Sliders, CheckCircle2, AlertTriangle, XCircle, ArrowRight } from 'lucide-react';
import { B2BTransactionPayload, TransactionScoreResponse } from '../types';
import { api } from '../services/api';

interface SimulatorProps {
  onTransactionScored: (result: TransactionScoreResponse) => void;
}

export const Simulator: React.FC<SimulatorProps> = ({ onTransactionScored }) => {
  const [loading, setLoading] = useState(false);
  const [latestResult, setLatestResult] = useState<TransactionScoreResponse | null>(null);
  const [customMode, setCustomMode] = useState(false);

  // Custom form state
  const [customTx, setCustomTx] = useState<B2BTransactionPayload>({
    transaction_reference: '',
    amount: 125.0,
    currency: 'USD',
    merchant_category: 'retail',
    transaction_hour: 14,
    country: 'US',
    customer_age_days: 120,
    transactions_last_hour: 1,
    average_transaction_amount: 80.0,
    failed_transactions_last_day: 0,
    device_trust_score: 0.88,
  });

  const sendScoringRequest = async (payload: B2BTransactionPayload) => {
    setLoading(true);
    try {
      // Ensure unique reference
      const tx = {
        ...payload,
        transaction_reference: `sim_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
      };
      const result = await api.scoreTransaction(tx);
      setLatestResult(result);
      onTransactionScored(result);
    } catch (err: any) {
      alert(`Scoring failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Presets
  const runPreset = (type: 'legit' | 'velocity' | 'ato' | 'new_account') => {
    if (type === 'legit') {
      sendScoringRequest({
        transaction_reference: '',
        amount: 42.50,
        currency: 'USD',
        merchant_category: 'grocery',
        transaction_hour: 15,
        country: 'US',
        customer_age_days: 180,
        transactions_last_hour: 0,
        average_transaction_amount: 40.0,
        failed_transactions_last_day: 0,
        device_trust_score: 0.96,
      });
    } else if (type === 'velocity') {
      sendScoringRequest({
        transaction_reference: '',
        amount: 15.0,
        currency: 'USD',
        merchant_category: 'digital_goods',
        transaction_hour: 3,
        country: 'US',
        customer_age_days: 12,
        transactions_last_hour: 8,
        average_transaction_amount: 25.0,
        failed_transactions_last_day: 4,
        device_trust_score: 0.12,
      });
    } else if (type === 'ato') {
      sendScoringRequest({
        transaction_reference: '',
        amount: 3800.0,
        currency: 'USD',
        merchant_category: 'electronics',
        transaction_hour: 2,
        country: 'MX',
        customer_age_days: 340,
        transactions_last_hour: 2,
        average_transaction_amount: 60.0,
        failed_transactions_last_day: 2,
        device_trust_score: 0.08,
      });
    } else if (type === 'new_account') {
      sendScoringRequest({
        transaction_reference: '',
        amount: 850.0,
        currency: 'USD',
        merchant_category: 'crypto',
        transaction_hour: 20,
        country: 'US',
        customer_age_days: 4,
        transactions_last_hour: 1,
        average_transaction_amount: 120.0,
        failed_transactions_last_day: 1,
        device_trust_score: 0.35,
      });
    }
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-5 shadow-lg">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 gap-2">
        <div>
          <div className="flex items-center space-x-2">
            <Zap className="h-5 w-5 text-amber-400" />
            <h2 className="text-base font-semibold text-white tracking-tight">
              Real-Time Transaction Simulator
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Inject synthetic transactions into the production ML pipeline & rules engine.
          </p>
        </div>

        <button
          onClick={() => setCustomMode(!customMode)}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
            customMode
              ? 'bg-blue-600 text-white border-blue-500'
              : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700'
          }`}
        >
          <Sliders className="h-3.5 w-3.5" />
          <span>{customMode ? 'Hide Custom Builder' : 'Custom Builder'}</span>
        </button>
      </div>

      {/* Preset Buttons */}
      <div className="mt-4 grid grid-cols-2 lg:grid-cols-4 gap-2.5">
        <button
          disabled={loading}
          onClick={() => runPreset('legit')}
          className="flex flex-col items-start p-3 rounded-lg bg-slate-950/70 hover:bg-slate-800/80 border border-emerald-500/30 hover:border-emerald-500/60 transition-all text-left group"
        >
          <div className="flex items-center space-x-1.5 text-emerald-400 text-xs font-semibold">
            <CheckCircle2 className="h-3.5 w-3.5" />
            <span>Normal Purchase</span>
          </div>
          <span className="text-[11px] text-slate-400 mt-1">$42.50 Grocery • High Trust (0.96)</span>
        </button>

        <button
          disabled={loading}
          onClick={() => runPreset('velocity')}
          className="flex flex-col items-start p-3 rounded-lg bg-slate-950/70 hover:bg-slate-800/80 border border-rose-500/30 hover:border-rose-500/60 transition-all text-left group"
        >
          <div className="flex items-center space-x-1.5 text-rose-400 text-xs font-semibold">
            <AlertOctagon className="h-3.5 w-3.5" />
            <span>Velocity Attack</span>
          </div>
          <span className="text-[11px] text-slate-400 mt-1">8 tx/hr • Night Hour • Low Trust</span>
        </button>

        <button
          disabled={loading}
          onClick={() => runPreset('ato')}
          className="flex flex-col items-start p-3 rounded-lg bg-slate-950/70 hover:bg-slate-800/80 border border-rose-500/30 hover:border-rose-500/60 transition-all text-left group"
        >
          <div className="flex items-center space-x-1.5 text-rose-400 text-xs font-semibold">
            <AlertOctagon className="h-3.5 w-3.5" />
            <span>Account Takeover</span>
          </div>
          <span className="text-[11px] text-slate-400 mt-1">$3,800 Spike (63x) • Foreign IP</span>
        </button>

        <button
          disabled={loading}
          onClick={() => runPreset('new_account')}
          className="flex flex-col items-start p-3 rounded-lg bg-slate-950/70 hover:bg-slate-800/80 border border-amber-500/30 hover:border-amber-500/60 transition-all text-left group"
        >
          <div className="flex items-center space-x-1.5 text-amber-400 text-xs font-semibold">
            <AlertTriangle className="h-3.5 w-3.5" />
            <span>New Account Risk</span>
          </div>
          <span className="text-[11px] text-slate-400 mt-1">Age 4d • $850 Crypto Buy</span>
        </button>
      </div>

      {/* Custom Transaction Form */}
      {customMode && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendScoringRequest(customTx);
          }}
          className="mt-4 p-4 rounded-lg bg-slate-950/80 border border-slate-800 text-xs space-y-3"
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div>
              <label className="text-slate-400 block mb-1">Amount (USD)</label>
              <input
                type="number"
                step="0.01"
                value={customTx.amount}
                onChange={(e) => setCustomTx({ ...customTx, amount: parseFloat(e.target.value) || 0 })}
                className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-white"
              />
            </div>

            <div>
              <label className="text-slate-400 block mb-1">Category</label>
              <select
                value={customTx.merchant_category}
                onChange={(e) => setCustomTx({ ...customTx, merchant_category: e.target.value })}
                className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-white"
              >
                <option value="grocery">Grocery</option>
                <option value="retail">Retail</option>
                <option value="dining">Dining</option>
                <option value="electronics">Electronics</option>
                <option value="crypto">Crypto</option>
                <option value="gaming">Gaming</option>
                <option value="gift_cards">Gift Cards</option>
                <option value="travel">Travel</option>
              </select>
            </div>

            <div>
              <label className="text-slate-400 block mb-1">Tx Last Hour</label>
              <input
                type="number"
                value={customTx.transactions_last_hour}
                onChange={(e) => setCustomTx({ ...customTx, transactions_last_hour: parseInt(e.target.value) || 0 })}
                className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-white"
              />
            </div>

            <div>
              <label className="text-slate-400 block mb-1">Device Trust (0 to 1)</label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={customTx.device_trust_score}
                onChange={(e) => setCustomTx({ ...customTx, device_trust_score: parseFloat(e.target.value) || 0 })}
                className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-white"
              />
            </div>
          </div>

          <div className="flex justify-end pt-2">
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium flex items-center space-x-1.5"
            >
              <span>Score Custom Transaction</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </form>
      )}

      {/* Latest Evaluation Result Drawer / Banner */}
      {latestResult && (
        <div className="mt-4 p-4 rounded-xl bg-slate-950 border border-slate-800 animate-in fade-in slide-in-from-top-2">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
            <div>
              <span className="text-[11px] font-mono text-slate-400">REFERENCE: {latestResult.transaction_reference}</span>
              <div className="flex items-center space-x-2 mt-1">
                <span className="text-sm font-semibold text-white">Live Decision Result:</span>
                <span
                  className={`text-xs px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider ${
                    latestResult.recommended_action === 'APPROVE'
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      : latestResult.recommended_action === 'REVIEW'
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                      : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                  }`}
                >
                  {latestResult.recommended_action} ({latestResult.risk_level} RISK)
                </span>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="text-right">
                <div className="text-[11px] text-slate-400">COMPOSITE RISK SCORE</div>
                <div className="text-xl font-bold font-mono text-white">
                  {(latestResult.risk_score * 100).toFixed(1)}%
                </div>
              </div>

              {latestResult.latency_ms !== undefined && (
                <div className="text-right pl-3 border-l border-slate-800">
                  <div className="text-[11px] text-slate-400">LATENCY</div>
                  <div className="text-sm font-bold text-emerald-400 font-mono">
                    {latestResult.latency_ms}ms
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Triggered Rules & Reason Codes */}
          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
            <div>
              <span className="text-slate-400 font-medium block mb-1">Triggered Rules:</span>
              <div className="flex flex-wrap gap-1.5">
                {latestResult.triggered_rules.length > 0 ? (
                  latestResult.triggered_rules.map((rule) => (
                    <span
                      key={rule}
                      className="px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20 font-mono text-[11px]"
                    >
                      {rule}
                    </span>
                  ))
                ) : (
                  <span className="text-slate-400 italic">No deterministic rules triggered (clean transaction)</span>
                )}
              </div>
            </div>

            <div>
              <span className="text-slate-400 font-medium block mb-1">Attributed Reason Codes:</span>
              <div className="flex flex-wrap gap-1.5">
                {latestResult.reason_codes.map((code) => (
                  <span
                    key={code}
                    className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[11px]"
                  >
                    {code}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
