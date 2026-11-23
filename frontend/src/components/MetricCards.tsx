import React from 'react';
import { Activity, ShieldCheck, AlertTriangle, ShieldX, Clock } from 'lucide-react';

interface MetricCardsProps {
  total: number;
  approved: number;
  reviewed: number;
  declined: number;
  avgLatencyMs: number;
}

export const MetricCards: React.FC<MetricCardsProps> = ({
  total,
  approved,
  reviewed,
  declined,
  avgLatencyMs,
}) => {
  const declineRate = total > 0 ? ((declined / total) * 100).toFixed(1) : '0.0';
  const reviewRate = total > 0 ? ((reviewed / total) * 100).toFixed(1) : '0.0';

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Total Scored */}
      <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-4 shadow-sm hover:border-slate-700/80 transition-all">
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-medium uppercase tracking-wider">Total Scored</span>
          <Activity className="h-4 w-4 text-blue-400" />
        </div>
        <div className="flex items-baseline space-x-2">
          <span className="text-2xl font-bold text-white tracking-tight">{total.toLocaleString()}</span>
          <span className="text-xs text-slate-400">tx authorizations</span>
        </div>
        <p className="text-xs text-slate-400 mt-2 flex items-center space-x-1">
          <Clock className="h-3 w-3 inline text-slate-400" />
          <span>Avg latency ~{avgLatencyMs}ms</span>
        </p>
      </div>

      {/* Approved Traffic */}
      <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-4 shadow-sm hover:border-slate-700/80 transition-all">
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-medium uppercase tracking-wider">Approved (Low Risk)</span>
          <ShieldCheck className="h-4 w-4 text-emerald-400" />
        </div>
        <div className="flex items-baseline space-x-2">
          <span className="text-2xl font-bold text-emerald-400 tracking-tight">{approved.toLocaleString()}</span>
          <span className="text-xs text-slate-400">
            ({total > 0 ? (((approved) / total) * 100).toFixed(1) : 0}%)
          </span>
        </div>
        <p className="text-xs text-slate-400 mt-2">Passed risk thresholds (&lt; 0.30)</p>
      </div>

      {/* Flagged for Review */}
      <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-4 shadow-sm hover:border-slate-700/80 transition-all">
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-medium uppercase tracking-wider">Flagged For Review</span>
          <AlertTriangle className="h-4 w-4 text-amber-400" />
        </div>
        <div className="flex items-baseline space-x-2">
          <span className="text-2xl font-bold text-amber-400 tracking-tight">{reviewed.toLocaleString()}</span>
          <span className="text-xs text-slate-400">({reviewRate}%)</span>
        </div>
        <p className="text-xs text-slate-400 mt-2">Medium risk queue (0.30 - 0.70)</p>
      </div>

      {/* Declined / Fraud */}
      <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-4 shadow-sm hover:border-slate-700/80 transition-all">
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-medium uppercase tracking-wider">Declined (High Risk)</span>
          <ShieldX className="h-4 w-4 text-rose-400" />
        </div>
        <div className="flex items-baseline space-x-2">
          <span className="text-2xl font-bold text-rose-400 tracking-tight">{declined.toLocaleString()}</span>
          <span className="text-xs text-slate-400">({declineRate}%)</span>
        </div>
        <p className="text-xs text-slate-400 mt-2">Blocked via ML + deterministic rules</p>
      </div>
    </div>
  );
};
