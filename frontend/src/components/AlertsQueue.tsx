import React from 'react';
import { AlertItem } from '../types';
import { AlertOctagon, CheckCircle, Clock, ShieldAlert, XCircle } from 'lucide-react';
import { api } from '../services/api';

interface AlertsQueueProps {
  alerts: AlertItem[];
  onAlertUpdated: () => void;
}

export const AlertsQueue: React.FC<AlertsQueueProps> = ({ alerts, onAlertUpdated }) => {
  const handleUpdate = async (id: number, status: string) => {
    await api.updateAlertStatus(id, status);
    onAlertUpdated();
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl overflow-hidden shadow-lg">
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <AlertOctagon className="h-5 w-5 text-rose-400" />
          <div>
            <h3 className="text-sm font-semibold text-white tracking-tight">
              Fraud Alerts Investigation Queue
            </h3>
            <p className="text-xs text-slate-400">
              Suspicious authorization events flagged for manual or automated resolution.
            </p>
          </div>
        </div>
        <span className="px-2.5 py-1 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20 text-xs font-semibold">
          {alerts.length} Active Alerts
        </span>
      </div>

      <div className="divide-y divide-slate-800/60 max-h-96 overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="p-8 text-center text-slate-400 text-xs">
            No active fraud alerts. Clean transaction stream!
          </div>
        ) : (
          alerts.map((alert) => (
            <div
              key={alert.id}
              className="p-4 hover:bg-slate-800/30 transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
            >
              <div>
                <div className="flex items-center space-x-2">
                  <span className="font-mono font-semibold text-white">
                    {alert.transaction_reference || `TX #${alert.transaction_id}`}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                      alert.severity === 'HIGH'
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                        : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                    }`}
                  >
                    {alert.severity} SEVERITY
                  </span>
                  <span className="text-slate-400 text-[11px]">
                    Status: <strong className="text-slate-300 uppercase">{alert.alert_status}</strong>
                  </span>
                </div>
                <p className="text-slate-400 mt-1">
                  Fraud Score: <strong className="text-white">{(alert.fraud_score * 100).toFixed(1)}%</strong> • Threshold: {alert.threshold}
                </p>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => handleUpdate(alert.id, 'investigating')}
                  className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded font-medium text-[11px] border border-slate-700"
                >
                  Investigate
                </button>
                <button
                  onClick={() => handleUpdate(alert.id, 'resolved')}
                  className="px-2.5 py-1 bg-rose-600/80 hover:bg-rose-600 text-white rounded font-medium text-[11px]"
                >
                  Confirm Fraud
                </button>
                <button
                  onClick={() => handleUpdate(alert.id, 'false_positive')}
                  className="px-2.5 py-1 bg-emerald-600/80 hover:bg-emerald-600 text-white rounded font-medium text-[11px]"
                >
                  False Positive
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
