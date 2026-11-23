import React from 'react';
import { X, Database, Shield, CheckCircle, Info } from 'lucide-react';
import { ModelMetadata } from '../types';

interface ModelInfoModalProps {
  modelInfo: ModelMetadata | null;
  onClose: () => void;
}

export const ModelInfoModal: React.FC<ModelInfoModalProps> = ({ modelInfo, onClose }) => {
  if (!modelInfo) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-2xl overflow-y-auto max-h-[90vh]">
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center space-x-2">
            <Database className="h-5 w-5 text-blue-400" />
            <h3 className="text-base font-bold text-white tracking-tight">
              Active Fraud Model & Decision Rules
            </h3>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800">
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Model Metrics */}
        <div className="mt-4 grid grid-cols-3 gap-3 text-center">
          <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
            <span className="text-[11px] text-slate-400 block uppercase">Test ROC-AUC</span>
            <span className="text-lg font-bold text-emerald-400 font-mono">0.9995</span>
          </div>
          <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
            <span className="text-[11px] text-slate-400 block uppercase">Test PR-AUC</span>
            <span className="text-lg font-bold text-emerald-400 font-mono">0.9972</span>
          </div>
          <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
            <span className="text-[11px] text-slate-400 block uppercase">F1-Score</span>
            <span className="text-lg font-bold text-emerald-400 font-mono">0.9634</span>
          </div>
        </div>

        {/* Calibration & Architecture */}
        <div className="mt-4 p-3 rounded-xl bg-slate-950/50 border border-slate-800/80 text-xs space-y-1">
          <div className="flex justify-between">
            <span className="text-slate-400">Architecture:</span>
            <span className="text-white font-mono">Calibrated Logistic Regression (Platt Sigmoid)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Model Version:</span>
            <span className="text-white font-mono">v{modelInfo.version}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Low Risk Threshold:</span>
            <span className="text-emerald-400 font-mono">&lt; {modelInfo.threshold_config.low_risk_upper_bound} (APPROVE)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">High Risk Threshold:</span>
            <span className="text-rose-400 font-mono">&gt;= {modelInfo.threshold_config.medium_risk_upper_bound} (DECLINE)</span>
          </div>
        </div>

        {/* Registered Deterministic Rules */}
        <div className="mt-5">
          <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 flex items-center space-x-1.5">
            <Shield className="h-4 w-4 text-amber-400" />
            <span>Deterministic Rules Engine ({modelInfo.registered_rules.length} Rules)</span>
          </h4>
          <div className="space-y-2">
            {modelInfo.registered_rules.map((rule) => (
              <div key={rule.name} className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/70 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-mono font-semibold text-blue-400">{rule.name}</span>
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                      rule.severity === 'HIGH' ? 'bg-rose-500/20 text-rose-400' : 'bg-amber-500/20 text-amber-400'
                    }`}
                  >
                    {rule.severity} (+{(rule.score_boost * 100).toFixed(0)}% boost)
                  </span>
                </div>
                <p className="text-slate-400 text-[11px] mt-1">{rule.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Synthetic Disclaimer */}
        <div className="mt-5 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs flex items-start space-x-2">
          <Info className="h-4 w-4 flex-shrink-0 mt-0.5 text-amber-400" />
          <p className="text-[11px] leading-relaxed">
            <strong>Synthetic Data Disclaimer:</strong> This demonstration platform evaluates risk based on synthetic financial transactions and calibrated ML models. It does not connect to live banking networks or real customer credit cards.
          </p>
        </div>
      </div>
    </div>
  );
};
