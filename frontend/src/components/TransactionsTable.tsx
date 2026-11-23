import React, { useState } from 'react';
import { TransactionDetail, RiskLevel, RecommendedAction } from '../types';
import { ShieldCheck, AlertTriangle, ShieldX, Eye, ArrowUpDown, ChevronRight } from 'lucide-react';

interface TransactionsTableProps {
  transactions: TransactionDetail[];
  onSelectTransaction: (tx: TransactionDetail) => void;
  filterAction: string;
  onFilterChange: (action: string) => void;
}

export const TransactionsTable: React.FC<TransactionsTableProps> = ({
  transactions,
  onSelectTransaction,
  filterAction,
  onFilterChange,
}) => {
  const getBadge = (action: RecommendedAction) => {
    switch (action) {
      case 'APPROVE':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <ShieldCheck className="h-3 w-3 mr-1" />
            APPROVE
          </span>
        );
      case 'REVIEW':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <AlertTriangle className="h-3 w-3 mr-1" />
            REVIEW
          </span>
        );
      case 'DECLINE':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <ShieldX className="h-3 w-3 mr-1" />
            DECLINE
          </span>
        );
    }
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl overflow-hidden shadow-lg">
      {/* Header & Filter Controls */}
      <div className="p-4 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-white tracking-tight">
            Scored Transactions Stream
          </h3>
          <p className="text-xs text-slate-400">
            Real-time authorization evaluations recorded across merchant accounts.
          </p>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center space-x-1.5 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
          {['ALL', 'APPROVE', 'REVIEW', 'DECLINE'].map((action) => (
            <button
              key={action}
              onClick={() => onFilterChange(action)}
              className={`px-2.5 py-1 rounded font-medium transition-colors ${
                filterAction === action
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {action}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-300">
          <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
            <tr>
              <th className="py-3 px-4">Reference</th>
              <th className="py-3 px-4">Amount</th>
              <th className="py-3 px-4">Category</th>
              <th className="py-3 px-4">Risk Score</th>
              <th className="py-3 px-4">Recommendation</th>
              <th className="py-3 px-4">Decision Rules</th>
              <th className="py-3 px-4 text-right">Inspect</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {transactions.length === 0 ? (
              <tr>
                <td colSpan={7} className="py-8 text-center text-slate-400">
                  No transactions recorded yet. Use the simulator above to inject live transactions!
                </td>
              </tr>
            ) : (
              transactions.map((tx) => {
                let parsedRules: string[] = [];
                try {
                  parsedRules = JSON.parse(tx.triggered_rules || '[]');
                } catch {
                  parsedRules = [];
                }

                return (
                  <tr
                    key={tx.id || tx.transaction_reference}
                    onClick={() => onSelectTransaction(tx)}
                    className="hover:bg-slate-800/40 cursor-pointer transition-colors group"
                  >
                    <td className="py-3 px-4 font-mono text-slate-300">
                      {tx.transaction_reference}
                    </td>
                    <td className="py-3 px-4 font-semibold text-white">
                      ${tx.amount.toFixed(2)} <span className="text-slate-400 font-normal">{tx.currency}</span>
                    </td>
                    <td className="py-3 px-4 capitalize text-slate-300">
                      {tx.merchant_category || 'Retail'}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-2">
                        <div className="w-14 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              tx.risk_score >= 0.70
                                ? 'bg-rose-500'
                                : tx.risk_score >= 0.30
                                ? 'bg-amber-400'
                                : 'bg-emerald-500'
                            }`}
                            style={{ width: `${Math.min(100, Math.max(5, tx.risk_score * 100))}%` }}
                          />
                        </div>
                        <span className="font-mono text-xs font-semibold text-white">
                          {(tx.risk_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      {getBadge(tx.recommended_action)}
                    </td>
                    <td className="py-3 px-4">
                      {parsedRules.length > 0 ? (
                        <div className="flex items-center space-x-1">
                          <span className="px-1.5 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20 text-[10px] font-mono">
                            {parsedRules[0]}
                          </span>
                          {parsedRules.length > 1 && (
                            <span className="text-[10px] text-slate-400">+{parsedRules.length - 1}</span>
                          )}
                        </div>
                      ) : (
                        <span className="text-slate-400 text-[11px]">—</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <ChevronRight className="h-4 w-4 inline text-slate-400 group-hover:text-white transition-colors" />
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
