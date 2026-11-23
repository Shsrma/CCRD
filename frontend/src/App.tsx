import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { MetricCards } from './components/MetricCards';
import { Simulator } from './components/Simulator';
import { TransactionsTable } from './components/TransactionsTable';
import { AlertsQueue } from './components/AlertsQueue';
import { AnalyticsCharts } from './components/AnalyticsCharts';
import { ModelInfoModal } from './components/ModelInfoModal';
import { ApiKeyModal } from './components/ApiKeyModal';
import { TransactionDetail, AlertItem, ModelMetadata, TransactionScoreResponse } from './types';
import { api } from './services/api';
import { ShieldCheck, AlertTriangle, ShieldX, X, RefreshCw } from 'lucide-react';

export const App: React.FC = () => {
  const [transactions, setTransactions] = useState<TransactionDetail[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelMetadata | null>(null);
  const [selectedTx, setSelectedTx] = useState<TransactionDetail | null>(null);

  // Modals
  const [showModelModal, setShowModelModal] = useState(false);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);

  // Filter
  const [filterAction, setFilterAction] = useState('ALL');

  // Latency tracking
  const [avgLatency, setAvgLatency] = useState(3);

  useEffect(() => {
    loadInitialData();
  }, [filterAction]);

  const loadInitialData = async () => {
    try {
      const [txData, alertData, info] = await Promise.all([
        api.getTransactions(0, 50, undefined, filterAction),
        api.getAlerts(),
        api.getModelInfo(),
      ]);
      setTransactions(txData.transactions || []);
      setAlerts(alertData || []);
      setModelInfo(info);
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
    }
  };

  const handleTransactionScored = (result: TransactionScoreResponse) => {
    if (result.latency_ms) {
      setAvgLatency(result.latency_ms);
    }
    // Refresh table and alerts
    loadInitialData();
  };

  // Metrics computation
  const total = transactions.length;
  const approved = transactions.filter((t) => t.recommended_action === 'APPROVE').length;
  const reviewed = transactions.filter((t) => t.recommended_action === 'REVIEW').length;
  const declined = transactions.filter((t) => t.recommended_action === 'DECLINE').length;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-blue-500 selection:text-white">
      {/* Navigation */}
      <Navbar
        modelInfo={modelInfo}
        onOpenModelInfo={() => setShowModelModal(true)}
        onOpenApiKeys={() => setShowApiKeyModal(true)}
      />

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 lg:px-8 py-6 space-y-6">
        {/* Metric Cards */}
        <MetricCards
          total={total}
          approved={approved}
          reviewed={reviewed}
          declined={declined}
          avgLatencyMs={avgLatency}
        />

        {/* Real-Time Transaction Simulator */}
        <Simulator onTransactionScored={handleTransactionScored} />

        {/* Analytics & Charts */}
        <AnalyticsCharts transactions={transactions} />

        {/* Tables Split View */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <TransactionsTable
              transactions={transactions}
              onSelectTransaction={(tx) => setSelectedTx(tx)}
              filterAction={filterAction}
              onFilterChange={(act) => setFilterAction(act)}
            />
          </div>

          <div>
            <AlertsQueue
              alerts={alerts}
              onAlertUpdated={loadInitialData}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-4 px-4 text-center text-xs text-slate-400">
        <p>
          CCRD Real-Time Fraud Intelligence Platform • B2B Multi-Tenant SaaS Demo • Synthetic Data Engine
        </p>
      </footer>

      {/* Modals */}
      {showModelModal && (
        <ModelInfoModal modelInfo={modelInfo} onClose={() => setShowModelModal(false)} />
      )}

      {showApiKeyModal && (
        <ApiKeyModal onClose={() => setShowApiKeyModal(false)} />
      )}

      {/* Transaction Details Modal */}
      {selectedTx && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 shadow-2xl text-xs space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div>
                <span className="text-[10px] text-slate-400 uppercase tracking-wider">Transaction Inspection</span>
                <h3 className="text-sm font-mono font-bold text-white">{selectedTx.transaction_reference}</h3>
              </div>
              <button onClick={() => setSelectedTx(null)} className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800">
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-2 text-slate-300">
              <div className="p-2 bg-slate-950 rounded">
                <span className="text-slate-400 block text-[10px]">AMOUNT</span>
                <strong className="text-white text-sm">${selectedTx.amount.toFixed(2)} {selectedTx.currency}</strong>
              </div>
              <div className="p-2 bg-slate-950 rounded">
                <span className="text-slate-400 block text-[10px]">MERCHANT CATEGORY</span>
                <strong className="text-white capitalize">{selectedTx.merchant_category || 'Retail'}</strong>
              </div>
              <div className="p-2 bg-slate-950 rounded">
                <span className="text-slate-400 block text-[10px]">COMPOSITE RISK</span>
                <strong className="text-white text-sm font-mono">{(selectedTx.risk_score * 100).toFixed(1)}%</strong>
              </div>
              <div className="p-2 bg-slate-950 rounded">
                <span className="text-slate-400 block text-[10px]">DECISION</span>
                <strong className="text-white uppercase">{selectedTx.recommended_action}</strong>
              </div>
            </div>

            <div className="p-3 bg-slate-950 rounded border border-slate-800 space-y-1 text-slate-400 text-[11px]">
              <div>Customer Account Age: <span className="text-white">{selectedTx.customer_age_days} days</span></div>
              <div>Transactions Last Hour: <span className="text-white">{selectedTx.transactions_last_hour}</span></div>
              <div>Failed Attempts (24h): <span className="text-white">{selectedTx.failed_transactions_last_day}</span></div>
              <div>Device Trust Score: <span className="text-white">{selectedTx.device_trust_score}</span></div>
              <div>Evaluation Model: <span className="text-white font-mono">{selectedTx.model_version}</span></div>
            </div>

            <div className="pt-2">
              <button
                onClick={() => setSelectedTx(null)}
                className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg font-medium"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
