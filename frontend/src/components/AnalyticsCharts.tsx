import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';
import { TransactionDetail } from '../types';

interface AnalyticsChartsProps {
  transactions: TransactionDetail[];
}

export const AnalyticsCharts: React.FC<AnalyticsChartsProps> = ({ transactions }) => {
  // 1. Risk score buckets
  const buckets = [
    { range: '0-20%', count: 0 },
    { range: '20-40%', count: 0 },
    { range: '40-60%', count: 0 },
    { range: '60-80%', count: 0 },
    { range: '80-100%', count: 0 },
  ];

  // 2. Category distribution
  const categoryMap: Record<string, { category: string; total: number; fraud: number }> = {};

  // 3. Action distribution
  let approvedCount = 0;
  let reviewCount = 0;
  let declineCount = 0;

  transactions.forEach((tx) => {
    // Risk score bucket
    const score = tx.risk_score || 0;
    if (score < 0.2) buckets[0].count++;
    else if (score < 0.4) buckets[1].count++;
    else if (score < 0.6) buckets[2].count++;
    else if (score < 0.8) buckets[3].count++;
    else buckets[4].count++;

    // Category
    const cat = tx.merchant_category || 'Other';
    if (!categoryMap[cat]) categoryMap[cat] = { category: cat, total: 0, fraud: 0 };
    categoryMap[cat].total++;
    if (tx.recommended_action === 'DECLINE') categoryMap[cat].fraud++;

    // Actions
    if (tx.recommended_action === 'APPROVE') approvedCount++;
    else if (tx.recommended_action === 'REVIEW') reviewCount++;
    else if (tx.recommended_action === 'DECLINE') declineCount++;
  });

  const categoryData = Object.values(categoryMap).slice(0, 6);

  const actionPieData = [
    { name: 'Approved', value: approvedCount, color: '#10b981' },
    { name: 'Review', value: reviewCount, color: '#f59e0b' },
    { name: 'Declined', value: declineCount, color: '#f43f5e' },
  ].filter((d) => d.value > 0);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      {/* Risk Score Distribution Histogram */}
      <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-4 shadow-lg lg:col-span-2">
        <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
          Calibrated Risk Score Distribution
        </h4>
        <p className="text-[11px] text-slate-400 mb-4">
          Frequency of evaluated transactions across probabilistic risk bands.
        </p>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={buckets} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <XAxis dataKey="range" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
              />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Decision Ratio Donut */}
      <div className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-4 shadow-lg flex flex-col justify-between">
        <div>
          <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
            Decision Arbitration Ratio
          </h4>
          <p className="text-[11px] text-slate-400 mb-4">
            Threshold arbitration split across all transactions.
          </p>
        </div>
        <div className="h-52 w-full flex items-center justify-center">
          {actionPieData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={actionPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={45}
                  outerRadius={70}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {actionPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend
                  verticalAlign="bottom"
                  height={24}
                  wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }}
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <span className="text-xs text-slate-400">Awaiting transactions...</span>
          )}
        </div>
      </div>
    </div>
  );
};
