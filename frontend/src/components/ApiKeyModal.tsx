import React, { useState, useEffect } from 'react';
import { X, Key, Plus, Trash2, Copy, Check, Terminal, Shield } from 'lucide-react';
import { ApiKeyItem } from '../types';
import { api } from '../services/api';

interface ApiKeyModalProps {
  onClose: () => void;
}

export const ApiKeyModal: React.FC<ApiKeyModalProps> = ({ onClose }) => {
  const [keys, setKeys] = useState<ApiKeyItem[]>([]);
  const [newKeyName, setNewKeyName] = useState('');
  const [freshRawKey, setFreshRawKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadKeys();
  }, []);

  const loadKeys = async () => {
    try {
      const data = await api.listApiKeys();
      setKeys(data);
    } catch {
      setKeys([]);
    }
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newKeyName.trim()) return;
    setLoading(true);
    try {
      const res = await api.createApiKey(newKeyName.trim());
      setFreshRawKey(res.raw_key);
      setNewKeyName('');
      await loadKeys();
    } catch (err: any) {
      alert(`Failed to create API key: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRevoke = async (id: number) => {
    if (!confirm('Are you sure you want to revoke this API key?')) return;
    await api.revokeApiKey(id);
    await loadKeys();
  };

  const copyToClipboard = () => {
    if (!freshRawKey) return;
    navigator.clipboard.writeText(freshRawKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-xl w-full p-6 shadow-2xl text-xs space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center space-x-2">
            <Key className="h-5 w-5 text-amber-400" />
            <h3 className="text-base font-bold text-white tracking-tight">
              Programmatic B2B API Keys
            </h3>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800">
            <X className="h-5 w-5" />
          </button>
        </div>

        <p className="text-slate-400">
          Use API keys with the <code className="text-amber-300 font-mono">X-API-Key</code> header to programmatically score transactions in real time from your payment gateway or backend.
        </p>

        {/* Freshly Minted Secret Banner */}
        {freshRawKey && (
          <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-xs text-white">New API Key Created:</span>
              <button
                onClick={copyToClipboard}
                className="flex items-center space-x-1 px-2 py-0.5 rounded bg-amber-500 text-slate-950 font-bold hover:bg-amber-400 transition-colors"
              >
                {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
            <p className="font-mono text-xs bg-slate-950 p-2 rounded break-all select-all text-white border border-slate-800">
              {freshRawKey}
            </p>
            <span className="text-[11px] text-amber-400 block">
              ⚠️ Store this secret key safely. For security, it will never be displayed again.
            </span>
          </div>
        )}

        {/* Create API Key Form */}
        <form onSubmit={handleCreate} className="flex space-x-2">
          <input
            type="text"
            placeholder="Key Name (e.g. Production Stripe Ingestion)"
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
            className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
          />
          <button
            type="submit"
            disabled={loading || !newKeyName.trim()}
            className="px-3 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-lg font-semibold flex items-center space-x-1"
          >
            <Plus className="h-4 w-4" />
            <span>Generate</span>
          </button>
        </form>

        {/* API Keys List */}
        <div className="space-y-2 pt-2">
          <span className="text-slate-400 font-semibold block uppercase text-[10px] tracking-wider">
            Active Keys ({keys.length})
          </span>
          {keys.length === 0 ? (
            <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800 text-center text-slate-400">
              No API keys configured. Click 'Generate' above to create one.
            </div>
          ) : (
            keys.map((k) => (
              <div
                key={k.id}
                className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 flex items-center justify-between"
              >
                <div>
                  <span className="font-medium text-white block">{k.key_name}</span>
                  <span className="font-mono text-slate-400 text-[11px]">
                    Prefix: <strong className="text-slate-300">{k.key_prefix}...</strong>
                  </span>
                </div>
                <button
                  onClick={() => handleRevoke(k.id)}
                  className="p-1.5 text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 rounded transition-colors"
                  title="Revoke Key"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))
          )}
        </div>

        {/* cURL Example */}
        <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 space-y-1">
          <div className="flex items-center space-x-1 text-slate-400 text-[11px]">
            <Terminal className="h-3 w-3" />
            <span>Example cURL Integration:</span>
          </div>
          <pre className="text-[10px] font-mono text-slate-300 overflow-x-auto p-1.5 bg-slate-900 rounded">
{`curl -X POST http://localhost:8000/api/v1/transactions/score \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: ccrd_live_..." \\
  -d '{"amount": 42.50, "currency": "USD", "merchant_category": "grocery"}'`}
          </pre>
        </div>
      </div>
    </div>
  );
};
