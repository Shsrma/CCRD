import React from 'react';
import { ShieldAlert, Activity, Key, Database, ExternalLink, Sparkles } from 'lucide-react';
import { ModelMetadata } from '../types';

interface NavbarProps {
  modelInfo: ModelMetadata | null;
  onOpenModelInfo: () => void;
  onOpenApiKeys: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  modelInfo,
  onOpenModelInfo,
  onOpenApiKeys,
}) => {
  return (
    <header className="sticky top-0 z-40 bg-slate-950/80 backdrop-blur-md border-b border-slate-800/80 px-4 lg:px-8 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <ShieldAlert className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg tracking-tight text-white">CCRD</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 font-semibold border border-blue-500/20">
                Fraud Intelligence SaaS
              </span>
            </div>
            <p className="text-xs text-slate-400">Real-Time Risk Scoring & Decision Engine</p>
          </div>
        </div>

        {/* Status & Actions */}
        <div className="flex items-center space-x-3">
          {/* Synthetic Demo Pill */}
          <div className="hidden sm:flex items-center space-x-1.5 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs font-medium">
            <Sparkles className="h-3.5 w-3.5 text-amber-400" />
            <span>Synthetic Demonstration</span>
          </div>

          {/* Model Status */}
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="text-slate-300 font-medium">
              {modelInfo?.is_operational ? `Model Online (v${modelInfo.version})` : 'Model Initializing...'}
            </span>
          </div>

          {/* Action Buttons */}
          <button
            onClick={onOpenModelInfo}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-medium text-slate-300 transition-colors"
            title="Inspect Model Metadata & Decision Rules"
          >
            <Database className="h-3.5 w-3.5 text-slate-400" />
            <span className="hidden md:inline">Model Info</span>
          </button>

          <button
            onClick={onOpenApiKeys}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-medium text-slate-300 transition-colors"
            title="Manage Programmatic B2B API Keys"
          >
            <Key className="h-3.5 w-3.5 text-slate-400" />
            <span className="hidden md:inline">API Keys</span>
          </button>

          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            className="flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors shadow-sm shadow-blue-500/30"
          >
            <span>Swagger API</span>
            <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      </div>
    </header>
  );
};
