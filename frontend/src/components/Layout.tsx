import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import {
  Activity,
  BarChart3,
  Cpu,
  Database,
  FileSpreadsheet,
  GitCompare,
  Home,
  Layers,
  Sliders,
  Sparkles,
  UploadCloud,
  ChevronRight,
} from 'lucide-react';
import { useAurora } from '../context/AuroraContext';

const WORKFLOW_STEPS = [
  { path: '/upload', label: '1. Upload Dataset', icon: UploadCloud },
  { path: '/overview', label: '2. Dataset Overview', icon: FileSpreadsheet },
  { path: '/select-approach', label: '3. Model Selection', icon: Layers },
  { path: '/processing', label: '4. Processing', icon: Cpu },
  { path: '/results', label: '5. Results Dashboard', icon: BarChart3 },
  { path: '/comparison', label: '6. Model Comparison', icon: GitCompare },
  { path: '/tuning', label: '7. Hyperparameter Tuning', icon: Sliders },
];

export const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { dataset, latestAnalysis } = useAurora();

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Top Primary Header (Deep Navy) */}
      <header className="sticky top-0 z-40 bg-navy-900 text-white border-b border-slate-800/80 shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <Link to="/" className="flex items-center gap-3 group">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-electric group-hover:scale-105 transition-transform">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-display font-extrabold text-lg tracking-tight text-white">
                    AURORA
                  </span>
                  <span className="text-[10px] font-mono uppercase tracking-widest px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-400/30">
                    Hybrid AutoML
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 hidden sm:block leading-none mt-0.5">
                  Intelligent Model Recommendation System
                </p>
              </div>
            </Link>

            <nav className="hidden md:flex items-center gap-1 ml-4 pl-4 border-l border-slate-800">
              <Link
                to="/dashboard"
                className={`px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-1.5 transition-colors ${
                  location.pathname === '/dashboard'
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/70'
                }`}
              >
                <Home className="w-4 h-4" />
                Dashboard
              </Link>
              <Link
                to="/upload"
                className={`px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-1.5 transition-colors ${
                  location.pathname === '/upload'
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/70'
                }`}
              >
                <UploadCloud className="w-4 h-4" />
                Upload Dataset
              </Link>
              {latestAnalysis && (
                <>
                  <Link
                    to="/results"
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-1.5 transition-colors ${
                      location.pathname === '/results'
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-300 hover:text-white hover:bg-slate-800/70'
                    }`}
                  >
                    <BarChart3 className="w-4 h-4" />
                    Results
                  </Link>
                  <Link
                    to="/comparison"
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-1.5 transition-colors ${
                      location.pathname === '/comparison'
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-300 hover:text-white hover:bg-slate-800/70'
                    }`}
                  >
                    <GitCompare className="w-4 h-4" />
                    Comparison
                  </Link>
                  <Link
                    to="/tuning"
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-1.5 transition-colors ${
                      location.pathname === '/tuning'
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-300 hover:text-white hover:bg-slate-800/70'
                    }`}
                  >
                    <Sliders className="w-4 h-4" />
                    Tuning
                  </Link>
                </>
              )}
            </nav>
          </div>

          {/* Right Status & Quick Action */}
          <div className="flex items-center gap-3">
            {dataset ? (
              <button
                onClick={() => navigate('/overview')}
                className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/90 hover:bg-slate-800 border border-slate-700 text-xs text-slate-200 transition-colors"
                title="View active dataset overview"
              >
                <Database className="w-3.5 h-3.5 text-blue-400" />
                <span className="font-medium truncate max-w-[150px]">{dataset.dataset_name}</span>
                <span className="px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300 font-mono text-[10px] uppercase">
                  {dataset.task_type}
                </span>
              </button>
            ) : (
              <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/60 text-xs text-slate-400 border border-slate-800">
                <Activity className="w-3.5 h-3.5 text-slate-500" />
                No dataset active
              </span>
            )}

            <button
              onClick={() => navigate('/upload')}
              className="px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs sm:text-sm font-semibold shadow-sm transition-all flex items-center gap-1.5"
            >
              <UploadCloud className="w-4 h-4" />
              <span>New Analysis</span>
            </button>
          </div>
        </div>

        {/* Workflow Pipeline Stepper Bar */}
        <div className="bg-navy-950/90 border-t border-slate-800/80">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2 overflow-x-auto">
            <div className="flex items-center gap-1 min-w-max">
              {WORKFLOW_STEPS.map((step, index) => {
                const Icon = step.icon;
                const isActive = location.pathname === step.path;
                return (
                  <React.Fragment key={step.path}>
                    <Link
                      to={step.path}
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        isActive
                          ? 'bg-blue-600/20 text-blue-300 border border-blue-500/40 shadow-sm'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                      }`}
                    >
                      <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-blue-400' : 'text-slate-500'}`} />
                      <span>{step.label}</span>
                    </Link>
                    {index < WORKFLOW_STEPS.length - 1 && (
                      <ChevronRight className="w-3.5 h-3.5 text-slate-700 flex-shrink-0" />
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Compact Application Footer */}
      <footer className="bg-white border-t border-slate-200/80 py-4 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <span className="font-bold text-slate-800">AURORA</span>
            <span>— Intelligent Model Recommendation System</span>
          </div>
          <div className="flex items-center gap-4">
            <Link to="/" className="hover:text-blue-600 transition-colors">Landing Page</Link>
            <Link to="/dashboard" className="hover:text-blue-600 transition-colors">Dashboard</Link>
            <Link to="/upload" className="hover:text-blue-600 transition-colors">Start Analysis</Link>
          </div>
        </div>
      </footer>
    </div>
  );
};

