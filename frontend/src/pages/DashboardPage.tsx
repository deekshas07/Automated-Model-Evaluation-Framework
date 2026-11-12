import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  ArrowRight,
  BarChart3,
  Clock,
  Cpu,
  Database,
  FileSpreadsheet,
  GitCompare,
  Layers,
  Sliders,
  Sparkles,
  UploadCloud,
  CheckCircle2,
} from 'lucide-react';
import { useAurora } from '../context/AuroraContext';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { dataset, latestAnalysis, history, setLatestAnalysis } = useAurora();

  const handleOpenHistoryItem = (item: any) => {
    setLatestAnalysis(item);
    navigate('/results');
  };

  return (
    <div className="space-y-8">
      {/* Welcome Banner */}
      <div className="rounded-2xl bg-navy-hero text-white p-8 lg:p-10 shadow-premium-lg border border-slate-800 relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-navy pointer-events-none" />
        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div className="space-y-3 max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/20 border border-blue-400/30 text-blue-300 text-xs font-mono">
              <Sparkles className="w-3.5 h-3.5" />
              <span>AURORA Workspace</span>
            </div>
            <h1 className="font-display font-extrabold text-3xl sm:text-4xl tracking-tight text-white">
              Welcome to AURORA
            </h1>
            <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
              Manage your tabular dataset analysis, execute performance-based model evaluation and rule-based model recommendation, compare selection approaches, and optimize hyperparameters.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => navigate('/upload')}
              className="px-6 py-3.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-electric transition-all flex items-center gap-2"
            >
              <UploadCloud className="w-4 h-4" />
              <span>Start New Analysis</span>
            </button>
            {dataset && (
              <button
                onClick={() => navigate('/overview')}
                className="px-5 py-3.5 rounded-xl bg-slate-800/90 hover:bg-slate-800 text-slate-100 font-semibold text-sm border border-slate-700 transition-all flex items-center gap-2"
              >
                <FileSpreadsheet className="w-4 h-4 text-blue-400" />
                <span>Review Active Dataset</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Active Workspace Status Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Card 1: Active Dataset */}
        <div className="aurora-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono uppercase tracking-wider text-slate-500 font-semibold">
                Active Dataset
              </span>
              <Database className="w-4 h-4 text-blue-600" />
            </div>
            {dataset ? (
              <>
                <h3 className="font-display font-bold text-lg text-navy-900 truncate">
                  {dataset.dataset_name}
                </h3>
                <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-600">
                  <span className="px-2 py-0.5 rounded bg-slate-100 font-mono">
                    {dataset.rows.toLocaleString()} rows
                  </span>
                  <span className="px-2 py-0.5 rounded bg-slate-100 font-mono">
                    {dataset.columns} cols
                  </span>
                  <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-700 font-mono uppercase font-semibold">
                    {dataset.task_type}
                  </span>
                </div>
                <p className="mt-2 text-xs text-slate-500">
                  Target column: <strong className="text-slate-700">{dataset.target_column}</strong>
                </p>
              </>
            ) : (
              <>
                <h3 className="font-display font-bold text-lg text-slate-400">
                  No Dataset Loaded
                </h3>
                <p className="mt-1 text-xs text-slate-500 leading-relaxed">
                  Upload a CSV dataset to inspect features and begin model selection.
                </p>
              </>
            )}
          </div>
          <div className="mt-5 pt-3 border-t border-slate-100">
            <Link
              to={dataset ? '/overview' : '/upload'}
              className="text-xs font-semibold text-blue-600 hover:text-blue-700 inline-flex items-center gap-1"
            >
              <span>{dataset ? 'Open Dataset Overview' : 'Upload Dataset'}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>

        {/* Card 2: Latest Model A (Performance-Based) */}
        <div className="aurora-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono uppercase tracking-wider text-blue-600 font-semibold">
                Approach 1 • Model A
              </span>
              <BarChart3 className="w-4 h-4 text-blue-600" />
            </div>
            {latestAnalysis?.approach_1?.selected_model_a ? (
              <>
                <h3 className="font-display font-bold text-lg text-navy-900">
                  {latestAnalysis.approach_1.selected_model_a.Model}
                </h3>
                <p className="mt-1 text-xs text-slate-500">
                  Selected via measured performance across{' '}
                  {latestAnalysis.approach_1.candidate_count} candidate models.
                </p>
                <div className="mt-3 inline-flex items-center gap-2 px-2.5 py-1 rounded-lg bg-blue-50 border border-blue-200/60 text-xs font-mono text-blue-800">
                  {latestAnalysis.task_type === 'classification' ? (
                    <span>
                      Accuracy:{' '}
                      <strong>
                        {(
                          (latestAnalysis.approach_1.selected_model_a.Accuracy || 0) * 100
                        ).toFixed(2)}
                        %
                      </strong>
                    </span>
                  ) : (
                    <span>
                      R² Score:{' '}
                      <strong>
                        {(latestAnalysis.approach_1.selected_model_a['R2 Score'] || 0).toFixed(4)}
                      </strong>
                    </span>
                  )}
                </div>
              </>
            ) : (
              <>
                <h3 className="font-display font-bold text-lg text-slate-400">
                  Not Evaluated Yet
                </h3>
                <p className="mt-1 text-xs text-slate-500 leading-relaxed">
                  Run Approach 1 to train and evaluate candidate models on your dataset.
                </p>
              </>
            )}
          </div>
          <div className="mt-5 pt-3 border-t border-slate-100">
            <Link
              to={latestAnalysis ? '/results' : '/select-approach'}
              className="text-xs font-semibold text-blue-600 hover:text-blue-700 inline-flex items-center gap-1"
            >
              <span>{latestAnalysis ? 'View Results Dashboard' : 'Select Approach'}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>

        {/* Card 3: Latest Model B (Rule-Based Recommendation) */}
        <div className="aurora-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono uppercase tracking-wider text-cyan-700 font-semibold">
                Approach 2 • Model B
              </span>
              <Sparkles className="w-4 h-4 text-cyan-600" />
            </div>
            {latestAnalysis?.approach_2?.selected_model_b ? (
              <>
                <h3 className="font-display font-bold text-lg text-navy-900">
                  {latestAnalysis.approach_2.selected_model_b}
                </h3>
                <p className="mt-1 text-xs text-slate-500">
                  Recommended by AURORA’s rule engine ({latestAnalysis.approach_2.rules_triggered_count}{' '}
                  rules matched).
                </p>
                <div className="mt-3 inline-flex items-center gap-2 px-2.5 py-1 rounded-lg bg-cyan-50 border border-cyan-200/60 text-xs font-mono text-cyan-800">
                  <span>
                    Rule Recommendation Score:{' '}
                    <strong>{latestAnalysis.approach_2.recommendation_score} pts</strong>
                  </span>
                </div>
              </>
            ) : (
              <>
                <h3 className="font-display font-bold text-lg text-slate-400">
                  No Recommendation Yet
                </h3>
                <p className="mt-1 text-xs text-slate-500 leading-relaxed">
                  Run Approach 2 to identify the single highest-scoring model via dataset rules.
                </p>
              </>
            )}
          </div>
          <div className="mt-5 pt-3 border-t border-slate-100">
            <Link
              to={latestAnalysis ? '/comparison' : '/select-approach'}
              className="text-xs font-semibold text-cyan-700 hover:text-cyan-800 inline-flex items-center gap-1"
            >
              <span>{latestAnalysis ? 'Compare Approaches' : 'Select Approach'}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </div>

      {/* Overview of AURORA's Capabilities & Workflow Navigation */}
      <div className="aurora-card p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-6 border-b border-slate-100">
          <div>
            <h2 className="font-display font-bold text-xl text-navy-900">
              AURORA Workflow & Capabilities
            </h2>
            <p className="text-sm text-slate-600 mt-0.5">
              Navigate directly to any stage of the model recommendation and evaluation workflow.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5 mt-6">
          {[
            {
              title: '1. Dataset Upload',
              desc: 'Upload a preprocessed CSV dataset with validation and format verification.',
              path: '/upload',
              icon: UploadCloud,
              badge: 'Input',
            },
            {
              title: '2. Dataset Overview',
              desc: 'Inspect rows, columns, target variable, task type, missingness, and symbolic column analysis.',
              path: '/overview',
              icon: FileSpreadsheet,
              badge: 'Profiling',
            },
            {
              title: '3. Model Selection Approach',
              desc: 'Choose Approach 1 (Performance-Based), Approach 2 (Rule-Based), or Combined Execution.',
              path: '/select-approach',
              icon: Layers,
              badge: 'Selection',
            },
            {
              title: '4. Results Dashboard',
              desc: 'Explore Model A metrics, Model B rule scores, candidate leaderboards, 5-Fold CV, and SHAP.',
              path: '/results',
              icon: BarChart3,
              badge: 'Evaluation',
            },
            {
              title: '5. Model Comparison',
              desc: 'Compare empirical evaluation (Model A) with rule-based recommendation (Model B).',
              path: '/comparison',
              icon: GitCompare,
              badge: 'Comparison',
            },
            {
              title: '6. Hyperparameter Tuning',
              desc: 'Optimize candidate classification models using RandomizedSearchCV and Stratified K-Fold.',
              path: '/tuning',
              icon: Sliders,
              badge: 'Optimization',
            },
          ].map((card) => {
            const Icon = card.icon;
            return (
              <Link
                key={card.path}
                to={card.path}
                className="group p-5 rounded-xl border border-slate-200/80 hover:border-blue-500/40 bg-slate-50/50 hover:bg-white transition-all flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="w-9 h-9 rounded-lg bg-blue-600/10 text-blue-600 flex items-center justify-center group-hover:bg-blue-600 group-hover:text-white transition-colors">
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-slate-200/70 text-slate-700">
                      {card.badge}
                    </span>
                  </div>
                  <h3 className="font-display font-bold text-base text-navy-900 group-hover:text-blue-600 transition-colors">
                    {card.title}
                  </h3>
                  <p className="text-xs text-slate-600 mt-1.5 leading-relaxed">{card.desc}</p>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-200/60 flex items-center justify-between text-xs font-semibold text-blue-600">
                  <span>Open Screen</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Recent Analysis History (Backed by /api/analysis/history) */}
      <div className="aurora-card p-6 sm:p-8">
        <div className="flex items-center justify-between pb-4 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <Clock className="w-5 h-5 text-blue-600" />
            <div>
              <h2 className="font-display font-bold text-lg text-navy-900">
                Recent Analysis Sessions
              </h2>
              <p className="text-xs text-slate-500">
                Recorded runs persisted by the AURORA backend during your workflow sessions.
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-slate-500">
            {history.length} {history.length === 1 ? 'record' : 'records'}
          </span>
        </div>

        {history.length === 0 ? (
          <div className="py-10 text-center space-y-3">
            <Cpu className="w-8 h-8 text-slate-400 mx-auto" />
            <div className="text-sm font-semibold text-slate-700">
              No analysis runs recorded yet
            </div>
            <p className="text-xs text-slate-500 max-w-md mx-auto">
              Upload a dataset and run Approach 1, Approach 2, or both to populate your analysis history.
            </p>
            <button
              onClick={() => navigate('/upload')}
              className="mt-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold transition-all inline-flex items-center gap-1.5"
            >
              <UploadCloud className="w-3.5 h-3.5" />
              <span>Start First Analysis</span>
            </button>
          </div>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-mono uppercase">
                  <th className="py-3 px-3">Dataset</th>
                  <th className="py-3 px-3">Task Type</th>
                  <th className="py-3 px-3">Mode</th>
                  <th className="py-3 px-3">Model A (Performance)</th>
                  <th className="py-3 px-3">Model B (Rule Engine)</th>
                  <th className="py-3 px-3">Completed</th>
                  <th className="py-3 px-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {history.map((item) => (
                  <tr key={item.analysis_id} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-3 px-3 font-semibold text-navy-900">
                      {item.dataset_name}
                      <span className="block text-[11px] font-normal text-slate-400">
                        {item.rows} rows × {item.columns} cols
                      </span>
                    </td>
                    <td className="py-3 px-3">
                      <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-700 font-mono uppercase text-[10px] font-semibold">
                        {item.task_type}
                      </span>
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-600">{item.mode}</td>
                    <td className="py-3 px-3">
                      {item.approach_1?.selected_model_a ? (
                        <div>
                          <span className="font-semibold text-slate-800">
                            {item.approach_1.selected_model_a.Model}
                          </span>
                          <span className="block text-[11px] font-mono text-blue-600">
                            {item.task_type === 'classification'
                              ? `Acc: ${((item.approach_1.selected_model_a.Accuracy || 0) * 100).toFixed(2)}%`
                              : `R²: ${(item.approach_1.selected_model_a['R2 Score'] || 0).toFixed(4)}`}
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-400">—</span>
                      )}
                    </td>
                    <td className="py-3 px-3">
                      {item.approach_2?.selected_model_b ? (
                        <div>
                          <span className="font-semibold text-slate-800">
                            {item.approach_2.selected_model_b}
                          </span>
                          <span className="block text-[11px] font-mono text-cyan-700">
                            Rule Score: {item.approach_2.recommendation_score} pts
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-400">—</span>
                      )}
                    </td>
                    <td className="py-3 px-3 text-slate-500 font-mono">
                      {new Date(item.completed_at).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <button
                        onClick={() => handleOpenHistoryItem(item)}
                        className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-blue-600 hover:text-white text-slate-700 font-semibold transition-colors inline-flex items-center gap-1"
                      >
                        <span>Inspect</span>
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

