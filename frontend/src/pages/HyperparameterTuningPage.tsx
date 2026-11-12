import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertCircle,
  ArrowRight,
  Award,
  CheckCircle2,
  Cpu,
  Loader2,
  Sliders,
  Sparkles,
  UploadCloud,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useAurora } from '../context/AuroraContext';
import { auroraApi } from '../services/api';
import { TunedModelResult, TuningConfigurationsResponse } from '../types/aurora';

export const HyperparameterTuningPage: React.FC = () => {
  const navigate = useNavigate();
  const { dataset, latestAnalysis, setLatestAnalysis } = useAurora();

  const [configs, setConfigs] = useState<TuningConfigurationsResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('Random Forest');
  const [selectedStage, setSelectedStage] = useState<string>('approach_1');
  const [isTuning, setIsTuning] = useState<boolean>(false);
  const [tuningError, setTuningError] = useState<string | null>(null);
  const [activeTunedResult, setActiveTunedResult] = useState<TunedModelResult | null>(null);

  useEffect(() => {
    auroraApi
      .getTuningConfigurations()
      .then((res) => {
        setConfigs(res);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (latestAnalysis?.approach_1?.selected_model_a?.Model) {
      const mA = latestAnalysis.approach_1.selected_model_a.Model;
      setSelectedModel(mA);
      if (latestAnalysis.tuning_results && latestAnalysis.tuning_results[mA]) {
        setActiveTunedResult(latestAnalysis.tuning_results[mA]);
      }
    } else if (latestAnalysis?.approach_2?.selected_model_b) {
      setSelectedModel(latestAnalysis.approach_2.selected_model_b);
      setSelectedStage('approach_2');
    }
  }, [latestAnalysis]);

  if (!dataset && !latestAnalysis) {
    return (
      <div className="aurora-card p-12 text-center max-w-2xl mx-auto space-y-4">
        <div className="w-14 h-14 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto">
          <Sliders className="w-7 h-7" />
        </div>
        <h2 className="font-display font-bold text-2xl text-navy-900">
          Dataset Required for Hyperparameter Tuning
        </h2>
        <p className="text-sm text-slate-600 max-w-md mx-auto">
          Upload a dataset and run model selection first before launching RandomizedSearchCV hyperparameter optimization.
        </p>
        <button
          onClick={() => navigate('/upload')}
          className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-electric transition-all inline-flex items-center gap-2"
        >
          <UploadCloud className="w-4 h-4" />
          <span>Upload Dataset</span>
        </button>
      </div>
    );
  }

  const currentTaskType = latestAnalysis?.task_type || dataset?.task_type || 'classification';
  const modelAName = latestAnalysis?.approach_1?.selected_model_a?.Model;
  const modelBName = latestAnalysis?.approach_2?.selected_model_b;

  const availableModels = configs ? Object.keys(configs.models) : [];
  const activeParamSpace = configs?.models?.[selectedModel] || {};

  const handleRunTuning = async () => {
    setIsTuning(true);
    setTuningError(null);
    try {
      const res = await auroraApi.optimizeHyperparameters(selectedModel, selectedStage);
      setActiveTunedResult(res);
      if (latestAnalysis) {
        setLatestAnalysis({
          ...latestAnalysis,
          tuning_results: {
            ...(latestAnalysis.tuning_results || {}),
            [selectedModel]: res,
          },
        });
      }
    } catch (err: any) {
      setTuningError(
        err?.response?.data?.detail ||
          err?.message ||
          'Hyperparameter tuning failed on the backend.'
      );
    } finally {
      setIsTuning(false);
    }
  };

  // Prepare comparison chart if baseline metrics exist
  const comparisonChartData =
    activeTunedResult && activeTunedResult.baseline_metrics
      ? [
          {
            metric: 'Accuracy',
            Baseline: Number((activeTunedResult.baseline_metrics.Accuracy * 100).toFixed(2)),
            Tuned: Number((activeTunedResult.Accuracy * 100).toFixed(2)),
          },
          {
            metric: 'Precision',
            Baseline: Number((activeTunedResult.baseline_metrics.Precision * 100).toFixed(2)),
            Tuned: Number((activeTunedResult.Precision * 100).toFixed(2)),
          },
          {
            metric: 'Recall',
            Baseline: Number((activeTunedResult.baseline_metrics.Recall * 100).toFixed(2)),
            Tuned: Number((activeTunedResult.Recall * 100).toFixed(2)),
          },
          {
            metric: 'F1 Score',
            Baseline: Number((activeTunedResult.baseline_metrics['F1 Score'] * 100).toFixed(2)),
            Tuned: Number((activeTunedResult['F1 Score'] * 100).toFixed(2)),
          },
        ]
      : [];

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-blue-600 font-semibold">
            <span>Hyperparameter Optimization</span>
            <span>•</span>
            <span>RandomizedSearchCV</span>
          </div>
          <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-navy-900 tracking-tight">
            Optimize Your Model
          </h1>
          <p className="text-slate-600 text-base">
            Explore suitable hyperparameter configurations using AURORA’s existing tuning functionality.
          </p>
        </div>

        {configs && (
          <div className="px-4 py-2.5 rounded-xl bg-navy-900 text-white text-xs font-mono flex items-center gap-3">
            <span>Method: {configs.search_method}</span>
            <span>•</span>
            <span>n_iter: {configs.n_iter}</span>
            <span>•</span>
            <span>Scoring: {configs.scoring}</span>
          </div>
        )}
      </div>

      {/* Regression Notice if Current Task is Regression */}
      {currentTaskType === 'regression' && (
        <div className="p-5 rounded-2xl bg-amber-50 border border-amber-200 flex items-start gap-3 text-amber-900">
          <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
          <div className="text-xs sm:text-sm">
            <p className="font-bold">
              Backend Tuning Scope Notice (<code className="font-mono">hyperparameter_tuner.py</code>)
            </p>
            <p className="mt-1 text-amber-800">
              AURORA’s existing <code className="font-mono">tune_selected_model()</code> module is configured specifically for the 15 classification models using <code className="font-mono">StratifiedKFold</code> and <code className="font-mono">f1_weighted</code> scoring. Load a classification dataset to execute live hyperparameter tuning.
            </p>
          </div>
        </div>
      )}

      {/* Configuration & Parameter Space Selection Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Model Selector & Search Space Preview */}
        <div className="lg:col-span-5 aurora-card p-6 space-y-6">
          <div className="border-b border-slate-100 pb-3">
            <h2 className="font-display font-bold text-base text-navy-900">
              1. Select Model & Search Configuration
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Choose Model A, Model B, or any candidate model configured in <code className="font-mono">hyperparameter_tuner.py</code>.
            </p>
          </div>

          {/* Quick-Select Pills for Model A and Model B */}
          {(modelAName || modelBName) && (
            <div className="space-y-2">
              <div className="text-xs font-mono uppercase text-slate-400 font-semibold">
                Identified Models in Current Run:
              </div>
              <div className="flex flex-wrap gap-2">
                {modelAName && (
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedModel(modelAName);
                      setSelectedStage('approach_1');
                      if (latestAnalysis?.tuning_results?.[modelAName]) {
                        setActiveTunedResult(latestAnalysis.tuning_results[modelAName]);
                      }
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all flex items-center gap-1.5 ${
                      selectedModel === modelAName
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100'
                    }`}
                  >
                    <Award className="w-3.5 h-3.5" />
                    <span>Model A: {modelAName}</span>
                  </button>
                )}

                {modelBName && availableModels.includes(modelBName) && (
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedModel(modelBName);
                      setSelectedStage('approach_2');
                      if (latestAnalysis?.tuning_results?.[modelBName]) {
                        setActiveTunedResult(latestAnalysis.tuning_results[modelBName]);
                      }
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all flex items-center gap-1.5 ${
                      selectedModel === modelBName
                        ? 'bg-cyan-600 text-white border-cyan-600'
                        : 'bg-cyan-50 text-cyan-800 border-cyan-200 hover:bg-cyan-100'
                    }`}
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>Model B: {modelBName}</span>
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Dropdown for All Configured Models in hyperparameter_tuner.py */}
          <div className="space-y-2">
            <label className="block text-xs font-mono uppercase text-slate-500 font-semibold">
              Candidate Model Registry (<code className="font-mono">hyperparameter_tuner.py</code>):
            </label>
            <select
              value={selectedModel}
              onChange={(e) => {
                const m = e.target.value;
                setSelectedModel(m);
                if (latestAnalysis?.tuning_results?.[m]) {
                  setActiveTunedResult(latestAnalysis.tuning_results[m]);
                } else {
                  setActiveTunedResult(null);
                }
              }}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-300 text-sm font-semibold text-navy-900 focus:outline-none focus:border-blue-600"
            >
              {availableModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          {/* Parameter Search Space Table */}
          <div className="space-y-2">
            <div className="text-xs font-mono uppercase text-slate-500 font-semibold">
              Configured Parameter Search Space (<code className="font-mono">param_dist</code>):
            </div>
            <div className="rounded-xl bg-slate-50 border border-slate-200/80 divide-y divide-slate-200/60 text-xs">
              {Object.entries(activeParamSpace).map(([param, vals]) => (
                <div key={param} className="p-3 flex flex-col gap-1">
                  <span className="font-mono font-bold text-blue-700">{param}</span>
                  <span className="font-mono text-slate-600">
                    {Array.isArray(vals)
                      ? vals.map((v) => (v === null ? 'None' : String(v))).join(', ')
                      : JSON.stringify(vals)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <button
            type="button"
            disabled={isTuning || currentTaskType === 'regression'}
            onClick={handleRunTuning}
            className={`w-full py-3.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all ${
              isTuning || currentTaskType === 'regression'
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-500 text-white shadow-electric'
            }`}
          >
            {isTuning ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Running RandomizedSearchCV ({selectedModel})...</span>
              </>
            ) : (
              <>
                <Sliders className="w-4 h-4" />
                <span>Run Hyperparameter Tuning for {selectedModel}</span>
              </>
            )}
          </button>

          {tuningError && (
            <div className="p-3.5 rounded-xl bg-red-50 border border-red-200 text-xs text-red-700">
              {tuningError}
            </div>
          )}
        </div>

        {/* Right: Actual Tuning Results Returned by Backend */}
        <div className="lg:col-span-7 aurora-card p-6 sm:p-8 flex flex-col justify-between">
          {activeTunedResult ? (
            <div className="space-y-6">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-4">
                <div>
                  <span className="text-xs font-mono uppercase text-emerald-700 font-semibold flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    Optimization Complete ({activeTunedResult.tuning_time_sec}s)
                  </span>
                  <h2 className="font-display font-extrabold text-2xl text-navy-900 mt-1">
                    Tuned {activeTunedResult.Model}
                  </h2>
                </div>
                <span className="px-3 py-1 rounded-full bg-blue-50 text-blue-700 font-mono text-xs font-semibold border border-blue-200">
                  CV F1-Weighted: {(activeTunedResult['CV Score'] * 100).toFixed(2)}%
                </span>
              </div>

              {/* Best Parameters Discovered */}
              <div className="space-y-2.5">
                <h3 className="text-xs font-mono uppercase text-slate-500 font-semibold">
                  Best Hyperparameters Found (<code className="font-mono">search.best_params_</code>):
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {Object.entries(activeTunedResult['Best Parameters']).map(([k, v]) => (
                    <div
                      key={k}
                      className="p-3.5 rounded-xl bg-navy-900 text-white border border-slate-800"
                    >
                      <div className="text-[11px] font-mono text-blue-300 truncate">{k}</div>
                      <div className="text-sm font-mono font-bold text-white mt-1">
                        {v === null ? 'None' : String(v)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Tuned Evaluation Metrics */}
              <div className="space-y-2.5">
                <h3 className="text-xs font-mono uppercase text-slate-500 font-semibold">
                  Tuned Test-Set Evaluation Metrics:
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3.5 rounded-xl bg-blue-50/80 border border-blue-200">
                    <div className="text-[11px] font-mono uppercase text-blue-700">Accuracy</div>
                    <div className="text-lg font-mono font-bold text-navy-900 mt-0.5">
                      {(activeTunedResult.Accuracy * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                    <div className="text-[11px] font-mono uppercase text-slate-500">Precision</div>
                    <div className="text-lg font-mono font-bold text-navy-900 mt-0.5">
                      {(activeTunedResult.Precision * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                    <div className="text-[11px] font-mono uppercase text-slate-500">Recall</div>
                    <div className="text-lg font-mono font-bold text-navy-900 mt-0.5">
                      {(activeTunedResult.Recall * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="p-3.5 rounded-xl bg-cyan-50/80 border border-cyan-200">
                    <div className="text-[11px] font-mono uppercase text-cyan-800">F1 Score</div>
                    <div className="text-lg font-mono font-bold text-navy-900 mt-0.5">
                      {(activeTunedResult['F1 Score'] * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Baseline vs. Tuned Chart */}
              {comparisonChartData.length > 0 && (
                <div className="space-y-3 pt-2">
                  <h3 className="text-xs font-mono uppercase text-slate-500 font-semibold">
                    Default Baseline vs. Tuned Model Performance:
                  </h3>
                  <div className="h-56 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={comparisonChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                        <XAxis dataKey="metric" tick={{ fontSize: 12, fill: '#334155' }} />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#475569' }} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#0A1128',
                            borderColor: '#1E293B',
                            borderRadius: '12px',
                            color: '#FFFFFF',
                            fontSize: '12px',
                          }}
                        />
                        <Legend wrapperStyle={{ fontSize: '12px' }} />
                        <Bar dataKey="Baseline" name="Default Parameters (%)" fill="#94A3B8" radius={[6, 6, 0, 0]} />
                        <Bar dataKey="Tuned" name="Tuned Model (%)" fill="#2563EB" radius={[6, 6, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              <div className="pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500 font-mono">
                <span>Saved Artifact: {activeTunedResult['Model Path']}</span>
                <span>{activeTunedResult.search_method}</span>
              </div>
            </div>
          ) : (
            <div className="my-auto py-16 text-center space-y-3">
              <div className="w-14 h-14 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center mx-auto">
                <Cpu className="w-7 h-7" />
              </div>
              <h3 className="font-display font-bold text-lg text-navy-900">
                Ready to Optimize {selectedModel}
              </h3>
              <p className="text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
                Click "Run Hyperparameter Tuning for {selectedModel}" on the left to execute <code className="font-mono">RandomizedSearchCV</code> on the backend and view the discovered best parameters and test metrics.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

