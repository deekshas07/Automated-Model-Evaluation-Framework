import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Cpu,
  Database,
  GitCompare,
  Layers,
  Loader2,
  Sparkles,
  UploadCloud,
} from 'lucide-react';
import { useAurora } from '../context/AuroraContext';
import { auroraApi } from '../services/api';
import { AnalysisMode } from '../types/aurora';

export const ModelSelectionPage: React.FC = () => {
  const navigate = useNavigate();
  const {
    dataset,
    selectedApproach,
    setSelectedApproach,
    selectionMetric,
    setSelectionMetric,
    trainModelBOnRun,
    setTrainModelBOnRun,
    setActiveJobId,
    setJobStatus,
  } = useAurora();

  const [isStarting, setIsStarting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  if (!dataset) {
    return (
      <div className="aurora-card p-12 text-center max-w-2xl mx-auto space-y-4">
        <div className="w-14 h-14 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto">
          <Layers className="w-7 h-7" />
        </div>
        <h2 className="font-display font-bold text-2xl text-navy-900">
          Dataset Required Before Model Selection
        </h2>
        <p className="text-sm text-slate-600 max-w-md mx-auto">
          Upload and review a dataset first so AURORA can configure the appropriate classification or regression workflow.
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

  const metricOptions =
    dataset.task_type === 'classification'
      ? ['Accuracy', 'F1 Score', 'Precision', 'Recall']
      : ['R2 Score', 'RMSE', 'MAE'];

  const handleLaunchAnalysis = async () => {
    setIsStarting(true);
    setErrorMsg(null);
    try {
      setJobStatus(null);
      const res = await auroraApi.startAnalysis({
        mode: selectedApproach,
        task_type: dataset.task_type,
        selection_metric: selectionMetric,
        train_model_b: trainModelBOnRun,
      });
      setActiveJobId(res.job_id);
      navigate('/processing');
    } catch (err: any) {
      setErrorMsg(
        err?.response?.data?.detail ||
          err?.message ||
          'Failed to start analysis job on the backend.'
      );
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-blue-600 font-semibold">
            <span>Step 3 of 4</span>
            <span>•</span>
            <span>Workflow Strategy</span>
          </div>
          <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-navy-900 tracking-tight">
            Choose Your Model-Selection Approach
          </h1>
          <p className="text-slate-600 text-base">
            Select how AURORA should identify a suitable machine-learning model.
          </p>
        </div>

        <div className="inline-flex items-center gap-2.5 px-3.5 py-2 rounded-xl bg-white border border-slate-200 shadow-sm text-xs">
          <Database className="w-4 h-4 text-blue-600" />
          <span className="font-semibold text-navy-900">{dataset.dataset_name}</span>
          <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-700 font-mono uppercase font-semibold">
            {dataset.task_type}
          </span>
        </div>
      </div>

      {/* Two Premium Option Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Option Card 1: Approach 1 — Performance-Based Selection */}
        <div
          onClick={() => setSelectedApproach('approach_1')}
          className={`cursor-pointer rounded-2xl p-7 border-2 transition-all flex flex-col justify-between ${
            selectedApproach === 'approach_1'
              ? 'bg-white border-blue-600 shadow-premium-lg ring-4 ring-blue-600/10'
              : 'bg-white border-slate-200/90 hover:border-blue-400 shadow-premium'
          }`}
        >
          <div>
            <div className="flex items-center justify-between mb-5">
              <div className="w-12 h-12 rounded-xl bg-blue-600/10 text-blue-600 flex items-center justify-center">
                <BarChart3 className="w-6 h-6" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono uppercase px-2.5 py-1 rounded-full bg-blue-50 text-blue-700 font-semibold border border-blue-200/60">
                  Outputs Model A
                </span>
                {selectedApproach === 'approach_1' && (
                  <CheckCircle2 className="w-5 h-5 text-blue-600" />
                )}
              </div>
            </div>

            <h2 className="font-display font-bold text-xl text-navy-900 mb-2.5">
              Approach 1 — Performance-Based Selection
            </h2>
            <p className="text-slate-600 text-sm leading-relaxed">
              Evaluate the configured candidate models and identify the best-performing model based on measured evaluation metrics.
            </p>

            <div className="mt-6 pt-5 border-t border-slate-100 space-y-2.5 text-xs text-slate-600">
              <div className="font-mono uppercase text-[11px] text-slate-400 font-semibold">
                Workflow Execution Steps:
              </div>
              <ul className="space-y-1.5">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-600" />
                  <span>Loads candidate model registry for {dataset.task_type}</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-600" />
                  <span>Trains and evaluates each configured candidate model</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-600" />
                  <span>Selects the top-performing model as Model A</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs font-semibold">
            <span className="text-slate-500">Selection Basis: Empirical Metrics</span>
            <span className="text-blue-600 font-mono">
              {selectedApproach === 'approach_1' ? 'Selected' : 'Select Approach 1'}
            </span>
          </div>
        </div>

        {/* Option Card 2: Approach 2 — Rule-Based Recommendation */}
        <div
          onClick={() => setSelectedApproach('approach_2')}
          className={`cursor-pointer rounded-2xl p-7 border-2 transition-all flex flex-col justify-between ${
            selectedApproach === 'approach_2'
              ? 'bg-white border-cyan-600 shadow-premium-lg ring-4 ring-cyan-600/10'
              : 'bg-white border-slate-200/90 hover:border-cyan-500 shadow-premium'
          }`}
        >
          <div>
            <div className="flex items-center justify-between mb-5">
              <div className="w-12 h-12 rounded-xl bg-cyan-600/10 text-cyan-600 flex items-center justify-center">
                <Sparkles className="w-6 h-6" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono uppercase px-2.5 py-1 rounded-full bg-cyan-50 text-cyan-800 font-semibold border border-cyan-200/60">
                  Outputs Model B
                </span>
                {selectedApproach === 'approach_2' && (
                  <CheckCircle2 className="w-5 h-5 text-cyan-600" />
                )}
              </div>
            </div>

            <h2 className="font-display font-bold text-xl text-navy-900 mb-2.5">
              Approach 2 — Rule-Based Recommendation
            </h2>
            <p className="text-slate-600 text-sm leading-relaxed">
              Use AURORA’s existing rule engine to recommend the single highest-scoring model based on dataset characteristics.
            </p>

            <div className="mt-6 pt-5 border-t border-slate-100 space-y-2.5 text-xs text-slate-600">
              <div className="font-mono uppercase text-[11px] text-slate-400 font-semibold">
                Workflow Execution Steps:
              </div>
              <ul className="space-y-1.5">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-600" />
                  <span>
                    Applies {dataset.task_type === 'classification' ? '160 classification' : '85 regression'} heuristic rules
                  </span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-600" />
                  <span>Calculates model recommendation scores & explanations</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-600" />
                  <span>Selects exclusively ONE highest-scoring model as Model B</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs font-semibold">
            <span className="text-slate-500">Selection Basis: Rule Recommendation Score</span>
            <span className="text-cyan-700 font-mono">
              {selectedApproach === 'approach_2' ? 'Selected' : 'Select Approach 2'}
            </span>
          </div>
        </div>
      </div>

      {/* Combined Execution Banner (Supported by Backend) */}
      <div
        onClick={() => setSelectedApproach('both')}
        className={`cursor-pointer rounded-2xl p-6 border-2 transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
          selectedApproach === 'both'
            ? 'bg-navy-900 text-white border-blue-500 shadow-premium-lg'
            : 'bg-white text-navy-900 border-slate-200 hover:border-blue-400 shadow-premium'
        }`}
      >
        <div className="flex items-start sm:items-center gap-4">
          <div
            className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${
              selectedApproach === 'both'
                ? 'bg-blue-500/20 text-blue-300 border border-blue-400/30'
                : 'bg-indigo-50 text-indigo-600'
            }`}
          >
            <GitCompare className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-display font-bold text-lg">
                Execute Both Approaches (Recommended for Full Comparison)
              </h3>
              <span
                className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded-full font-semibold ${
                  selectedApproach === 'both'
                    ? 'bg-blue-500 text-white'
                    : 'bg-blue-100 text-blue-700'
                }`}
              >
                Model A + Model B
              </span>
            </div>
            <p
              className={`text-xs sm:text-sm mt-1 ${
                selectedApproach === 'both' ? 'text-slate-300' : 'text-slate-600'
              }`}
            >
              Run Performance-Based Selection (Approach 1) and Rule-Based Recommendation (Approach 2) together on the same preprocessed dataset to compare Model A and Model B side-by-side.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 self-end sm:self-center flex-shrink-0">
          {selectedApproach === 'both' ? (
            <span className="px-3 py-1.5 rounded-lg bg-blue-600 text-white text-xs font-mono font-semibold flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4" />
              Selected
            </span>
          ) : (
            <span className="px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 text-xs font-mono font-semibold">
              Select Combined Run
            </span>
          )}
        </div>
      </div>

      {/* Evaluation Metric Configuration Bar */}
      {(selectedApproach === 'approach_1' || selectedApproach === 'both') && (
        <div className="aurora-card p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="font-display font-bold text-sm text-navy-900">
              Approach 1 Selection Metric ({dataset.task_type.toUpperCase()})
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Choose the primary measured evaluation metric used to rank candidate models and select Model A.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {metricOptions.map((metric) => (
              <button
                key={metric}
                type="button"
                onClick={() => setSelectionMetric(metric)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
                  selectionMetric === metric
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                {metric}
              </button>
            ))}
          </div>
        </div>
      )}

      {errorMsg && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-xs sm:text-sm text-red-700 font-medium">
          {errorMsg}
        </div>
      )}

      {/* Navigation & Launch Footer */}
      <div className="flex items-center justify-between pt-2">
        <button
          type="button"
          onClick={() => navigate('/overview')}
          className="px-5 py-3 rounded-xl bg-white hover:bg-slate-100 text-slate-700 font-semibold text-sm border border-slate-200 transition-all inline-flex items-center gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Dataset Overview</span>
        </button>

        <button
          type="button"
          disabled={isStarting}
          onClick={handleLaunchAnalysis}
          className="px-8 py-3.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-electric transition-all inline-flex items-center gap-2.5"
        >
          {isStarting ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Starting AURORA Workflow...</span>
            </>
          ) : (
            <>
              <Cpu className="w-4 h-4" />
              <span>Run AURORA Model Selection</span>
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
      </div>
    </div>
  );
};

