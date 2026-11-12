import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertCircle,
  ArrowRight,
  Award,
  BarChart3,
  CheckCircle2,
  Cpu,
  GitCompare,
  Info,
  Loader2,
  Sliders,
  Sparkles,
  UploadCloud,
} from 'lucide-react';
import { useAurora } from '../context/AuroraContext';
import { auroraApi } from '../services/api';

export const ModelComparisonPage: React.FC = () => {
  const navigate = useNavigate();
  const { latestAnalysis, setLatestAnalysis } = useAurora();

  const [isTrainingB, setIsTrainingB] = useState<boolean>(false);
  const [trainError, setTrainError] = useState<string | null>(null);

  if (!latestAnalysis) {
    return (
      <div className="aurora-card p-12 text-center max-w-2xl mx-auto space-y-4">
        <div className="w-14 h-14 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto">
          <GitCompare className="w-7 h-7" />
        </div>
        <h2 className="font-display font-bold text-2xl text-navy-900">
          No Analysis Available for Comparison
        </h2>
        <p className="text-sm text-slate-600 max-w-md mx-auto">
          Run an analysis first to compare Performance-Based Selection (Approach 1) with Rule-Based Recommendation (Approach 2).
        </p>
        <button
          onClick={() => navigate('/upload')}
          className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-electric transition-all inline-flex items-center gap-2"
        >
          <UploadCloud className="w-4 h-4" />
          <span>Start New Analysis</span>
        </button>
      </div>
    );
  }

  const isClassification = latestAnalysis.task_type === 'classification';
  const app1 = latestAnalysis.approach_1;
  const app2 = latestAnalysis.approach_2;
  const modelA = app1?.selected_model_a;
  const modelBName = app2?.selected_model_b;
  const modelBTrained = app2?.trained_metrics;

  const isSameModel =
    modelA &&
    modelBName &&
    (modelA.Model.toLowerCase() === modelBName.toLowerCase() ||
      (modelBName === 'LDA' && modelA.Model === 'Linear Discriminant Analysis') ||
      (modelBName === 'QDA' && modelA.Model === 'Quadratic Discriminant Analysis') ||
      (modelBName === 'MLP' && modelA.Model === 'Multi Layer Perceptron'));

  const handleTrainModelB = async () => {
    setIsTrainingB(true);
    setTrainError(null);
    try {
      const updated = await auroraApi.trainRecommendedModelB();
      setLatestAnalysis(updated);
    } catch (err: any) {
      setTrainError(
        err?.response?.data?.detail ||
          err?.message ||
          'Could not train recommended Model B.'
      );
    } finally {
      setIsTrainingB(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-blue-600 font-semibold">
            <span>Approach Synthesis</span>
            <span>•</span>
            <span>{latestAnalysis.dataset_name}</span>
          </div>
          <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-navy-900 tracking-tight">
            Compare Model Selection Approaches
          </h1>
          <p className="text-slate-600 text-base">
            Understand the difference between performance-based evaluation and rule-based recommendation.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => navigate('/results')}
            className="px-4 py-2.5 rounded-xl bg-white hover:bg-slate-50 text-slate-700 font-semibold text-xs sm:text-sm border border-slate-200 shadow-sm transition-all inline-flex items-center gap-2"
          >
            <BarChart3 className="w-4 h-4 text-blue-600" />
            <span>Back to Results Dashboard</span>
          </button>

          {isClassification && (
            <button
              onClick={() => navigate('/tuning')}
              className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs sm:text-sm shadow-electric transition-all inline-flex items-center gap-2"
            >
              <Sliders className="w-4 h-4" />
              <span>Proceed to Hyperparameter Tuning</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Agreement / Divergence Banner */}
      {modelA && modelBName && (
        <div
          className={`p-5 rounded-2xl border flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
            isSameModel
              ? 'bg-emerald-50/70 border-emerald-200 text-emerald-950'
              : 'bg-blue-50/70 border-blue-200 text-navy-900'
          }`}
        >
          <div className="flex items-start sm:items-center gap-3.5">
            <div
              className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${
                isSameModel
                  ? 'bg-emerald-600 text-white'
                  : 'bg-blue-600 text-white'
              }`}
            >
              <GitCompare className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-display font-bold text-base">
                {isSameModel
                  ? `Convergent Selection: Both Approaches Identified ${modelA.Model}`
                  : `Complementary Selection: Model A (${modelA.Model}) vs. Model B (${modelBName})`}
              </h3>
              <p className="text-xs text-slate-600 mt-0.5">
                {isSameModel
                  ? 'The empirical performance evaluation (Approach 1) and the dataset heuristic rule engine (Approach 2) independently converged on the same model.'
                  : 'Approach 1 identified Model A through measured empirical metrics on test data, while Approach 2 recommended Model B based on structural dataset characteristics.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Side-by-Side Comparison Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Approach 1 (Model A) */}
        <div className="aurora-card p-6 sm:p-8 space-y-6 border-t-4 border-t-blue-600 flex flex-col justify-between">
          <div className="space-y-5">
            <div className="flex items-center justify-between">
              <span className="px-3 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200 text-xs font-mono font-semibold">
                Approach 1 • Empirical Selection
              </span>
              <Award className="w-5 h-5 text-blue-600" />
            </div>

            {modelA ? (
              <>
                <div>
                  <div className="text-xs font-mono uppercase text-slate-400">
                    Selected Model A
                  </div>
                  <h2 className="font-display font-extrabold text-2xl sm:text-3xl text-navy-900 mt-0.5">
                    {modelA.Model}
                  </h2>
                </div>

                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Selection Mechanism:</span>
                    <span className="font-semibold text-navy-900">
                      Candidate Registry Training & Evaluation
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Candidates Trained:</span>
                    <span className="font-mono font-semibold text-navy-900">
                      {app1?.candidate_count} Models
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Primary Decision Metric:</span>
                    <span className="font-mono font-semibold text-blue-600">
                      {app1?.selection_metric}
                    </span>
                  </div>
                </div>

                {/* Measured Predictive Performance */}
                <div className="space-y-2.5">
                  <div className="text-xs font-mono uppercase text-slate-500 font-semibold">
                    Measured Predictive Performance (Model A):
                  </div>
                  {isClassification ? (
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3.5 rounded-xl bg-blue-50/70 border border-blue-200">
                        <div className="text-[11px] font-mono text-blue-700 uppercase">Accuracy</div>
                        <div className="text-lg font-mono font-bold text-navy-900">
                          {((modelA.Accuracy || 0) * 100).toFixed(2)}%
                        </div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                        <div className="text-[11px] font-mono text-slate-500 uppercase">F1 Score</div>
                        <div className="text-lg font-mono font-bold text-navy-900">
                          {((modelA['F1 Score'] || 0) * 100).toFixed(2)}%
                        </div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                        <div className="text-[11px] font-mono text-slate-500 uppercase">Precision</div>
                        <div className="text-lg font-mono font-bold text-navy-900">
                          {((modelA.Precision || 0) * 100).toFixed(2)}%
                        </div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                        <div className="text-[11px] font-mono text-slate-500 uppercase">Recall</div>
                        <div className="text-lg font-mono font-bold text-navy-900">
                          {((modelA.Recall || 0) * 100).toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3.5 rounded-xl bg-blue-50/70 border border-blue-200">
                        <div className="text-[11px] font-mono text-blue-700 uppercase">R² Score</div>
                        <div className="text-lg font-mono font-bold text-navy-900">
                          {(modelA['R2 Score'] ?? 0).toFixed(4)}
                        </div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                        <div className="text-[11px] font-mono text-slate-500 uppercase">RMSE</div>
                        <div className="text-lg font-mono font-bold text-navy-900">
                          {(modelA.RMSE ?? 0).toFixed(4)}
                        </div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                        <div className="text-[11px] font-mono text-slate-500 uppercase">MAE</div>
                        <div className="text-lg font-mono font-bold text-navy-900">
                          {(modelA.MAE ?? 0).toFixed(4)}
                        </div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                        <div className="text-[11px] font-mono text-slate-500 uppercase">MSE</div>
                        <div className="text-lg font-mono font-bold text-navy-900">
                          {(modelA.MSE ?? 0).toFixed(4)}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="py-10 text-center text-sm text-slate-500">
                Approach 1 was not executed in this session.
              </div>
            )}
          </div>

          <div className="pt-4 border-t border-slate-100 text-xs text-slate-500">
            Empirical validation requires training every candidate model on the dataset split.
          </div>
        </div>

        {/* Right Column: Approach 2 (Model B) */}
        <div className="aurora-card p-6 sm:p-8 space-y-6 border-t-4 border-t-cyan-600 flex flex-col justify-between">
          <div className="space-y-5">
            <div className="flex items-center justify-between">
              <span className="px-3 py-1 rounded-full bg-cyan-50 text-cyan-800 border border-cyan-200 text-xs font-mono font-semibold">
                Approach 2 • Rule-Based Recommendation
              </span>
              <Sparkles className="w-5 h-5 text-cyan-600" />
            </div>

            {app2 && modelBName ? (
              <>
                <div>
                  <div className="text-xs font-mono uppercase text-slate-400">
                    Recommended Model B
                  </div>
                  <h2 className="font-display font-extrabold text-2xl sm:text-3xl text-navy-900 mt-0.5">
                    {modelBName}
                  </h2>
                </div>

                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Selection Mechanism:</span>
                    <span className="font-semibold text-navy-900">
                      Deterministic Heuristic Rule Engine
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Rule Recommendation Score:</span>
                    <span className="font-mono font-bold text-cyan-700">
                      {app2.recommendation_score} points (Heuristic Score)
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Rules Triggered:</span>
                    <span className="font-mono font-semibold text-navy-900">
                      {app2.rules_triggered_count} dataset rules matched
                    </span>
                  </div>
                </div>

                {/* Predictive Performance Status for Model B (Exact Required Notice if Untrained) */}
                <div className="space-y-2.5">
                  <div className="text-xs font-mono uppercase text-slate-500 font-semibold">
                    Predictive Performance Status (Model B):
                  </div>

                  {modelBTrained ? (
                    <div className="space-y-3">
                      {isClassification ? (
                        <div className="grid grid-cols-2 gap-3">
                          <div className="p-3.5 rounded-xl bg-cyan-50/70 border border-cyan-200">
                            <div className="text-[11px] font-mono text-cyan-800 uppercase">
                              Accuracy
                            </div>
                            <div className="text-lg font-mono font-bold text-navy-900">
                              {((modelBTrained.Accuracy || 0) * 100).toFixed(2)}%
                            </div>
                          </div>
                          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                            <div className="text-[11px] font-mono text-slate-500 uppercase">
                              F1 Score
                            </div>
                            <div className="text-lg font-mono font-bold text-navy-900">
                              {((modelBTrained['F1 Score'] || 0) * 100).toFixed(2)}%
                            </div>
                          </div>
                          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                            <div className="text-[11px] font-mono text-slate-500 uppercase">
                              Precision
                            </div>
                            <div className="text-lg font-mono font-bold text-navy-900">
                              {((modelBTrained.Precision || 0) * 100).toFixed(2)}%
                            </div>
                          </div>
                          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                            <div className="text-[11px] font-mono text-slate-500 uppercase">
                              Recall
                            </div>
                            <div className="text-lg font-mono font-bold text-navy-900">
                              {((modelBTrained.Recall || 0) * 100).toFixed(2)}%
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 gap-3">
                          <div className="p-3.5 rounded-xl bg-cyan-50/70 border border-cyan-200">
                            <div className="text-[11px] font-mono text-cyan-800 uppercase">
                              R² Score
                            </div>
                            <div className="text-lg font-mono font-bold text-navy-900">
                              {(modelBTrained['R2 Score'] ?? 0).toFixed(4)}
                            </div>
                          </div>
                          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                            <div className="text-[11px] font-mono text-slate-500 uppercase">
                              RMSE
                            </div>
                            <div className="text-lg font-mono font-bold text-navy-900">
                              {(modelBTrained.RMSE ?? 0).toFixed(4)}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="p-4 rounded-xl bg-amber-50/90 border border-amber-200/90 space-y-3">
                      <div className="flex items-start gap-2.5 text-amber-900">
                        <Info className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                        <div className="text-xs sm:text-sm">
                          <p className="font-bold">
                            Predictive performance is not available for this recommendation.
                          </p>
                          <p className="text-xs text-amber-800 mt-1 leading-relaxed">
                            Approach 2 selects Model B using heuristic rule scoring ({app2.recommendation_score} pts) without model training. The rule recommendation score reflects structural suitability, not predictive accuracy or F1-score.
                          </p>
                        </div>
                      </div>

                      {app2.trainer_available && (
                        <div className="pt-1 flex items-center justify-end">
                          <button
                            type="button"
                            disabled={isTrainingB}
                            onClick={handleTrainModelB}
                            className="px-3.5 py-1.5 rounded-lg bg-navy-900 hover:bg-navy-800 text-white text-xs font-semibold transition-all inline-flex items-center gap-1.5"
                          >
                            {isTrainingB ? (
                              <>
                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                <span>Training {modelBName}...</span>
                              </>
                            ) : (
                              <>
                                <Cpu className="w-3.5 h-3.5" />
                                <span>Train & Evaluate {modelBName} Now</span>
                              </>
                            )}
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  {trainError && (
                    <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700">
                      {trainError}
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="py-10 text-center text-sm text-slate-500">
                Approach 2 was not executed in this session.
              </div>
            )}
          </div>

          <div className="pt-4 border-t border-slate-100 text-xs text-slate-500">
            Rule-based recommendation analyzes dataset dimensionality, feature types, and class balance directly.
          </div>
        </div>
      </div>

      {/* Methodology & Rule Reasoning Breakdown Table */}
      <div className="aurora-card overflow-hidden">
        <div className="p-6 border-b border-slate-100">
          <h3 className="font-display font-bold text-lg text-navy-900">
            Methodology & Architectural Comparison
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            How Approach 1 (Performance-Based Selection) differs from Approach 2 (Rule-Based Recommendation).
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-mono uppercase">
                <th className="py-3.5 px-4">Comparison Dimension</th>
                <th className="py-3.5 px-4 text-blue-700">Approach 1 — Performance-Based</th>
                <th className="py-3.5 px-4 text-cyan-800">Approach 2 — Rule-Based Recommendation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              <tr>
                <td className="py-3.5 px-4 font-semibold text-navy-900">Selected Output</td>
                <td className="py-3.5 px-4 font-bold text-navy-900">
                  {modelA ? `Model A: ${modelA.Model}` : 'Not Executed'}
                </td>
                <td className="py-3.5 px-4 font-bold text-navy-900">
                  {modelBName ? `Model B: ${modelBName}` : 'Not Executed'}
                </td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-semibold text-navy-900">Selection Principle</td>
                <td className="py-3.5 px-4 text-slate-700">
                  Trains all configured candidate models and ranks them by measured test-set evaluation metrics.
                </td>
                <td className="py-3.5 px-4 text-slate-700">
                  Evaluates dataset properties (sample count, feature ratio, data types, missingness, imbalance) against task-specific rules.
                </td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-semibold text-navy-900">Primary Score / Metric</td>
                <td className="py-3.5 px-4 font-mono text-blue-700">
                  {modelA
                    ? isClassification
                      ? `Accuracy: ${((modelA.Accuracy || 0) * 100).toFixed(2)}% | F1: ${((modelA['F1 Score'] || 0) * 100).toFixed(2)}%`
                      : `R² Score: ${(modelA['R2 Score'] ?? 0).toFixed(4)} | RMSE: ${(modelA.RMSE ?? 0).toFixed(4)}`
                    : '—'}
                </td>
                <td className="py-3.5 px-4 font-mono text-cyan-800">
                  {app2
                    ? `Rule Recommendation Score: ${app2.recommendation_score} pts (${app2.rules_triggered_count} rules)`
                    : '—'}
                </td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-semibold text-navy-900">Measured Predictive Performance</td>
                <td className="py-3.5 px-4 text-slate-700">
                  Always available (computed during candidate evaluation).
                </td>
                <td className="py-3.5 px-4 text-slate-700">
                  {modelBTrained
                    ? isClassification
                      ? `Accuracy: ${((modelBTrained.Accuracy || 0) * 100).toFixed(2)}% | F1: ${((modelBTrained['F1 Score'] || 0) * 100).toFixed(2)}%`
                      : `R² Score: ${(modelBTrained['R2 Score'] ?? 0).toFixed(4)}`
                    : 'Predictive performance is not available for this recommendation.'}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

