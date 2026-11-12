import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Award,
  BarChart3,
  CheckCircle2,
  Cpu,
  Eye,
  GitCompare,
  Info,
  Layers,
  Loader2,
  ShieldCheck,
  Sliders,
  Sparkles,
  UploadCloud,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useAurora } from '../context/AuroraContext';
import { auroraApi } from '../services/api';

export const ResultsDashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { latestAnalysis, setLatestAnalysis } = useAurora();

  const [isRunningCV, setIsRunningCV] = useState<boolean>(false);
  const [isRunningShap, setIsRunningShap] = useState<boolean>(false);
  const [shapUrl, setShapUrl] = useState<string | null>(
    latestAnalysis?.shap_available ? `/api/reports/shap-image?t=${Date.now()}` : null
  );
  const [actionError, setActionError] = useState<string | null>(null);

  if (!latestAnalysis) {
    return (
      <div className="aurora-card p-12 text-center max-w-2xl mx-auto space-y-4">
        <div className="w-14 h-14 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto">
          <BarChart3 className="w-7 h-7" />
        </div>
        <h2 className="font-display font-bold text-2xl text-navy-900">
          No Model Selection Results Available
        </h2>
        <p className="text-sm text-slate-600 max-w-md mx-auto">
          Upload a dataset and run AURORA’s model-selection workflow to view Model A, Model B, evaluation metrics, and charts.
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
  const candidates = app1?.candidate_models || [];

  const handleRunCV = async () => {
    if (!modelA) return;
    setIsRunningCV(true);
    setActionError(null);
    try {
      const cvRes = await auroraApi.runCrossValidation(modelA.Model, modelA['Model Path']);
      setLatestAnalysis({
        ...latestAnalysis,
        cross_validation: cvRes,
      });
    } catch (err: any) {
      setActionError(
        err?.response?.data?.detail || err?.message || '5-Fold Cross-Validation failed.'
      );
    } finally {
      setIsRunningCV(false);
    }
  };

  const handleRunShap = async () => {
    if (!modelA) return;
    setIsRunningShap(true);
    setActionError(null);
    try {
      const res = await auroraApi.generateShapPlot(modelA.Model, modelA['Model Path']);
      setShapUrl(res.shap_image_url);
      setLatestAnalysis({
        ...latestAnalysis,
        shap_available: true,
      });
    } catch (err: any) {
      setActionError(
        err?.response?.data?.detail || err?.message || 'SHAP summary plot generation failed.'
      );
    } finally {
      setIsRunningShap(false);
    }
  };

  // Prepare chart data from actual candidate results
  const chartData = candidates.map((c) => {
    if (isClassification) {
      return {
        name: c.Model,
        Accuracy: Number(((c.Accuracy || 0) * 100).toFixed(2)),
        'F1 Score': Number(((c['F1 Score'] || 0) * 100).toFixed(2)),
        Precision: Number(((c.Precision || 0) * 100).toFixed(2)),
        Recall: Number(((c.Recall || 0) * 100).toFixed(2)),
      };
    } else {
      return {
        name: c.Model,
        'R2 Score': Number((c['R2 Score'] || 0).toFixed(4)),
        RMSE: Number((c.RMSE || 0).toFixed(4)),
        MAE: Number((c.MAE || 0).toFixed(4)),
      };
    }
  });

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-blue-600 font-semibold">
            <span>Step 4 of 4</span>
            <span>•</span>
            <span>Dataset: {latestAnalysis.dataset_name}</span>
            <span>•</span>
            <span>{latestAnalysis.task_type.toUpperCase()}</span>
          </div>
          <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-navy-900 tracking-tight">
            Your Model Selection Results
          </h1>
          <p className="text-slate-600 text-base">
            Explore the models identified by AURORA and review their evaluation results.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => navigate('/comparison')}
            className="px-5 py-2.5 rounded-xl bg-white hover:bg-slate-50 text-navy-900 font-semibold text-xs sm:text-sm border border-slate-200 shadow-sm transition-all inline-flex items-center gap-2"
          >
            <GitCompare className="w-4 h-4 text-blue-600" />
            <span>Compare Approaches</span>
          </button>

          {isClassification && (
            <button
              onClick={() => navigate('/tuning')}
              className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs sm:text-sm shadow-electric transition-all inline-flex items-center gap-2"
            >
              <Sliders className="w-4 h-4" />
              <span>Optimize Model Hyperparameters</span>
            </button>
          )}
        </div>
      </div>

      {/* Primary Cards: Model A (Approach 1) & Model B (Approach 2) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model A Card — Selected Through Measured Model Performance */}
        <div className="aurora-card p-6 sm:p-7 border-t-4 border-t-blue-600 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 mb-4">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200 text-xs font-mono font-semibold">
                <Award className="w-3.5 h-3.5" />
                Model A • Approach 1
              </span>
              <span className="text-xs font-mono text-slate-500">
                Selected through measured model performance
              </span>
            </div>

            {modelA ? (
              <>
                <h2 className="font-display font-extrabold text-2xl sm:text-3xl text-navy-900">
                  {modelA.Model}
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                  Best-performing model across {app1?.candidate_count} trained candidate models (ranked by{' '}
                  <strong className="text-slate-700">{app1?.selection_metric}</strong>).
                </p>

                {/* Measured Metrics Grid */}
                {isClassification ? (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6">
                    <div className="p-3.5 rounded-xl bg-blue-50/70 border border-blue-200/70">
                      <div className="text-[11px] font-mono uppercase text-blue-700">Accuracy</div>
                      <div className="text-lg font-mono font-bold text-navy-900 mt-0.5">
                        {((modelA.Accuracy || 0) * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
                      <div className="text-[11px] font-mono uppercase text-slate-500">Precision</div>
                      <div className="text-lg font-mono font-bold text-navy-900 mt-0.5">
                        {((modelA.Precision || 0) * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
                      <div className="text-[11px] font-mono uppercase text-slate-500">Recall</div>
                      <div className="text-lg font-mono font-bold text-navy-900 mt-0.5">
                        {((modelA.Recall || 0) * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
                      <div className="text-[11px] font-mono uppercase text-slate-500">F1 Score</div>
                      <div className="text-lg font-mono font-bold text-navy-900 mt-0.5">
                        {((modelA['F1 Score'] || 0) * 100).toFixed(2)}%
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6">
                    <div className="p-3.5 rounded-xl bg-blue-50/70 border border-blue-200/70">
                      <div className="text-[11px] font-mono uppercase text-blue-700">R² Score</div>
                      <div className="text-lg font-mono font-bold text-navy-900 mt-0.5">
                        {(modelA['R2 Score'] ?? 0).toFixed(4)}
                      </div>
                    </div>
                    <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
                      <div className="text-[11px] font-mono uppercase text-slate-500">RMSE</div>
                      <div className="text-lg font-mono font-bold text-navy-900 mt-0.5">
                        {(modelA.RMSE ?? 0).toFixed(4)}
                      </div>
                    </div>
                    <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
                      <div className="text-[11px] font-mono uppercase text-slate-500">MAE</div>
                      <div className="text-lg font-mono font-bold text-navy-900 mt-0.5">
                        {(modelA.MAE ?? 0).toFixed(4)}
                      </div>
                    </div>
                    <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
                      <div className="text-[11px] font-mono uppercase text-slate-500">MSE</div>
                      <div className="text-lg font-mono font-bold text-navy-900 mt-0.5">
                        {(modelA.MSE ?? 0).toFixed(4)}
                      </div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="py-8 text-center text-sm text-slate-500">
                Approach 1 (Performance-Based Selection) was not included in this run.
              </div>
            )}
          </div>

          <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Selection Basis: Measured Empirical Evaluation</span>
            {modelA && (
              <span className="font-mono text-slate-600">Saved: {modelA['Model Path']}</span>
            )}
          </div>
        </div>

        {/* Model B Card — Selected Through Rule-Based Recommendation */}
        <div className="aurora-card p-6 sm:p-7 border-t-4 border-t-cyan-600 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 mb-4">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-50 text-cyan-800 border border-cyan-200 text-xs font-mono font-semibold">
                <Sparkles className="w-3.5 h-3.5" />
                Model B • Approach 2
              </span>
              <span className="text-xs font-mono text-slate-500">
                Selected through rule-based recommendation
              </span>
            </div>

            {app2 && app2.selected_model_b ? (
              <>
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h2 className="font-display font-extrabold text-2xl sm:text-3xl text-navy-900">
                      {app2.selected_model_b}
                    </h2>
                    <p className="text-xs text-slate-500 mt-1">{app2.rule_engine_type}</p>
                  </div>

                  {/* Distinct Heuristic Recommendation Score Pill (Never presented as Accuracy/F1) */}
                  <div className="p-3 rounded-xl bg-cyan-50 border border-cyan-200 text-right flex-shrink-0">
                    <div className="text-[10px] font-mono uppercase tracking-wider text-cyan-800 font-semibold">
                      Rule Recommendation Score
                    </div>
                    <div className="text-xl font-mono font-extrabold text-cyan-900 mt-0.5">
                      {app2.recommendation_score} pts
                    </div>
                    <div className="text-[10px] text-cyan-700">
                      Heuristic suitability points
                    </div>
                  </div>
                </div>

                {/* Triggered Rules / Reasons */}
                <div className="mt-5 space-y-2">
                  <div className="text-xs font-mono uppercase text-slate-500 font-semibold">
                    Triggered Dataset Rules ({app2.reasons.length}):
                  </div>
                  <div className="max-h-36 overflow-y-auto pr-1 space-y-1.5">
                    {app2.reasons.map((reason, idx) => (
                      <div
                        key={idx}
                        className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/70 text-xs text-slate-700 flex items-start gap-2"
                      >
                        <CheckCircle2 className="w-3.5 h-3.5 text-cyan-600 flex-shrink-0 mt-0.5" />
                        <span>{reason}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="py-8 text-center text-sm text-slate-500">
                Approach 2 (Rule-Based Recommendation) was not included in this run.
              </div>
            )}
          </div>

          <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Selection Basis: Highest-Scoring Rule Engine Output</span>
            <span className="font-mono text-cyan-700 font-semibold">
              Note: Rule score is not a predictive metric
            </span>
          </div>
        </div>
      </div>

      {/* Approach 1 Candidate Model Visualization & Leaderboard Table */}
      {app1 && candidates.length > 0 && (
        <>
          {/* Bar Chart of Candidate Models */}
          <div className="aurora-card p-6 sm:p-8 space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-4">
              <div>
                <h3 className="font-display font-bold text-lg text-navy-900">
                  Approach 1 — Candidate Model Performance Comparison
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Measured evaluation metrics across all {candidates.length} trained candidate models.
                </p>
              </div>
              <span className="text-xs font-mono px-2.5 py-1 rounded bg-blue-50 text-blue-700 font-semibold">
                Best Model A: {modelA?.Model}
              </span>
            </div>

            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  margin={{ top: 10, right: 20, left: 0, bottom: 55 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis
                    dataKey="name"
                    angle={-28}
                    textAnchor="end"
                    interval={0}
                    height={70}
                    tick={{ fontSize: 11, fill: '#334155' }}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: '#475569' }}
                    domain={isClassification ? [0, 100] : ['auto', 'auto']}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0A1128',
                      borderColor: '#1E293B',
                      borderRadius: '12px',
                      color: '#FFFFFF',
                      fontSize: '12px',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '12px' }} />
                  {isClassification ? (
                    <>
                      <Bar dataKey="Accuracy" name="Accuracy (%)" fill="#2563EB" radius={[6, 6, 0, 0]} />
                      <Bar dataKey="F1 Score" name="F1 Score (%)" fill="#06B6D4" radius={[6, 6, 0, 0]} />
                    </>
                  ) : (
                    <>
                      <Bar dataKey="R2 Score" name="R² Score" fill="#2563EB" radius={[6, 6, 0, 0]} />
                      <Bar dataKey="RMSE" name="RMSE" fill="#06B6D4" radius={[6, 6, 0, 0]} />
                    </>
                  )}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Complete Candidate Models Table */}
          <div className="aurora-card overflow-hidden">
            <div className="p-6 border-b border-slate-100 flex items-center justify-between">
              <div>
                <h3 className="font-display font-bold text-lg text-navy-900">
                  Evaluated Candidate Models Leaderboard
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Sorted by {app1.selection_metric}. Model A is highlighted at the top.
                </p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-mono uppercase">
                    <th className="py-3.5 px-4">Rank</th>
                    <th className="py-3.5 px-4">Candidate Model</th>
                    {isClassification ? (
                      <>
                        <th className="py-3.5 px-4">Accuracy</th>
                        <th className="py-3.5 px-4">Precision</th>
                        <th className="py-3.5 px-4">Recall</th>
                        <th className="py-3.5 px-4">F1 Score</th>
                      </>
                    ) : (
                      <>
                        <th className="py-3.5 px-4">R² Score</th>
                        <th className="py-3.5 px-4">RMSE</th>
                        <th className="py-3.5 px-4">MAE</th>
                        <th className="py-3.5 px-4">MSE</th>
                      </>
                    )}
                    <th className="py-3.5 px-4">Role / Tag</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {candidates.map((cand, idx) => {
                    const isModelA = modelA?.Model === cand.Model;
                    const isModelB =
                      app2?.selected_model_b === cand.Model ||
                      (app2?.selected_model_b === 'LDA' &&
                        cand.Model === 'Linear Discriminant Analysis') ||
                      (app2?.selected_model_b === 'QDA' &&
                        cand.Model === 'Quadratic Discriminant Analysis') ||
                      (app2?.selected_model_b === 'MLP' &&
                        cand.Model === 'Multi Layer Perceptron');

                    return (
                      <tr
                        key={cand.Model}
                        className={`hover:bg-slate-50/90 transition-colors ${
                          isModelA ? 'bg-blue-50/50' : ''
                        }`}
                      >
                        <td className="py-3.5 px-4 font-mono font-bold text-slate-500">
                          #{idx + 1}
                        </td>
                        <td className="py-3.5 px-4 font-semibold text-navy-900">{cand.Model}</td>
                        {isClassification ? (
                          <>
                            <td className="py-3.5 px-4 font-mono font-bold text-blue-700">
                              {((cand.Accuracy || 0) * 100).toFixed(2)}%
                            </td>
                            <td className="py-3.5 px-4 font-mono text-slate-700">
                              {((cand.Precision || 0) * 100).toFixed(2)}%
                            </td>
                            <td className="py-3.5 px-4 font-mono text-slate-700">
                              {((cand.Recall || 0) * 100).toFixed(2)}%
                            </td>
                            <td className="py-3.5 px-4 font-mono text-slate-700">
                              {((cand['F1 Score'] || 0) * 100).toFixed(2)}%
                            </td>
                          </>
                        ) : (
                          <>
                            <td className="py-3.5 px-4 font-mono font-bold text-blue-700">
                              {(cand['R2 Score'] ?? 0).toFixed(4)}
                            </td>
                            <td className="py-3.5 px-4 font-mono text-slate-700">
                              {(cand.RMSE ?? 0).toFixed(4)}
                            </td>
                            <td className="py-3.5 px-4 font-mono text-slate-700">
                              {(cand.MAE ?? 0).toFixed(4)}
                            </td>
                            <td className="py-3.5 px-4 font-mono text-slate-700">
                              {(cand.MSE ?? 0).toFixed(4)}
                            </td>
                          </>
                        )}
                        <td className="py-3.5 px-4 space-x-1.5">
                          {isModelA && (
                            <span className="px-2 py-0.5 rounded bg-blue-600 text-white font-mono text-[10px] uppercase font-bold">
                              Model A (Best Performance)
                            </span>
                          )}
                          {isModelB && (
                            <span className="px-2 py-0.5 rounded bg-cyan-600 text-white font-mono text-[10px] uppercase font-bold">
                              Model B (Rule Recommended)
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* 5-Fold Stratified Cross-Validation & SHAP Explainability Section (From run.py Steps 5 & 6) */}
      {modelA && isClassification && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 5-Fold Cross-Validation Card */}
          <div className="aurora-card p-6 space-y-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-5 h-5 text-blue-600" />
                  <div>
                    <h3 className="font-display font-bold text-base text-navy-900">
                      5-Fold Stratified Cross-Validation
                    </h3>
                    <p className="text-xs text-slate-500">
                      Validate {modelA.Model} across 5 stratified folds (<code className="font-mono">perform_cross_validation</code>)
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  disabled={isRunningCV}
                  onClick={handleRunCV}
                  className="px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold transition-all inline-flex items-center gap-1.5"
                >
                  {isRunningCV ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Validating...</span>
                    </>
                  ) : (
                    <span>Run 5-Fold CV</span>
                  )}
                </button>
              </div>

              {latestAnalysis.cross_validation ? (
                <div className="mt-4 space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3.5 rounded-xl bg-blue-50/70 border border-blue-200">
                      <div className="text-[11px] font-mono uppercase text-blue-700">
                        Mean CV Accuracy
                      </div>
                      <div className="text-xl font-mono font-bold text-navy-900 mt-0.5">
                        {(latestAnalysis.cross_validation.mean_accuracy * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                      <div className="text-[11px] font-mono uppercase text-slate-500">
                        Standard Deviation
                      </div>
                      <div className="text-xl font-mono font-bold text-navy-900 mt-0.5">
                        ±{(latestAnalysis.cross_validation.std_accuracy * 100).toFixed(2)}%
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    {latestAnalysis.cross_validation.fold_scores.map((score, i) => (
                      <span
                        key={i}
                        className="px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 font-mono text-xs"
                      >
                        Fold {i + 1}: {(score * 100).toFixed(2)}%
                      </span>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="text-xs text-slate-500 py-6">
                  Click "Run 5-Fold CV" to evaluate fold-by-fold stability for {modelA.Model}.
                </p>
              )}
            </div>
          </div>

          {/* Explainable AI (SHAP) Card */}
          <div className="aurora-card p-6 space-y-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div className="flex items-center gap-2">
                  <Eye className="w-5 h-5 text-blue-600" />
                  <div>
                    <h3 className="font-display font-bold text-base text-navy-900">
                      Explainable AI (SHAP Feature Importance)
                    </h3>
                    <p className="text-xs text-slate-500">
                      Generate SHAP summary plot for {modelA.Model} (<code className="font-mono">generate_shap</code>)
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  disabled={isRunningShap}
                  onClick={handleRunShap}
                  className="px-3.5 py-1.5 rounded-lg bg-navy-900 hover:bg-navy-800 text-white text-xs font-semibold transition-all inline-flex items-center gap-1.5"
                >
                  {isRunningShap ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Generating SHAP...</span>
                    </>
                  ) : (
                    <span>Generate SHAP Plot</span>
                  )}
                </button>
              </div>

              {shapUrl ? (
                <div className="mt-4 rounded-xl border border-slate-200 p-2 bg-white">
                  <img
                    src={shapUrl}
                    alt="SHAP Feature Importance Summary Plot"
                    className="w-full h-auto rounded-lg object-contain max-h-72 mx-auto"
                  />
                </div>
              ) : (
                <p className="text-xs text-slate-500 py-6">
                  Click "Generate SHAP Plot" to compute Shapley Additive Explanations and visualize feature impact.
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {actionError && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-xs text-red-700">
          {actionError}
        </div>
      )}
    </div>
  );
};

