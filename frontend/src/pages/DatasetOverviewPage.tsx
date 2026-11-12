import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Database,
  Download,
  FileSpreadsheet,
  Hash,
  Layers,
  Sparkles,
  Table,
  Target,
  UploadCloud,
} from 'lucide-react';
import { useAurora } from '../context/AuroraContext';
import { TaskType } from '../types/aurora';

export const DatasetOverviewPage: React.FC = () => {
  const navigate = useNavigate();
  const { dataset, updateDatasetTaskType } = useAurora();
  const [activeTab, setActiveTab] = useState<'features' | 'preview' | 'preprocessing'>('features');
  const [updatingTask, setUpdatingTask] = useState<boolean>(false);

  if (!dataset) {
    return (
      <div className="aurora-card p-12 text-center max-w-2xl mx-auto space-y-4">
        <div className="w-14 h-14 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto">
          <FileSpreadsheet className="w-7 h-7" />
        </div>
        <h2 className="font-display font-bold text-2xl text-navy-900">
          No Dataset Loaded
        </h2>
        <p className="text-sm text-slate-600 max-w-md mx-auto">
          Please upload a CSV dataset first to inspect its rows, columns, target variable, data types, and preprocessing recommendations.
        </p>
        <button
          onClick={() => navigate('/upload')}
          className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-electric transition-all inline-flex items-center gap-2"
        >
          <UploadCloud className="w-4 h-4" />
          <span>Go to Dataset Upload</span>
        </button>
      </div>
    );
  }

  const handleTaskSwitch = async (newTask: TaskType) => {
    if (newTask === dataset.task_type || updatingTask) return;
    setUpdatingTask(true);
    try {
      await updateDatasetTaskType(newTask);
    } finally {
      setUpdatingTask(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-blue-600 font-semibold">
            <span>Step 2 of 4</span>
            <span>•</span>
            <span>Dataset Profiling & Symbolic Preprocessing</span>
          </div>
          <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-navy-900 tracking-tight">
            Understand Your Dataset
          </h1>
          <p className="text-slate-600 text-base">
            Review the dataset characteristics before proceeding with model selection.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <a
            href="/api/reports/preprocessing-csv"
            download
            className="px-4 py-2.5 rounded-xl bg-white hover:bg-slate-50 text-slate-700 font-semibold text-xs sm:text-sm border border-slate-200 shadow-sm transition-all inline-flex items-center gap-2"
          >
            <Download className="w-4 h-4 text-blue-600" />
            <span>Download Preprocessing Report</span>
          </a>

          <button
            onClick={() => navigate('/select-approach')}
            className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-electric transition-all inline-flex items-center gap-2"
          >
            <span>Proceed to Model Selection</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Top Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* Dataset Name & Dimensions */}
        <div className="aurora-card p-5">
          <div className="flex items-center justify-between text-xs font-mono uppercase text-slate-500 mb-2">
            <span>Dataset Name</span>
            <Database className="w-4 h-4 text-blue-600" />
          </div>
          <div className="font-display font-bold text-lg text-navy-900 truncate" title={dataset.dataset_name}>
            {dataset.dataset_name}
          </div>
          <div className="mt-2 flex items-center gap-2 text-xs font-mono text-slate-600">
            <span className="px-2 py-0.5 rounded bg-slate-100">
              {dataset.rows.toLocaleString()} Rows
            </span>
            <span>×</span>
            <span className="px-2 py-0.5 rounded bg-slate-100">
              {dataset.columns} Columns
            </span>
          </div>
        </div>

        {/* Target Column */}
        <div className="aurora-card p-5">
          <div className="flex items-center justify-between text-xs font-mono uppercase text-slate-500 mb-2">
            <span>Target Variable (Last Col)</span>
            <Target className="w-4 h-4 text-blue-600" />
          </div>
          <div className="font-display font-bold text-lg text-navy-900 truncate">
            {dataset.target_column}
          </div>
          <div className="mt-2 text-xs text-slate-500 font-mono">
            Dtype: <strong className="text-slate-700">{dataset.task_detection.target_dtype}</strong> •{' '}
            {dataset.task_detection.unique_values} unique values
          </div>
        </div>

        {/* Detected Task Type */}
        <div className="aurora-card p-5">
          <div className="flex items-center justify-between text-xs font-mono uppercase text-slate-500 mb-2">
            <span>Detected Problem Type</span>
            <Layers className="w-4 h-4 text-blue-600" />
          </div>
          <div className="flex items-center justify-between gap-2">
            <span className="font-display font-bold text-lg text-navy-900 capitalize">
              {dataset.task_type}
            </span>
            <div className="inline-flex rounded-lg bg-slate-100 p-0.5 border border-slate-200 text-[11px] font-mono">
              <button
                type="button"
                onClick={() => handleTaskSwitch('classification')}
                className={`px-2 py-0.5 rounded-md transition-colors ${
                  dataset.task_type === 'classification'
                    ? 'bg-blue-600 text-white font-semibold'
                    : 'text-slate-600 hover:text-navy-900'
                }`}
              >
                Classif.
              </button>
              <button
                type="button"
                onClick={() => handleTaskSwitch('regression')}
                className={`px-2 py-0.5 rounded-md transition-colors ${
                  dataset.task_type === 'regression'
                    ? 'bg-blue-600 text-white font-semibold'
                    : 'text-slate-600 hover:text-navy-900'
                }`}
              >
                Regress.
              </button>
            </div>
          </div>
          <p className="mt-2 text-[11px] text-slate-500 line-clamp-1" title={dataset.task_detection.reason}>
            {dataset.task_detection.reason}
          </p>
        </div>

        {/* Missing Values & Data Types */}
        <div className="aurora-card p-5">
          <div className="flex items-center justify-between text-xs font-mono uppercase text-slate-500 mb-2">
            <span>Missing Values & Types</span>
            <Hash className="w-4 h-4 text-blue-600" />
          </div>
          <div className="flex items-center gap-2">
            <span className="font-display font-bold text-lg text-navy-900">
              {dataset.missing_summary.missing_cells.toLocaleString()}
            </span>
            <span className="text-xs font-mono text-slate-500">
              ({dataset.missing_summary.missing_percentage}% of cells)
            </span>
            {dataset.missing_summary.is_complete ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-500 ml-auto" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-amber-500 ml-auto" />
            )}
          </div>
          <div className="mt-2 flex items-center gap-2 text-xs font-mono text-slate-600">
            <span>Num: {dataset.dtype_summary.numeric}</span>
            <span>•</span>
            <span>Cat: {dataset.dtype_summary.categorical}</span>
            {dataset.dtype_summary.boolean > 0 && (
              <>
                <span>•</span>
                <span>Bool: {dataset.dtype_summary.boolean}</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Target Distribution & Symbolic Preprocessing Summary Row */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Target Variable Analysis */}
        <div className="lg:col-span-5 aurora-card p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <h3 className="font-display font-bold text-base text-navy-900">
                Target Variable Profile: <code className="text-blue-600 font-mono">{dataset.target_column}</code>
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                {dataset.task_detection.reason}
              </p>
            </div>
          </div>

          {dataset.task_detection.class_distribution ? (
            <div className="space-y-2.5 max-h-56 overflow-y-auto pr-1">
              {dataset.task_detection.class_distribution.map((cls) => (
                <div key={cls.label} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-mono font-semibold text-slate-800">
                      Class: {cls.label}
                    </span>
                    <span className="font-mono text-slate-500">
                      {cls.count} rows ({cls.percentage}%)
                    </span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-blue-600 to-cyan-500"
                      style={{ width: `${Math.min(100, Math.max(4, cls.percentage))}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : dataset.task_detection.target_statistics ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-1">
              {Object.entries(dataset.task_detection.target_statistics).map(([k, v]) => (
                <div key={k} className="p-3 rounded-xl bg-slate-50 border border-slate-200/70">
                  <div className="text-[11px] font-mono uppercase text-slate-500">{k}</div>
                  <div className="text-sm font-mono font-bold text-navy-900 mt-0.5">{v}</div>
                </div>
              ))}
            </div>
          ) : null}
        </div>

        {/* Right: AURORA Symbolic Preprocessing Summary */}
        <div className="lg:col-span-7 aurora-card p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-blue-600" />
              <div>
                <h3 className="font-display font-bold text-base text-navy-900">
                  AURORA Symbolic Preprocessing Recommendations
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Generated by <code className="font-mono">get_preprocessor().preprocess_column()</code> across {dataset.feature_count} input features
                </p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {dataset.preprocessing_summary.map((item) => (
              <div
                key={item.action}
                className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 flex items-center justify-between"
              >
                <div>
                  <div className="text-xs font-semibold text-navy-900">{item.label}</div>
                  <div className="text-[11px] font-mono text-slate-500">{item.action}</div>
                </div>
                <span className="w-8 h-8 rounded-lg bg-blue-600/10 text-blue-700 font-mono font-bold text-xs flex items-center justify-center">
                  {item.count}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Tabbed Inspection Area: Feature Information, Dataset Preview, Preprocessing Report */}
      <div className="aurora-card overflow-hidden">
        <div className="px-6 pt-4 border-b border-slate-200 flex flex-wrap items-center justify-between gap-4 bg-slate-50/60">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setActiveTab('features')}
              className={`px-4 py-2.5 text-xs sm:text-sm font-semibold border-b-2 transition-colors flex items-center gap-2 ${
                activeTab === 'features'
                  ? 'border-blue-600 text-blue-600 bg-white rounded-t-lg'
                  : 'border-transparent text-slate-600 hover:text-navy-900'
              }`}
            >
              <Layers className="w-4 h-4" />
              <span>Feature & Column Information ({dataset.columns})</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('preview')}
              className={`px-4 py-2.5 text-xs sm:text-sm font-semibold border-b-2 transition-colors flex items-center gap-2 ${
                activeTab === 'preview'
                  ? 'border-blue-600 text-blue-600 bg-white rounded-t-lg'
                  : 'border-transparent text-slate-600 hover:text-navy-900'
              }`}
            >
              <Table className="w-4 h-4" />
              <span>Dataset Preview (First {dataset.preview_rows.length} Rows)</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('preprocessing')}
              className={`px-4 py-2.5 text-xs sm:text-sm font-semibold border-b-2 transition-colors flex items-center gap-2 ${
                activeTab === 'preprocessing'
                  ? 'border-blue-600 text-blue-600 bg-white rounded-t-lg'
                  : 'border-transparent text-slate-600 hover:text-navy-900'
              }`}
            >
              <Sparkles className="w-4 h-4" />
              <span>Preprocessing Report ({dataset.preprocessing_report.length})</span>
            </button>
          </div>
        </div>

        {/* Tab 1: Feature Information Table */}
        {activeTab === 'features' && (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-mono uppercase">
                  <th className="py-3 px-4">#</th>
                  <th className="py-3 px-4">Column Name</th>
                  <th className="py-3 px-4">Role</th>
                  <th className="py-3 px-4">Data Type</th>
                  <th className="py-3 px-4">Semantic Type</th>
                  <th className="py-3 px-4">Missing</th>
                  <th className="py-3 px-4">Unique</th>
                  <th className="py-3 px-4">AURORA Action</th>
                  <th className="py-3 px-4">Sample Values</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {dataset.features.map((feat) => (
                  <tr
                    key={feat.name}
                    className={`hover:bg-slate-50/80 transition-colors ${
                      feat.is_target ? 'bg-blue-50/40' : ''
                    }`}
                  >
                    <td className="py-3 px-4 font-mono text-slate-400">{feat.position}</td>
                    <td className="py-3 px-4 font-semibold text-navy-900">{feat.name}</td>
                    <td className="py-3 px-4">
                      {feat.is_target ? (
                        <span className="px-2 py-0.5 rounded bg-blue-600 text-white font-mono text-[10px] uppercase font-bold">
                          Target
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-mono text-[10px] uppercase">
                          Feature
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-600">{feat.dtype}</td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-mono text-[11px]">
                        {feat.semantic_type}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono">
                      {feat.missing_count > 0 ? (
                        <span className="text-amber-700 font-semibold">
                          {feat.missing_count} ({feat.missing_percentage}%)
                        </span>
                      ) : (
                        <span className="text-emerald-700">0 (0%)</span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-700">{feat.unique_count}</td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200/60 font-mono text-[11px]">
                        {feat.recommended_action}
                      </span>
                      {!feat.is_target && (
                        <span className="ml-1.5 font-mono text-[10px] text-slate-400">
                          ({Math.round(feat.confidence * 100)}%)
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-500 truncate max-w-[220px]">
                      {feat.sample_values.join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Tab 2: Dataset Preview */}
        {activeTab === 'preview' && (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-mono">
                  <th className="py-3 px-3 bg-slate-100/80 text-slate-400">Row</th>
                  {dataset.preview_columns.map((col, idx) => {
                    const isTarget = idx === dataset.preview_columns.length - 1;
                    return (
                      <th
                        key={col}
                        className={`py-3 px-3 whitespace-nowrap ${
                          isTarget ? 'bg-blue-50 text-blue-800 font-bold' : ''
                        }`}
                      >
                        {col} {isTarget && '(Target)'}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {dataset.preview_rows.map((row, rIdx) => (
                  <tr key={rIdx} className="hover:bg-slate-50/80">
                    <td className="py-2.5 px-3 text-slate-400 bg-slate-50/40">{rIdx + 1}</td>
                    {dataset.preview_columns.map((col, cIdx) => {
                      const val = row[col];
                      const isTarget = cIdx === dataset.preview_columns.length - 1;
                      return (
                        <td
                          key={col}
                          className={`py-2.5 px-3 whitespace-nowrap ${
                            isTarget ? 'bg-blue-50/20 font-semibold text-navy-900' : 'text-slate-700'
                          }`}
                        >
                          {val === null || val === undefined ? (
                            <span className="text-amber-500 italic">null</span>
                          ) : (
                            String(val)
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Tab 3: Preprocessing Report */}
        {activeTab === 'preprocessing' && (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-mono uppercase">
                  <th className="py-3 px-4">Column</th>
                  <th className="py-3 px-4">Recommended Action</th>
                  <th className="py-3 px-4">Confidence</th>
                  <th className="py-3 px-4">Decision Source</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {dataset.preprocessing_report.map((row) => (
                  <tr key={row.Column} className="hover:bg-slate-50/80">
                    <td className="py-3 px-4 font-semibold text-navy-900">{row.Column}</td>
                    <td className="py-3 px-4">
                      <span className="px-2.5 py-1 rounded-md bg-blue-50 text-blue-700 font-mono font-medium">
                        {row['Recommended Action']}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-700">
                      {(row.Confidence * 100).toFixed(0)}%
                    </td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-mono uppercase text-[10px]">
                        {row.Source}
                      </span>
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

