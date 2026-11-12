import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Database,
  FileSpreadsheet,
  FileText,
  Loader2,
  UploadCloud,
  X,
} from 'lucide-react';
import { auroraApi } from '../services/api';
import { useAurora } from '../context/AuroraContext';
import { SampleDatasetItem } from '../types/aurora';

export const DatasetUploadPage: React.FC = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { setDataset, setSelectionMetric } = useAurora();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleDatasetItem[]>([]);
  const [loadingSampleName, setLoadingSampleName] = useState<string | null>(null);

  useEffect(() => {
    auroraApi
      .getSampleDatasets()
      .then((list) => setSamples(list))
      .catch(() => {});
  }, []);

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };

  const validateAndSelectFile = (file: File) => {
    setErrorMsg(null);
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setErrorMsg('Invalid file format. AURORA requires a valid .csv tabular dataset file.');
      setSelectedFile(null);
      return;
    }
    if (file.size === 0) {
      setErrorMsg('The selected CSV file is empty (0 bytes). Please choose a valid dataset.');
      setSelectedFile(null);
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      setErrorMsg('File exceeds the 100 MB maximum limit defined by RobustCSVParser.');
      setSelectedFile(null);
      return;
    }
    setSelectedFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      validateAndSelectFile(files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSelectFile(e.target.files[0]);
    }
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile) {
      setErrorMsg('Please select a CSV dataset file before continuing.');
      return;
    }
    setIsUploading(true);
    setErrorMsg(null);
    try {
      const overview = await auroraApi.uploadDataset(selectedFile);
      setDataset(overview);
      setSelectionMetric(overview.task_type === 'regression' ? 'R2 Score' : 'Accuracy');
      navigate('/overview');
    } catch (err: any) {
      const detail =
        err?.response?.data?.detail ||
        err?.message ||
        'Failed to upload and analyze dataset. Please verify the CSV format.';
      setErrorMsg(detail);
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadSample = async (filename: string) => {
    setLoadingSampleName(filename);
    setErrorMsg(null);
    try {
      const overview = await auroraApi.loadSampleDataset(filename);
      setDataset(overview);
      setSelectionMetric(overview.task_type === 'regression' ? 'R2 Score' : 'Accuracy');
      navigate('/overview');
    } catch (err: any) {
      const detail =
        err?.response?.data?.detail ||
        err?.message ||
        `Failed to load sample dataset ${filename}.`;
      setErrorMsg(detail);
    } finally {
      setLoadingSampleName(null);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Page Header */}
      <div className="space-y-2">
        <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-blue-600 font-semibold">
          <span>Step 1 of 4</span>
          <span>•</span>
          <span>Dataset Input</span>
        </div>
        <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-navy-900 tracking-tight">
          Start Your Analysis
        </h1>
        <p className="text-slate-600 text-base">
          Upload a preprocessed dataset to begin model selection and evaluation.
        </p>
      </div>

      {/* Main Upload Card */}
      <div className="aurora-card p-6 sm:p-8 space-y-6">
        {/* Drag-and-Drop Zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition-all ${
            isDragging
              ? 'border-blue-600 bg-blue-50/70'
              : selectedFile
              ? 'border-emerald-500/60 bg-emerald-50/20'
              : 'border-slate-300 hover:border-blue-500 bg-slate-50/60 hover:bg-slate-50'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            onChange={handleFileChange}
            className="hidden"
          />

          <div className="w-16 h-16 rounded-2xl bg-blue-600/10 text-blue-600 flex items-center justify-center mx-auto mb-4">
            <UploadCloud className="w-8 h-8" />
          </div>

          <h3 className="font-display font-bold text-lg text-navy-900">
            Drag and drop your CSV dataset here
          </h3>
          <p className="text-sm text-slate-500 mt-1 max-w-md mx-auto">
            or click the button below to browse files from your computer
          </p>

          <div className="mt-5">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
              className="px-5 py-2.5 rounded-xl bg-white hover:bg-slate-100 text-navy-900 font-semibold text-sm border border-slate-300 shadow-sm transition-all"
            >
              Browse Files
            </button>
          </div>

          <div className="mt-6 pt-5 border-t border-slate-200/70 flex flex-wrap items-center justify-center gap-6 text-xs text-slate-500 font-mono">
            <span>Supported Format: .CSV (Tabular)</span>
            <span>•</span>
            <span>Target Variable: Last Column</span>
            <span>•</span>
            <span>Max Size: 100 MB</span>
          </div>
        </div>

        {/* Selected File Preview Bar */}
        {selectedFile && (
          <div className="p-4 rounded-xl bg-blue-50/70 border border-blue-200 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded-lg bg-blue-600 text-white flex items-center justify-center flex-shrink-0">
                <FileText className="w-5 h-5" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <p className="font-semibold text-sm text-navy-900 truncate">
                    {selectedFile.name}
                  </p>
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-100 text-emerald-800">
                    <CheckCircle2 className="w-3 h-3" />
                    Validated CSV
                  </span>
                </div>
                <p className="text-xs text-slate-500 font-mono mt-0.5">
                  Size: {formatBytes(selectedFile.size)}
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setSelectedFile(null)}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200/60 transition-colors"
              title="Remove selected file"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Error Alert */}
        {errorMsg && (
          <div className="p-4 rounded-xl bg-red-50 border border-red-200 flex items-start gap-3 text-red-800">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div className="text-xs sm:text-sm">
              <p className="font-semibold">Upload Validation Error</p>
              <p className="mt-0.5 text-red-700">{errorMsg}</p>
            </div>
          </div>
        )}

        {/* Primary Action Bar */}
        <div className="flex items-center justify-between pt-2">
          <div className="text-xs text-slate-500">
            AURORA will analyze feature columns, generate a preprocessing report, and prepare the dataset.
          </div>
          <button
            type="button"
            disabled={!selectedFile || isUploading}
            onClick={handleUploadSubmit}
            className={`px-6 py-3 rounded-xl font-semibold text-sm flex items-center gap-2 transition-all ${
              !selectedFile || isUploading
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-500 text-white shadow-electric'
            }`}
          >
            {isUploading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Analyzing & Uploading...</span>
              </>
            ) : (
              <>
                <span>Upload & Inspect Dataset</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Built-in Benchmark Datasets from Backend datasets/ Folder */}
      {samples.length > 0 && (
        <div className="aurora-card p-6 sm:p-8 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <Database className="w-5 h-5 text-blue-600" />
              <div>
                <h2 className="font-display font-bold text-lg text-navy-900">
                  Or Select a Workspace Benchmark Dataset
                </h2>
                <p className="text-xs text-slate-500">
                  Available CSV datasets detected in the backend <code className="font-mono">datasets/</code> directory for rapid testing.
                </p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-2">
            {samples.map((s) => {
              const isBusy = loadingSampleName === s.filename;
              return (
                <button
                  key={s.filename}
                  type="button"
                  disabled={!!loadingSampleName || isUploading}
                  onClick={() => handleLoadSample(s.filename)}
                  className="text-left p-3.5 rounded-xl border border-slate-200 hover:border-blue-500/50 bg-slate-50/70 hover:bg-white transition-all flex flex-col justify-between group"
                >
                  <div>
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold text-xs text-navy-900 truncate group-hover:text-blue-600">
                        {s.filename}
                      </span>
                      <span
                        className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded font-semibold ${
                          s.task_type === 'classification'
                            ? 'bg-blue-100 text-blue-700'
                            : 'bg-purple-100 text-purple-700'
                        }`}
                      >
                        {s.task_type}
                      </span>
                    </div>
                    <div className="mt-2 flex items-center gap-2 text-[11px] font-mono text-slate-500">
                      <span>{s.rows} rows</span>
                      <span>•</span>
                      <span>{s.columns} cols</span>
                      <span>•</span>
                      <span>{formatBytes(s.file_size_bytes)}</span>
                    </div>
                  </div>
                  <div className="mt-2.5 pt-2 border-t border-slate-200/60 flex items-center justify-between text-[11px] text-slate-500">
                    <span className="truncate">Target: {s.target_column}</span>
                    {isBusy ? (
                      <Loader2 className="w-3.5 h-3.5 text-blue-600 animate-spin flex-shrink-0" />
                    ) : (
                      <ArrowRight className="w-3.5 h-3.5 text-slate-400 group-hover:text-blue-600 flex-shrink-0" />
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

