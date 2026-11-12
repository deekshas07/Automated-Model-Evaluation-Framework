export type TaskType = 'classification' | 'regression';
export type AnalysisMode = 'approach_1' | 'approach_2' | 'both';

export interface ClassDistributionItem {
  label: string;
  count: number;
  percentage: number;
}

export interface TargetStatistics {
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
}

export interface TaskDetectionDetails {
  task_type: TaskType;
  target_column: string;
  target_dtype: string;
  unique_values: number;
  unique_ratio: number;
  reason: string;
  class_distribution?: ClassDistributionItem[];
  target_statistics?: TargetStatistics;
}

export interface MissingSummary {
  total_cells: number;
  missing_cells: number;
  missing_percentage: number;
  columns_with_missing: number;
  is_complete: boolean;
}

export interface DtypeSummary {
  numeric: number;
  categorical: number;
  boolean: number;
}

export interface PreprocessingSummaryItem {
  action: string;
  label: string;
  count: number;
}

export interface PreprocessingReportRow {
  Column: string;
  'Recommended Action': string;
  Confidence: number;
  Source: string;
}

export interface FeatureColumnInfo {
  name: string;
  position: number;
  is_target: boolean;
  dtype: string;
  semantic_type: string;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  sample_values: string[];
  recommended_action: string;
  confidence: number;
  source: string;
  explanation?: string;
  stats?: {
    mean?: number;
    std?: number;
    min?: number;
    max?: number;
  };
}

export interface DatasetOverview {
  dataset_id: string;
  dataset_name: string;
  file_size_bytes: number;
  uploaded_at: string;
  preprocessed_path: string;
  report_path: string;
  rows: number;
  columns: number;
  feature_count: number;
  target_column: string;
  task_type: TaskType;
  task_detection: TaskDetectionDetails;
  missing_summary: MissingSummary;
  dtype_summary: DtypeSummary;
  preprocessing_summary: PreprocessingSummaryItem[];
  preprocessing_report: PreprocessingReportRow[];
  features: FeatureColumnInfo[];
  preview_columns: string[];
  preview_rows: Record<string, any>[];
}

export interface SampleDatasetItem {
  filename: string;
  file_size_bytes: number;
  rows: number;
  columns: number;
  target_column: string;
  task_type: TaskType;
}

export interface CandidateModelResult {
  Model: string;
  Accuracy?: number;
  Precision?: number;
  Recall?: number;
  'F1 Score'?: number;
  MAE?: number;
  MSE?: number;
  RMSE?: number;
  'R2 Score'?: number;
  'Model Path': string;
  training_time_sec?: number;
  status?: 'success' | 'failed';
  error?: string;
}

export interface Approach1Output {
  selected_model_a: CandidateModelResult;
  selection_metric: string;
  candidate_count: number;
  candidate_models: CandidateModelResult[];
  all_results: CandidateModelResult[];
}

export interface Approach2Output {
  selected_model_b: string | null;
  recommendation_score: number;
  reasons: string[];
  rules_triggered_count: number;
  message: string;
  trainer_available: boolean;
  is_trained: boolean;
  trained_metrics: CandidateModelResult | null;
  rule_engine_type: string;
}

export interface TunedModelResult {
  Model: string;
  'Best Parameters': Record<string, any>;
  'CV Score': number;
  Accuracy: number;
  Precision: number;
  Recall: number;
  'F1 Score': number;
  'Model Path': string;
  tuning_time_sec: number;
  stage: string;
  search_method: string;
  baseline_metrics?: {
    Accuracy: number;
    Precision: number;
    Recall: number;
    'F1 Score': number;
  } | null;
}

export interface CrossValidationResult {
  model_path: string;
  model_name: string;
  fold_scores: number[];
  mean_accuracy: number;
  std_accuracy: number;
  n_splits: number;
}

export interface AnalysisRecord {
  analysis_id: string;
  completed_at: string;
  dataset_name: string;
  rows: number;
  columns: number;
  feature_count: number;
  target_column: string;
  task_type: TaskType;
  mode: AnalysisMode;
  selection_metric: string;
  approach_1: Approach1Output | null;
  approach_2: Approach2Output | null;
  tuning_results: Record<string, TunedModelResult>;
  cross_validation: CrossValidationResult | null;
  shap_available: boolean;
}

export interface JobStage {
  key: string;
  title: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'error' | 'skipped';
  detail: string;
  started_at?: string;
  completed_at?: string;
}

export interface JobLogEntry {
  timestamp: string;
  message: string;
  level: 'info' | 'success' | 'warning' | 'error';
}

export interface Approach1Progress {
  total_models: number;
  completed_models: number;
  current_model: string | null;
  evaluated: CandidateModelResult[];
}

export interface AnalysisJobStatus {
  job_id: string;
  status: 'running' | 'completed' | 'error';
  mode: AnalysisMode;
  task_type: TaskType;
  created_at: string;
  current_stage: string;
  stages: JobStage[];
  logs: JobLogEntry[];
  approach_1_progress: Approach1Progress | null;
  result: AnalysisRecord | null;
  error: string | null;
}

export interface TuningConfigurationsResponse {
  search_method: string;
  n_iter: number;
  scoring: string;
  cv_strategy: string;
  supported_task_type: string;
  models: Record<string, Record<string, any>>;
}

