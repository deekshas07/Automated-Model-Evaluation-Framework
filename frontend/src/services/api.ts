import axios from 'axios';
import {
  AnalysisJobStatus,
  AnalysisMode,
  AnalysisRecord,
  CrossValidationResult,
  DatasetOverview,
  SampleDatasetItem,
  TaskType,
  TunedModelResult,
  TuningConfigurationsResponse,
} from '../types/aurora';

const apiClient = axios.create({
  baseURL: '/api',
  timeout: 600000, // 10 minutes for intensive tuning on large multi-class datasets
});

export const auroraApi = {
  async checkHealth() {
    const { data } = await apiClient.get('/health');
    return data;
  },

  async getSampleDatasets(): Promise<SampleDatasetItem[]> {
    const { data } = await apiClient.get<{ samples: SampleDatasetItem[] }>('/datasets/samples');
    return data.samples;
  },

  async uploadDataset(file: File): Promise<DatasetOverview> {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await apiClient.post<DatasetOverview>('/datasets/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return data;
  },

  async loadSampleDataset(filename: string): Promise<DatasetOverview> {
    const { data } = await apiClient.post<DatasetOverview>('/datasets/load-sample', { filename });
    return data;
  },

  async getCurrentDataset(): Promise<DatasetOverview> {
    const { data } = await apiClient.get<DatasetOverview>('/datasets/current');
    return data;
  },

  async updateTaskType(task_type: TaskType): Promise<DatasetOverview> {
    const { data } = await apiClient.patch<DatasetOverview>('/datasets/task-type', { task_type });
    return data;
  },

  async startAnalysis(params: {
    mode: AnalysisMode;
    task_type?: TaskType;
    selection_metric?: string;
    train_model_b?: boolean;
  }): Promise<{ job_id: string; status: string; mode: AnalysisMode; task_type: TaskType }> {
    const { data } = await apiClient.post('/analysis/start', params);
    return data;
  },

  async getJobStatus(jobId: string): Promise<AnalysisJobStatus> {
    const { data } = await apiClient.get<AnalysisJobStatus>(`/analysis/status/${jobId}`);
    return data;
  },

  async getLatestAnalysis(): Promise<AnalysisRecord> {
    const { data } = await apiClient.get<AnalysisRecord>('/analysis/latest');
    return data;
  },

  async getAnalysisHistory(): Promise<AnalysisRecord[]> {
    const { data } = await apiClient.get<{ history: AnalysisRecord[] }>('/analysis/history');
    return data.history;
  },

  async trainRecommendedModelB(): Promise<AnalysisRecord> {
    const { data } = await apiClient.post<AnalysisRecord>('/analysis/train-model-b');
    return data;
  },

  async getTuningConfigurations(): Promise<TuningConfigurationsResponse> {
    const { data } = await apiClient.get<TuningConfigurationsResponse>('/tuning/configurations');
    return data;
  },

  async optimizeHyperparameters(model_name: string, stage: string = 'approach_1'): Promise<TunedModelResult> {
    const { data } = await apiClient.post<TunedModelResult>('/tuning/optimize', {
      model_name,
      stage,
    });
    return data;
  },

  async runCrossValidation(model_name?: string, model_path?: string): Promise<CrossValidationResult> {
    const { data } = await apiClient.post<CrossValidationResult>('/evaluation/cross-validate', {
      model_name,
      model_path,
    });
    return data;
  },

  async generateShapPlot(model_name?: string, model_path?: string): Promise<{ status: string; shap_image_url: string }> {
    const { data } = await apiClient.post('/evaluation/shap', {
      model_name,
      model_path,
    });
    return data;
  },
};

