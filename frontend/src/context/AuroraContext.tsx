import React, { createContext, useContext, useEffect, useState } from 'react';
import {
  AnalysisJobStatus,
  AnalysisMode,
  AnalysisRecord,
  DatasetOverview,
  TaskType,
} from '../types/aurora';
import { auroraApi } from '../services/api';

interface AuroraContextValue {
  dataset: DatasetOverview | null;
  setDataset: React.Dispatch<React.SetStateAction<DatasetOverview | null>>;
  selectedApproach: AnalysisMode;
  setSelectedApproach: React.Dispatch<React.SetStateAction<AnalysisMode>>;
  selectionMetric: string;
  setSelectionMetric: React.Dispatch<React.SetStateAction<string>>;
  trainModelBOnRun: boolean;
  setTrainModelBOnRun: React.Dispatch<React.SetStateAction<boolean>>;
  activeJobId: string | null;
  setActiveJobId: React.Dispatch<React.SetStateAction<string | null>>;
  jobStatus: AnalysisJobStatus | null;
  setJobStatus: React.Dispatch<React.SetStateAction<AnalysisJobStatus | null>>;
  latestAnalysis: AnalysisRecord | null;
  setLatestAnalysis: React.Dispatch<React.SetStateAction<AnalysisRecord | null>>;
  history: AnalysisRecord[];
  refreshState: () => Promise<void>;
  updateDatasetTaskType: (taskType: TaskType) => Promise<void>;
  isInitialLoading: boolean;
}

const AuroraContext = createContext<AuroraContextValue | undefined>(undefined);

export const AuroraProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [dataset, setDataset] = useState<DatasetOverview | null>(null);
  const [selectedApproach, setSelectedApproach] = useState<AnalysisMode>('both');
  const [selectionMetric, setSelectionMetric] = useState<string>('Accuracy');
  const [trainModelBOnRun, setTrainModelBOnRun] = useState<boolean>(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<AnalysisJobStatus | null>(null);
  const [latestAnalysis, setLatestAnalysis] = useState<AnalysisRecord | null>(null);
  const [history, setHistory] = useState<AnalysisRecord[]>([]);
  const [isInitialLoading, setIsInitialLoading] = useState<boolean>(true);

  const refreshState = async () => {
    try {
      const [dsRes, histRes] = await Promise.allSettled([
        auroraApi.getCurrentDataset(),
        auroraApi.getAnalysisHistory(),
      ]);

      if (dsRes.status === 'fulfilled' && dsRes.value) {
        setDataset(dsRes.value);
        setSelectionMetric(dsRes.value.task_type === 'regression' ? 'R2 Score' : 'Accuracy');
      }

      if (histRes.status === 'fulfilled' && histRes.value) {
        setHistory(histRes.value);
        if (histRes.value.length > 0) {
          setLatestAnalysis(histRes.value[0]);
        }
      }
    } catch {
      // Ignore initial empty state errors
    } finally {
      setIsInitialLoading(false);
    }
  };

  useEffect(() => {
    refreshState();
  }, []);

  const updateDatasetTaskType = async (taskType: TaskType) => {
    const updated = await auroraApi.updateTaskType(taskType);
    setDataset(updated);
    setSelectionMetric(taskType === 'regression' ? 'R2 Score' : 'Accuracy');
  };

  return (
    <AuroraContext.Provider
      value={{
        dataset,
        setDataset,
        selectedApproach,
        setSelectedApproach,
        selectionMetric,
        setSelectionMetric,
        trainModelBOnRun,
        setTrainModelBOnRun,
        activeJobId,
        setActiveJobId,
        jobStatus,
        setJobStatus,
        latestAnalysis,
        setLatestAnalysis,
        history,
        refreshState,
        updateDatasetTaskType,
        isInitialLoading,
      }}
    >
      {children}
    </AuroraContext.Provider>
  );
};

export const useAurora = () => {
  const ctx = useContext(AuroraContext);
  if (!ctx) {
    throw new Error('useAurora must be used within an AuroraProvider');
  }
  return ctx;
};

