import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  CircleDot,
  Clock,
  Cpu,
  Loader2,
  RefreshCw,
  Terminal,
  XCircle,
} from 'lucide-react';
import { useAurora } from '../context/AuroraContext';
import { auroraApi } from '../services/api';

export const ProcessingPage: React.FC = () => {
  const navigate = useNavigate();
  const {
    activeJobId,
    jobStatus,
    setJobStatus,
    setLatestAnalysis,
    refreshState,
  } = useAurora();

  const [pollError, setPollError] = useState<string | null>(null);

  useEffect(() => {
    if (!activeJobId) return;

    let isMounted = true;
    let intervalId: any = null;

    const fetchStatus = async () => {
      try {
        const status = await auroraApi.getJobStatus(activeJobId);
        if (!isMounted) return;
        setJobStatus(status);
        setPollError(null);

        if (status.status === 'completed' && status.result) {
          setLatestAnalysis(status.result);
          refreshState();
          if (intervalId) clearInterval(intervalId);
        } else if (status.status === 'error') {
          if (intervalId) clearInterval(intervalId);
        }
      } catch (err: any) {
        if (!isMounted) return;
        setPollError(
          err?.response?.data?.detail ||
            err?.message ||
            'Lost connection while polling backend job status.'
        );
      }
    };

    fetchStatus();
    intervalId = setInterval(fetchStatus, 1000);

    return () => {
      isMounted = false;
      if (intervalId) clearInterval(intervalId);
    };
  }, [activeJobId]);

  if (!activeJobId && !jobStatus) {
    return (
      <div className="aurora-card p-12 text-center max-w-2xl mx-auto space-y-4">
        <div className="w-14 h-14 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto">
          <Cpu className="w-7 h-7" />
        </div>
        <h2 className="font-display font-bold text-2xl text-navy-900">
          No Active Processing Job
        </h2>
        <p className="text-sm text-slate-600 max-w-md mx-auto">
          Select a model-selection approach on the Model Selection screen to start an analysis run.
        </p>
        <button
          onClick={() => navigate('/select-approach')}
          className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-electric transition-all inline-flex items-center gap-2"
        >
          <span>Go to Model Selection</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    );
  }

  const isRunning = jobStatus?.status === 'running';
  const isCompleted = jobStatus?.status === 'completed';
  const isError = jobStatus?.status === 'error';

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-blue-600 font-semibold">
            <span>Workflow Execution</span>
            <span>•</span>
            <span>Job ID: {activeJobId}</span>
          </div>
          <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-navy-900 tracking-tight">
            AURORA Is Analyzing Your Dataset
          </h1>
          <p className="text-slate-600 text-base">
            Your dataset is being processed through the selected model-selection workflow.
          </p>
        </div>

        {isCompleted && (
          <button
            onClick={() => navigate('/results')}
            className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-electric transition-all inline-flex items-center gap-2"
          >
            <span>View Model Selection Results</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Main Stages & Live Backend Log Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Real Backend Workflow Stages */}
        <div className="lg:col-span-6 aurora-card p-6 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h2 className="font-display font-bold text-base text-navy-900">
              Execution Stages
            </h2>
            <span
              className={`px-2.5 py-0.5 rounded-full text-xs font-mono uppercase font-semibold ${
                isCompleted
                  ? 'bg-emerald-100 text-emerald-800'
                  : isError
                  ? 'bg-red-100 text-red-800'
                  : 'bg-blue-100 text-blue-700'
              }`}
            >
              {jobStatus?.status || 'initializing'}
            </span>
          </div>

          <div className="space-y-4">
            {(jobStatus?.stages || []).map((stage) => {
              const st = stage.status;
              return (
                <div
                  key={stage.key}
                  className={`p-4 rounded-xl border transition-all ${
                    st === 'running'
                      ? 'bg-blue-50/50 border-blue-400 shadow-sm'
                      : st === 'completed'
                      ? 'bg-emerald-50/20 border-emerald-200'
                      : st === 'error'
                      ? 'bg-red-50 border-red-300'
                      : 'bg-slate-50/60 border-slate-200/70 opacity-75'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5 flex-shrink-0">
                      {st === 'running' ? (
                        <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
                      ) : st === 'completed' ? (
                        <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                      ) : st === 'error' ? (
                        <XCircle className="w-5 h-5 text-red-600" />
                      ) : st === 'skipped' ? (
                        <CircleDot className="w-5 h-5 text-slate-400" />
                      ) : (
                        <Clock className="w-5 h-5 text-slate-400" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <h3 className="font-display font-bold text-sm text-navy-900">
                          {stage.title}
                        </h3>
                        <span className="text-[10px] font-mono uppercase text-slate-500">
                          {st}
                        </span>
                      </div>
                      <p className="text-xs text-slate-600 mt-0.5">{stage.description}</p>
                      {stage.detail && (
                        <div className="mt-2 px-2.5 py-1 rounded bg-white/90 border border-slate-200/80 text-xs font-mono text-navy-900">
                          {stage.detail}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Candidate Model Counter from Backend (When Approach 1 is Active) */}
          {jobStatus?.approach_1_progress && (
            <div className="p-4 rounded-xl bg-slate-900 text-white space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-blue-300">Approach 1 Candidate Registry</span>
                <span>
                  {jobStatus.approach_1_progress.completed_models} /{' '}
                  {jobStatus.approach_1_progress.total_models} Models Evaluated
                </span>
              </div>
              {jobStatus.approach_1_progress.current_model && (
                <div className="text-xs text-slate-300 font-mono">
                  Currently training:{' '}
                  <strong className="text-white">
                    {jobStatus.approach_1_progress.current_model}
                  </strong>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: Real Backend Event Logs & Evaluated Models Stream */}
        <div className="lg:col-span-6 aurora-card p-6 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between border-b border-slate-100 pb-3 mb-4">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-blue-600" />
                <h2 className="font-display font-bold text-base text-navy-900">
                  Backend Execution Log Stream
                </h2>
              </div>
              <span className="text-xs font-mono text-slate-400">
                {jobStatus?.logs?.length || 0} events
              </span>
            </div>

            <div className="bg-navy-950 text-slate-200 rounded-xl p-4 font-mono text-xs h-80 overflow-y-auto space-y-2 border border-slate-800">
              {(jobStatus?.logs || []).length === 0 ? (
                <div className="text-slate-500 italic">Waiting for backend events...</div>
              ) : (
                jobStatus!.logs.map((log, i) => (
                  <div key={i} className="flex items-start gap-2 leading-relaxed">
                    <span className="text-slate-500 flex-shrink-0">[{log.timestamp}]</span>
                    <span
                      className={
                        log.level === 'success'
                          ? 'text-emerald-400 font-semibold'
                          : log.level === 'error'
                          ? 'text-red-400 font-semibold'
                          : log.level === 'warning'
                          ? 'text-amber-300'
                          : 'text-slate-200'
                      }
                    >
                      {log.message}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Error Recovery State */}
          {(isError || pollError) && (
            <div className="p-4 rounded-xl bg-red-50 border border-red-200 space-y-3">
              <div className="flex items-start gap-2.5 text-red-800">
                <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                <div className="text-xs">
                  <p className="font-bold">Workflow Execution Error</p>
                  <p className="mt-0.5 text-red-700">{jobStatus?.error || pollError}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => navigate('/select-approach')}
                  className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white text-xs font-semibold inline-flex items-center gap-1.5"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Return & Retry</span>
                </button>
              </div>
            </div>
          )}

          {/* Success Callout */}
          {isCompleted && (
            <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 flex items-center justify-between gap-4">
              <div className="flex items-center gap-2.5 text-emerald-900 text-xs sm:text-sm font-medium">
                <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0" />
                <span>Analysis completed. Results are ready for inspection.</span>
              </div>
              <button
                onClick={() => navigate('/results')}
                className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold whitespace-nowrap inline-flex items-center gap-1.5"
              >
                <span>Open Results</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

