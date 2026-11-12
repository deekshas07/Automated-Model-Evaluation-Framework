import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Activity,
  ArrowRight,
  BarChart3,
  BookOpen,
  CheckCircle2,
  Cpu,
  Database,
  FileSpreadsheet,
  GitCompare,
  Layers,
  ShieldCheck,
  Sliders,
  Sparkles,
  UploadCloud,
  Zap,
} from 'lucide-react';

export const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-white text-slate-900">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-50 bg-navy-950/95 backdrop-blur-md border-b border-slate-800/80 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-electric">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-display font-extrabold text-xl tracking-tight text-white">
                  AURORA
                </span>
                <span className="text-[10px] font-mono uppercase tracking-widest px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-400/30">
                  Intelligent AutoML
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Intelligent Model Recommendation System
              </p>
            </div>
          </Link>

          <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-300">
            <button
              onClick={() => scrollToSection('features')}
              className="hover:text-white transition-colors"
            >
              Product
            </button>
            <button
              onClick={() => scrollToSection('how-it-works')}
              className="hover:text-white transition-colors"
            >
              How It Works
            </button>
            <button
              onClick={() => scrollToSection('architecture-preview')}
              className="hover:text-white transition-colors"
            >
              Model Selection
            </button>
            <Link to="/dashboard" className="hover:text-white transition-colors">
              Dashboard
            </Link>
          </nav>

          <div className="flex items-center gap-3">
            <Link
              to="/dashboard"
              className="hidden sm:inline-flex px-4 py-2 rounded-xl text-sm font-semibold text-slate-200 hover:text-white hover:bg-slate-800/80 border border-slate-700/80 transition-all"
            >
              Open Dashboard
            </Link>
            <button
              onClick={() => navigate('/upload')}
              className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold shadow-electric transition-all flex items-center gap-2"
            >
              <span>Start New Analysis</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative bg-navy-hero text-white overflow-hidden border-b border-slate-800">
        <div className="absolute inset-0 bg-grid-navy pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-28 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-10 items-center">
            {/* Left Column: Hero Copy & Actions */}
            <div className="lg:col-span-6 space-y-8">
              <div className="inline-flex items-center gap-2.5 px-3.5 py-1.5 rounded-full bg-blue-500/10 border border-blue-400/30 text-blue-300 text-xs font-medium">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                <span>Hybrid Symbolic & Empirical AutoML Framework</span>
              </div>

              <h1 className="font-display font-extrabold text-4xl sm:text-5xl lg:text-[54px] leading-[1.12] tracking-tight text-white">
                Intelligent Model Selection.{' '}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-cyan-300 to-blue-200">
                  Simplified.
                </span>
              </h1>

              <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-2xl">
                AURORA is a hybrid AutoML platform that combines performance-based model evaluation with rule-based intelligence to identify suitable machine-learning models for classification and regression tasks.
              </p>

              <div className="flex flex-wrap items-center gap-4 pt-2">
                <button
                  onClick={() => navigate('/upload')}
                  className="px-7 py-4 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-base shadow-electric transition-all flex items-center gap-2.5 group"
                >
                  <span>Start New Analysis</span>
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-0.5 transition-transform" />
                </button>

                <button
                  onClick={() => scrollToSection('how-it-works')}
                  className="px-7 py-4 rounded-xl bg-slate-800/80 hover:bg-slate-800 text-slate-100 font-semibold text-base border border-slate-700 transition-all"
                >
                  Explore How It Works
                </button>
              </div>

              {/* System Architecture Pillars */}
              <div className="pt-4 grid grid-cols-3 gap-4 border-t border-slate-800/90">
                <div>
                  <div className="text-xs font-mono uppercase tracking-wider text-blue-400">
                    Approach 1
                  </div>
                  <div className="text-sm font-semibold text-white mt-0.5">
                    Empirical Evaluation
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">
                    Candidate model training
                  </div>
                </div>
                <div>
                  <div className="text-xs font-mono uppercase tracking-wider text-cyan-400">
                    Approach 2
                  </div>
                  <div className="text-sm font-semibold text-white mt-0.5">
                    Rule Intelligence
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">
                    Task-specific heuristics
                  </div>
                </div>
                <div>
                  <div className="text-xs font-mono uppercase tracking-wider text-indigo-400">
                    Optimization
                  </div>
                  <div className="text-sm font-semibold text-white mt-0.5">
                    RandomizedSearchCV
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">
                    Hyperparameter search
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column: Custom AURORA Workflow & Architecture Preview */}
            <div className="lg:col-span-6">
              <div className="rounded-2xl bg-slate-900/90 border border-slate-700/80 shadow-2xl p-6 backdrop-blur-xl relative overflow-hidden">
                {/* Window Header */}
                <div className="flex items-center justify-between pb-4 mb-5 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-slate-700" />
                    <span className="w-3 h-3 rounded-full bg-slate-700" />
                    <span className="w-3 h-3 rounded-full bg-slate-700" />
                    <span className="ml-2 text-xs font-mono text-slate-400">
                      aurora://workflow-orchestrator
                    </span>
                  </div>
                  <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-blue-500/15 text-blue-300 border border-blue-500/30">
                    Dual-Path Engine
                  </span>
                </div>

                {/* Top Node: Dataset & Preprocessing */}
                <div className="p-4 rounded-xl bg-slate-800/70 border border-slate-700/70 mb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-lg bg-blue-500/20 border border-blue-400/30 flex items-center justify-center text-blue-400">
                        <Database className="w-4 h-4" />
                      </div>
                      <div>
                        <div className="text-xs font-mono uppercase tracking-wider text-slate-400">
                          Stage 1 • Input & Symbolic Analysis
                        </div>
                        <div className="text-sm font-semibold text-white">
                          Preprocessed Tabular Dataset + Task Detection
                        </div>
                      </div>
                    </div>
                    <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-900 text-cyan-300 border border-slate-700">
                      Classification | Regression
                    </span>
                  </div>
                </div>

                {/* Dual-Path Branches: Approach 1 & Approach 2 */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                  {/* Approach 1 Card */}
                  <div className="p-4 rounded-xl bg-gradient-to-b from-blue-950/60 to-slate-900 border border-blue-500/30 flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-blue-500/20 text-blue-300">
                          Approach 1
                        </span>
                        <Cpu className="w-4 h-4 text-blue-400" />
                      </div>
                      <h3 className="text-sm font-bold text-white">
                        Performance-Based Selection
                      </h3>
                      <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                        Trains configured candidate models from the registry and evaluates measured metrics.
                      </p>
                    </div>
                    <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between">
                      <span className="text-xs text-slate-300 font-medium">Output:</span>
                      <span className="text-xs font-mono font-semibold text-blue-300 bg-blue-500/15 px-2.5 py-1 rounded border border-blue-500/30">
                        Model A (Measured Best)
                      </span>
                    </div>
                  </div>

                  {/* Approach 2 Card */}
                  <div className="p-4 rounded-xl bg-gradient-to-b from-cyan-950/50 to-slate-900 border border-cyan-500/30 flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300">
                          Approach 2
                        </span>
                        <Sparkles className="w-4 h-4 text-cyan-400" />
                      </div>
                      <h3 className="text-sm font-bold text-white">
                        Rule-Based Recommendation
                      </h3>
                      <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                        Applies task-specific evaluation rules on dataset characteristics to score suitability.
                      </p>
                    </div>
                    <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between">
                      <span className="text-xs text-slate-300 font-medium">Output:</span>
                      <span className="text-xs font-mono font-semibold text-cyan-300 bg-cyan-500/15 px-2.5 py-1 rounded border border-cyan-500/30">
                        Model B (Single Top Rule)
                      </span>
                    </div>
                  </div>
                </div>

                {/* Convergence Node: Comparison, Tuning & XAI */}
                <div className="p-4 rounded-xl bg-slate-800/80 border border-slate-700/80">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2.5">
                      <GitCompare className="w-4 h-4 text-blue-400" />
                      <span className="text-xs font-bold uppercase tracking-wider text-white">
                        Unified Evaluation & Optimization Layer
                      </span>
                    </div>
                    <span className="text-[11px] font-mono text-slate-400">
                      Model A vs. Model B
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                      <div className="text-[11px] font-semibold text-slate-200">
                        Approach Comparison
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        Empirical vs. Heuristic
                      </div>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                      <div className="text-[11px] font-semibold text-slate-200">
                        Hyperparameter Tuning
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        RandomizedSearchCV
                      </div>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-900/90 border border-slate-800">
                      <div className="text-[11px] font-semibold text-slate-200">
                        Validation & SHAP
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        5-Fold CV + Explainability
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Key Feature Section */}
      <section id="features" className="py-20 lg:py-24 bg-slate-50 border-b border-slate-200/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <span className="text-xs font-mono uppercase tracking-widest px-3 py-1 rounded-full bg-blue-100 text-blue-700 font-semibold">
              Core Platform Capabilities
            </span>
            <h2 className="mt-4 font-display font-extrabold text-3xl sm:text-4xl text-navy-900 tracking-tight">
              One Dataset. Two Intelligent Approaches.
            </h2>
            <p className="mt-3 text-slate-600 text-base sm:text-lg">
              AURORA bridges empirical candidate model benchmarking with deterministic rule-based reasoning in one cohesive environment.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Feature 1 */}
            <div className="aurora-card aurora-card-hover p-8 flex flex-col justify-between">
              <div>
                <div className="w-12 h-12 rounded-xl bg-blue-600/10 border border-blue-600/20 flex items-center justify-center text-blue-600 mb-6">
                  <BarChart3 className="w-6 h-6" />
                </div>
                <div className="text-xs font-mono uppercase tracking-wider text-blue-600 font-semibold mb-1">
                  Feature 01 • Empirical Evaluation
                </div>
                <h3 className="font-display font-bold text-xl text-navy-900 mb-3">
                  Performance-Based Selection
                </h3>
                <p className="text-slate-600 leading-relaxed">
                  Train and evaluate configured candidate models to identify the best-performing model using the selected evaluation metric.
                </p>
              </div>
              <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                <span>Selects Model A via measured metrics</span>
                <span className="font-mono text-blue-600 font-semibold">Approach 1</span>
              </div>
            </div>

            {/* Feature 2 */}
            <div className="aurora-card aurora-card-hover p-8 flex flex-col justify-between">
              <div>
                <div className="w-12 h-12 rounded-xl bg-cyan-600/10 border border-cyan-600/20 flex items-center justify-center text-cyan-600 mb-6">
                  <Sparkles className="w-6 h-6" />
                </div>
                <div className="text-xs font-mono uppercase tracking-wider text-cyan-700 font-semibold mb-1">
                  Feature 02 • Heuristic Intelligence
                </div>
                <h3 className="font-display font-bold text-xl text-navy-900 mb-3">
                  Rule-Based Recommendation
                </h3>
                <p className="text-slate-600 leading-relaxed">
                  Leverage AURORA’s existing rule engine to identify the highest-scoring model recommendation based on dataset characteristics.
                </p>
              </div>
              <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                <span>Selects single top-scoring Model B</span>
                <span className="font-mono text-cyan-700 font-semibold">Approach 2</span>
              </div>
            </div>

            {/* Feature 3 */}
            <div className="aurora-card aurora-card-hover p-8 flex flex-col justify-between">
              <div>
                <div className="w-12 h-12 rounded-xl bg-indigo-600/10 border border-indigo-600/20 flex items-center justify-center text-indigo-600 mb-6">
                  <GitCompare className="w-6 h-6" />
                </div>
                <div className="text-xs font-mono uppercase tracking-wider text-indigo-600 font-semibold mb-1">
                  Feature 03 • Transparent Analysis
                </div>
                <h3 className="font-display font-bold text-xl text-navy-900 mb-3">
                  Model Comparison
                </h3>
                <p className="text-slate-600 leading-relaxed">
                  Compare the outputs of both approaches and understand how performance-based selection differs from rule-based recommendation.
                </p>
              </div>
              <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                <span>Side-by-side methodology & metric breakdown</span>
                <span className="font-mono text-indigo-600 font-semibold">Model A vs. Model B</span>
              </div>
            </div>

            {/* Feature 4 */}
            <div className="aurora-card aurora-card-hover p-8 flex flex-col justify-between">
              <div>
                <div className="w-12 h-12 rounded-xl bg-emerald-600/10 border border-emerald-600/20 flex items-center justify-center text-emerald-600 mb-6">
                  <Sliders className="w-6 h-6" />
                </div>
                <div className="text-xs font-mono uppercase tracking-wider text-emerald-700 font-semibold mb-1">
                  Feature 04 • Configuration Search
                </div>
                <h3 className="font-display font-bold text-xl text-navy-900 mb-3">
                  Hyperparameter Tuning
                </h3>
                <p className="text-slate-600 leading-relaxed">
                  Explore the existing hyperparameter-tuning capability to search for suitable model configurations.
                </p>
              </div>
              <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                <span>RandomizedSearchCV with Stratified K-Fold</span>
                <span className="font-mono text-emerald-700 font-semibold">Optimization</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 lg:py-24 bg-white border-b border-slate-200/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <span className="text-xs font-mono uppercase tracking-widest px-3 py-1 rounded-full bg-slate-100 text-slate-700 font-semibold">
              End-to-End Pipeline
            </span>
            <h2 className="mt-4 font-display font-extrabold text-3xl sm:text-4xl text-navy-900 tracking-tight">
              From Dataset to Model Insight
            </h2>
            <p className="mt-3 text-slate-600 text-base sm:text-lg">
              A guided workflow that brings dataset analysis, model selection, and evaluation into one unified experience.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                step: '01',
                title: 'Upload Dataset',
                desc: 'Upload a CSV dataset for validation and automated column-level preprocessing analysis.',
                icon: UploadCloud,
                link: '/upload',
              },
              {
                step: '02',
                title: 'Review Dataset',
                desc: 'Inspect rows, columns, target variable, task type, missingness, data types, and symbolic recommendations.',
                icon: FileSpreadsheet,
                link: '/overview',
              },
              {
                step: '03',
                title: 'Select Model-Selection Approach',
                desc: 'Choose Performance-Based Selection (Approach 1), Rule-Based Recommendation (Approach 2), or both.',
                icon: Layers,
                link: '/select-approach',
              },
              {
                step: '04',
                title: 'Analyze Results',
                desc: 'Explore Model A and Model B outputs, compare methodologies, and run hyperparameter optimization.',
                icon: BarChart3,
                link: '/results',
              },
            ].map((item, idx) => {
              const Icon = item.icon;
              return (
                <div
                  key={item.step}
                  onClick={() => navigate(item.link)}
                  className="cursor-pointer group relative rounded-2xl bg-slate-50/80 hover:bg-white border border-slate-200/90 hover:border-blue-500/40 p-6 shadow-sm hover:shadow-premium-lg transition-all flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-6">
                      <span className="font-mono font-bold text-xs px-2.5 py-1 rounded-md bg-navy-900 text-white">
                        STEP {item.step}
                      </span>
                      <div className="w-10 h-10 rounded-xl bg-blue-600/10 text-blue-600 flex items-center justify-center group-hover:bg-blue-600 group-hover:text-white transition-colors">
                        <Icon className="w-5 h-5" />
                      </div>
                    </div>
                    <h3 className="font-display font-bold text-lg text-navy-900 mb-2 group-hover:text-blue-600 transition-colors">
                      {item.title}
                    </h3>
                    <p className="text-sm text-slate-600 leading-relaxed">{item.desc}</p>
                  </div>
                  <div className="mt-6 pt-4 border-t border-slate-200/60 flex items-center justify-between text-xs font-semibold text-blue-600">
                    <span>Explore Stage {idx + 1}</span>
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Technical Architecture Highlight Section */}
      <section id="architecture-preview" className="py-16 bg-slate-50 border-b border-slate-200/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="aurora-card p-8 lg:p-10 bg-white">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
              <div className="lg:col-span-7 space-y-4">
                <span className="text-xs font-mono uppercase tracking-wider text-blue-600 font-semibold">
                  Preserving Authentic Backend Logic
                </span>
                <h3 className="font-display font-bold text-2xl sm:text-3xl text-navy-900">
                  Built Directly on AURORA’s Symbolic & Empirical Engine
                </h3>
                <p className="text-slate-600 text-sm sm:text-base leading-relaxed">
                  Every stage in the web platform invokes the underlying Python modules directly—from column profiling and 185+ preprocessing rules to candidate model training, task-specific model recommendation rules, 5-Fold Stratified Cross-Validation, and SHAP explainability.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                  {[
                    'Symbolic + Neural Column Preprocessor',
                    'Classification & Regression Task Detection',
                    'Single Highest-Scoring Rule Selection (Model B)',
                    'Empirical Candidate Benchmarking (Model A)',
                    'RandomizedSearchCV Hyperparameter Tuning',
                    '5-Fold Cross-Validation & SHAP Summary Plots',
                  ].map((point) => (
                    <div key={point} className="flex items-center gap-2 text-sm text-slate-700">
                      <CheckCircle2 className="w-4 h-4 text-blue-600 flex-shrink-0" />
                      <span>{point}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="lg:col-span-5 bg-navy-900 text-white rounded-2xl p-6 border border-slate-800 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <span className="text-xs font-mono text-blue-300 uppercase tracking-wider">
                    Workflow Guarantees
                  </span>
                  <ShieldCheck className="w-4 h-4 text-cyan-400" />
                </div>
                <ul className="space-y-3 text-xs text-slate-300 leading-relaxed">
                  <li className="flex items-start gap-2.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5 flex-shrink-0" />
                    <span>
                      <strong className="text-white">Model A (Approach 1):</strong> Selected strictly through measured evaluation metrics across configured candidate models.
                    </span>
                  </li>
                  <li className="flex items-start gap-2.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 flex-shrink-0" />
                    <span>
                      <strong className="text-white">Model B (Approach 2):</strong> Selected strictly as the single highest-scoring recommendation from AURORA’s deterministic rule engine.
                    </span>
                  </li>
                  <li className="flex items-start gap-2.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 mt-1.5 flex-shrink-0" />
                    <span>
                      <strong className="text-white">Clear Metric Separation:</strong> Heuristic rule recommendation scores are never conflated with predictive accuracy or F1-score.
                    </span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Final Call-To-Action Section */}
      <section className="py-20 bg-navy-hero text-white relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-navy pointer-events-none" />
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10 space-y-6">
          <h2 className="font-display font-extrabold text-3xl sm:text-4xl lg:text-5xl tracking-tight text-white">
            Ready to Find the Right Model?
          </h2>
          <p className="text-slate-300 text-base sm:text-lg max-w-2xl mx-auto leading-relaxed">
            Start your analysis with AURORA and explore intelligent model selection through a unified workflow.
          </p>
          <div className="pt-4 flex flex-wrap items-center justify-center gap-4">
            <button
              onClick={() => navigate('/dashboard')}
              className="px-8 py-4 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-base shadow-electric transition-all inline-flex items-center gap-2.5 group"
            >
              <span>Launch AURORA</span>
              <ArrowRight className="w-5 h-5 group-hover:translate-x-0.5 transition-transform" />
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-navy-950 text-slate-400 border-t border-slate-800/80 py-14">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-10 pb-10 border-b border-slate-800/80">
            {/* Brand & Description */}
            <div className="lg:col-span-2 space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center">
                  <Sparkles className="w-4 h-4 text-white" />
                </div>
                <span className="font-display font-extrabold text-lg text-white tracking-tight">
                  AURORA
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed max-w-sm">
                AURORA — Intelligent Model Recommendation System. A hybrid AutoML platform combining empirical candidate model evaluation with rule-based intelligence for tabular classification and regression datasets.
              </p>
            </div>

            {/* Product */}
            <div className="space-y-2.5">
              <h4 className="text-xs font-mono uppercase tracking-wider text-white font-semibold">
                Product
              </h4>
              <ul className="space-y-2 text-xs">
                <li>
                  <Link to="/dashboard" className="hover:text-white transition-colors">
                    Application Dashboard
                  </Link>
                </li>
                <li>
                  <Link to="/upload" className="hover:text-white transition-colors">
                    Dataset Upload
                  </Link>
                </li>
                <li>
                  <Link to="/overview" className="hover:text-white transition-colors">
                    Dataset Overview
                  </Link>
                </li>
              </ul>
            </div>

            {/* How It Works */}
            <div className="space-y-2.5">
              <h4 className="text-xs font-mono uppercase tracking-wider text-white font-semibold">
                How It Works
              </h4>
              <ul className="space-y-2 text-xs">
                <li>
                  <button
                    onClick={() => scrollToSection('how-it-works')}
                    className="hover:text-white transition-colors"
                  >
                    4-Stage Guided Workflow
                  </button>
                </li>
                <li>
                  <Link to="/processing" className="hover:text-white transition-colors">
                    Workflow Execution
                  </Link>
                </li>
                <li>
                  <Link to="/results" className="hover:text-white transition-colors">
                    Results Dashboard
                  </Link>
                </li>
              </ul>
            </div>

            {/* Model Selection */}
            <div className="space-y-2.5">
              <h4 className="text-xs font-mono uppercase tracking-wider text-white font-semibold">
                Model Selection
              </h4>
              <ul className="space-y-2 text-xs">
                <li>
                  <Link to="/select-approach" className="hover:text-white transition-colors">
                    Approach 1 — Performance
                  </Link>
                </li>
                <li>
                  <Link to="/select-approach" className="hover:text-white transition-colors">
                    Approach 2 — Rule Engine
                  </Link>
                </li>
                <li>
                  <Link to="/comparison" className="hover:text-white transition-colors">
                    Model Comparison
                  </Link>
                </li>
                <li>
                  <Link to="/tuning" className="hover:text-white transition-colors">
                    Hyperparameter Tuning
                  </Link>
                </li>
              </ul>
            </div>

            {/* Documentation & Project Info */}
            <div className="space-y-2.5">
              <h4 className="text-xs font-mono uppercase tracking-wider text-white font-semibold">
                Documentation & Project Info
              </h4>
              <ul className="space-y-2 text-xs text-slate-400">
                <li>Automated Model Evaluation Framework (AMEF)</li>
                <li>185+ Symbolic Preprocessing Rules</li>
                <li>160 Classification + 85 Regression Rules</li>
                <li>5-Fold Stratified CV & SHAP XAI</li>
              </ul>
            </div>
          </div>

          <div className="pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
            <div>
              AURORA — Intelligent Model Recommendation System • Academic & Engineering Demonstration Platform
            </div>
            <div className="font-mono text-[11px] text-slate-500">
              Powered by Python, FastAPI, Scikit-Learn, XGBoost, LightGBM, CatBoost & SHAP
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

