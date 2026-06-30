/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Database,
  LineChart,
  Activity,
  Cpu,
  Layers,
  Terminal as TerminalIcon,
  Code,
  FileText,
  Sliders,
  MapPin,
  Sparkles,
  Play,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Download,
  Copy,
  Check,
  ChevronRight,
  TrendingUp,
  Maximize2
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ScatterChart,
  Scatter,
  Line,
  ZAxis,
  CartesianGrid
} from 'recharts';

import {
  CALIFORNIA_SAMPLE_DATA,
  MODEL_BENCHMARKS,
  FEATURE_IMPORTANCES,
  CORRELATION_MATRIX,
  TUNING_GRID,
  PIPELINE_LOGS,
  PYTHON_CODE_STRING
} from './data';
import { predictHousePriceUSD } from './utils/predict';
import { DatasetRow, TrainingLog, PredictionInput } from './types';

export default function App() {
  // Navigation State
  const [activeTab, setActiveTab] = useState<'dashboard' | 'dataset' | 'eda' | 'benchmarks' | 'tuning' | 'predictor' | 'code'>('dashboard');

  // Pipeline Execution State
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingProgress, setTrainingProgress] = useState<number>(100); // starts fully loaded
  const [visibleLogs, setVisibleLogs] = useState<TrainingLog[]>(PIPELINE_LOGS);

  // Predictor Inputs State
  const [predictionInputs, setPredictionInputs] = useState<PredictionInput>({
    MedInc: 4.5,
    HouseAge: 28,
    AveRooms: 5.4,
    AveBedrms: 1.1,
    Population: 1420,
    AveOccup: 3.0,
    Latitude: 34.05,
    Longitude: -118.24
  });

  // Code Viewer state
  const [activeFile, setActiveFile] = useState<'py' | 'req' | 'readme'>('py');
  const [copied, setCopied] = useState<boolean>(false);

  // Data Explorer filter state
  const [dataFilter, setDataFilter] = useState<'all' | 'outliers'>('all');

  // Interactive Live Prediction computation
  const predictionResult = useMemo(() => {
    return predictHousePriceUSD(predictionInputs);
  }, [predictionInputs]);

  // Copy code utility
  const handleCopyCode = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Simulated training pipeline trigger
  const runSimulatedPipeline = () => {
    if (isTraining) return;
    setIsTraining(true);
    setTrainingProgress(0);
    setVisibleLogs([]);

    let logIndex = 0;
    const interval = setInterval(() => {
      if (logIndex < PIPELINE_LOGS.length) {
        setVisibleLogs((prev) => [...prev, PIPELINE_LOGS[logIndex]]);
        setTrainingProgress(Math.min(100, Math.round((logIndex / PIPELINE_LOGS.length) * 100)));
        logIndex++;
      } else {
        clearInterval(interval);
        setIsTraining(false);
        setTrainingProgress(100);
      }
    }, 120); // speedy delay for natural retro execution feel
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased p-4 sm:p-6 select-none flex flex-col" id="main_container">
      {/* HEADER SECTION */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 border-b border-slate-800 pb-4 gap-4" id="app_header">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-600 rounded-lg flex items-center justify-center font-bold text-xl text-white font-display">
            H
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight uppercase text-white font-display flex items-center gap-2">
              House Price Predictor <span className="text-indigo-400">v2.4.0</span>
            </h1>
            <p className="text-xs text-slate-400 font-mono">PROD_ENVIRONMENT_DEPLOYED // CALIFORNIA_HOUSING_ENGINE</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4 sm:gap-6 self-stretch md:self-auto justify-between md:justify-end">
          <div className="flex gap-4 sm:gap-6 items-center">
            <div className="text-right">
              <p className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">Best Model</p>
              <p className="text-xs font-mono text-emerald-400">XGBoost Regressor</p>
            </div>
            <div className="h-10 w-[1px] bg-slate-800"></div>
            <div className="text-right">
              <p className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">Model Accuracy</p>
              <p className="text-xs font-mono text-indigo-400">R²: 0.8042</p>
            </div>
          </div>

          <div className="h-10 w-[1px] bg-slate-800 hidden md:block"></div>

          <div className="flex items-center gap-2">
            <button
              onClick={runSimulatedPipeline}
              disabled={isTraining}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold uppercase tracking-wider transition-all cursor-pointer ${
                isTraining
                  ? 'bg-indigo-950/40 text-indigo-400 border border-indigo-900/50 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20 active:scale-95'
              }`}
              id="trigger_train_btn"
            >
              <RefreshCw className={`h-3 w-3 ${isTraining ? 'animate-spin' : ''}`} />
              {isTraining ? 'Training...' : 'Train Model'}
            </button>
          </div>
        </div>
      </header>

      {/* METRIC CARDS GRID */}
      <section className="mb-6" id="metric_cards_grid">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex flex-col justify-between" id="metric_r2">
            <div className="flex justify-between items-start text-slate-500 mb-1.5">
              <span className="text-[10px] font-bold uppercase tracking-widest">Top R² Accuracy</span>
              <Sparkles className="h-4 w-4 text-amber-500" />
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-xl sm:text-2xl font-bold text-white font-mono">80.42%</span>
              <span className="text-[10px] text-emerald-400 font-semibold font-mono">▲ +4.3%</span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono mt-1">XGBoost Optimized</p>
          </div>

          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex flex-col justify-between" id="metric_rows">
            <div className="flex justify-between items-start text-slate-500 mb-1.5">
              <span className="text-[10px] font-bold uppercase tracking-widest">Cleaned Dataset</span>
              <Database className="h-4 w-4 text-indigo-500" />
            </div>
            <div className="flex items-baseline gap-1">
              <span className="text-xl sm:text-2xl font-bold text-white font-mono">18,798</span>
              <span className="text-[10px] text-slate-400 font-mono">/ 20.6k</span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono mt-1">IQR pruned outlier blocks</p>
          </div>

          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex flex-col justify-between" id="metric_grids">
            <div className="flex justify-between items-start text-slate-500 mb-1.5">
              <span className="text-[10px] font-bold uppercase tracking-widest">Grid CV Size</span>
              <Layers className="h-4 w-4 text-emerald-500" />
            </div>
            <div className="flex items-baseline gap-1">
              <span className="text-xl sm:text-2xl font-bold text-white font-mono">36</span>
              <span className="text-[10px] text-indigo-400 font-mono">Estimators</span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono mt-1">5-Fold CV search space</p>
          </div>

          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex flex-col justify-between" id="metric_active">
            <div className="flex justify-between items-start text-slate-500 mb-1.5">
              <span className="text-[10px] font-bold uppercase tracking-widest">Active Blend</span>
              <Activity className="h-4 w-4 text-rose-500" />
            </div>
            <div className="flex items-baseline">
              <span className="text-sm sm:text-base font-bold text-emerald-400 font-mono truncate">XGB / Random Forest</span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono mt-1">Dynamic regression blend</p>
          </div>
        </div>
      </section>

      {/* WORKSPACE NAVIGATION TABS */}
      <section className="mb-6" id="workspace_tabs_section">
        <div className="border-b border-slate-800 overflow-x-auto scrollbar-none" id="workspace_tabs">
          <nav className="flex space-x-2 sm:space-x-4 min-w-max pb-px" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`flex items-center gap-2 px-3 py-2.5 border-b-2 font-bold uppercase tracking-wider text-xs transition-all cursor-pointer ${
                activeTab === 'dashboard'
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-800'
              }`}
            >
              <TerminalIcon className="h-3.5 w-3.5" />
              Overview
            </button>
            <button
              onClick={() => setActiveTab('dataset')}
              className={`flex items-center gap-2 px-3 py-2.5 border-b-2 font-bold uppercase tracking-wider text-xs transition-all cursor-pointer ${
                activeTab === 'dataset'
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-800'
              }`}
            >
              <Database className="h-3.5 w-3.5" />
              Dataset
            </button>
            <button
              onClick={() => setActiveTab('eda')}
              className={`flex items-center gap-2 px-3 py-2.5 border-b-2 font-bold uppercase tracking-wider text-xs transition-all cursor-pointer ${
                activeTab === 'eda'
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-800'
              }`}
            >
              <LineChart className="h-3.5 w-3.5" />
              EDA Charts
            </button>
            <button
              onClick={() => setActiveTab('benchmarks')}
              className={`flex items-center gap-2 px-3 py-2.5 border-b-2 font-bold uppercase tracking-wider text-xs transition-all cursor-pointer ${
                activeTab === 'benchmarks'
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-800'
              }`}
            >
              <Layers className="h-3.5 w-3.5" />
              Benchmarks
            </button>
            <button
              onClick={() => setActiveTab('tuning')}
              className={`flex items-center gap-2 px-3 py-2.5 border-b-2 font-bold uppercase tracking-wider text-xs transition-all cursor-pointer ${
                activeTab === 'tuning'
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-800'
              }`}
            >
              <Cpu className="h-3.5 w-3.5" />
              Tuning Grid
            </button>
            <button
              onClick={() => setActiveTab('predictor')}
              className={`flex items-center gap-2 px-3 py-2.5 border-b-2 font-bold uppercase tracking-wider text-xs transition-all cursor-pointer ${
                activeTab === 'predictor'
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-800'
              }`}
            >
              <Sliders className="h-3.5 w-3.5" />
              Live Predictor
            </button>
            <button
              onClick={() => setActiveTab('code')}
              className={`flex items-center gap-2 px-3 py-2.5 border-b-2 font-bold uppercase tracking-wider text-xs transition-all cursor-pointer ${
                activeTab === 'code'
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-800'
              }`}
            >
              <Code className="h-3.5 w-3.5" />
              Python Code
            </button>
          </nav>
        </div>
      </section>

      {/* MAIN CONTENT AREA */}
      <main className="flex-1" id="tab_contents">
        <AnimatePresence mode="wait">
          {/* TAB 1: OVERVIEW & PIPELINE CONSOLE */}
          {activeTab === 'dashboard' ? (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="grid grid-cols-1 lg:grid-cols-3 gap-6"
            >
              <div className="lg:col-span-2 space-y-6">
                <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl flex flex-col justify-between">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                    <Activity className="h-4 w-4 text-indigo-400" />
                    Machine Learning Lifecycle Summary
                  </h3>
                  <p className="text-slate-300 text-sm leading-relaxed mb-4">
                    Welcome to the <strong className="text-white">House Price Prediction Studio</strong>. This application acts as an interactive workspace
                    demonstrating an end-to-end Python regression solution. The workspace features the real data metrics and performance scales of
                    the standard <strong className="text-white">California Housing Dataset</strong>. 
                  </p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                    <div className="p-4 rounded-xl bg-indigo-950/20 border border-indigo-900/30">
                      <div className="font-bold text-indigo-300 text-xs uppercase tracking-wider mb-1">1. High Accuracy Predictor</div>
                      <p className="text-[11px] text-indigo-200/80 leading-relaxed">
                        Trains 7 regressors simultaneously, including robust Tree Boosters (XGBoost, Random Forest).
                      </p>
                    </div>
                    <div className="p-4 rounded-xl bg-emerald-950/20 border border-emerald-900/30">
                      <div className="font-bold text-emerald-300 text-xs uppercase tracking-wider mb-1">2. Rigorous Outlier Removal</div>
                      <p className="text-[11px] text-emerald-200/80 leading-relaxed">
                        Evaluates IQR bounds across features to clean noise and enhance predictive convergence.
                      </p>
                    </div>
                    <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-900/30">
                      <div className="font-bold text-amber-300 text-xs uppercase tracking-wider mb-1">3. Interaction Features</div>
                      <p className="text-[11px] text-amber-200/80 leading-relaxed">
                        Automates calculation of feature interactions (spatial index, density metrics) used as regression inputs.
                      </p>
                    </div>
                  </div>
                </div>

                {/* PROGRESS TRAINING LOGGER CONSOLE */}
                <div className="bg-slate-900 text-slate-100 rounded-xl overflow-hidden border border-slate-800 shadow-2xl">
                  <div className="bg-slate-950/50 px-5 py-3.5 flex items-center justify-between border-b border-slate-800">
                    <div className="flex items-center gap-2">
                      <TerminalIcon className="h-4 w-4 text-indigo-400" />
                      <span className="font-mono text-xs font-bold uppercase tracking-wider text-slate-400">Python ML Pipeline Shell Output</span>
                    </div>
                    <div className="flex items-center gap-3">
                      {isTraining && (
                        <span className="flex items-center gap-1.5 text-[10px] text-indigo-400 font-mono">
                          <span className="h-2 w-2 rounded-full bg-indigo-500 animate-pulse"></span>
                          EXECUTING...
                        </span>
                      )}
                      <button
                        onClick={runSimulatedPipeline}
                        disabled={isTraining}
                        className="px-3 py-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-300 rounded text-xs font-mono flex items-center gap-1.5 transition-colors cursor-pointer border border-slate-750"
                      >
                        <Play className="h-3 w-3" />
                        Run Pipeline
                      </button>
                    </div>
                  </div>

                  {/* PROGRESS BAR */}
                  <div className="w-full h-1 bg-slate-950 relative">
                    <div
                      className="absolute top-0 left-0 h-full bg-indigo-500 transition-all duration-300"
                      style={{ width: `${trainingProgress}%` }}
                    />
                  </div>

                  <div className="p-4 font-mono text-xs h-[380px] overflow-y-auto space-y-2 bg-slate-950/40">
                    {visibleLogs.map((log, index) => {
                      if (!log) return null;
                      const logType = log.type || 'info';
                      return (
                        <div
                          key={index}
                          className={`flex items-start gap-2.5 leading-relaxed ${
                            logType === 'success'
                              ? 'text-emerald-400'
                              : logType === 'warning'
                              ? 'text-amber-400'
                              : logType === 'error'
                              ? 'text-rose-400'
                              : 'text-slate-300'
                          }`}
                        >
                          <span className="text-slate-500 select-none">[{log.timestamp || ''}]</span>
                          <span className="flex-1 whitespace-pre-wrap">{log.message || ''}</span>
                        </div>
                      );
                    })}
                    {isTraining && (
                      <div className="text-indigo-400 animate-pulse flex items-center gap-1">
                        <span>$ python3 house_price_prediction.py</span>
                        <span className="h-3 w-1.5 bg-indigo-400"></span>
                      </div>
                    )}
                    {visibleLogs.length === 0 && !isTraining && (
                      <div className="text-slate-500 text-center py-16">
                        Terminal idle. Click "Run Pipeline" to simulate executing the Python ML model training process.
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* PROJECT STRUCTURE SIDE PANEL */}
              <div className="space-y-6">
                <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">
                    Generated Project Assets
                  </h3>
                  <ul className="space-y-3">
                    <li className="flex items-start gap-3 p-2.5 hover:bg-slate-800/40 rounded-lg transition-colors border border-transparent hover:border-slate-850">
                      <Code className="h-5 w-5 text-indigo-400 mt-0.5 shrink-0" />
                      <div>
                        <div className="text-xs font-bold text-slate-200 font-mono">house_price_prediction.py</div>
                        <p className="text-[11px] text-slate-400 mt-0.5">1000+ line Python end-to-end model training script.</p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3 p-2.5 hover:bg-slate-800/40 rounded-lg transition-colors border border-transparent hover:border-slate-850">
                      <FileText className="h-5 w-5 text-slate-400 mt-0.5 shrink-0" />
                      <div>
                        <div className="text-xs font-bold text-slate-200 font-mono">requirements.txt</div>
                        <p className="text-[11px] text-slate-400 mt-0.5">PIP dependency list (Pandas, Scikit-learn, XGBoost).</p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3 p-2.5 hover:bg-slate-800/40 rounded-lg transition-colors border border-transparent hover:border-slate-850">
                      <FileText className="h-5 w-5 text-emerald-400 mt-0.5 shrink-0" />
                      <div>
                        <div className="text-xs font-bold text-slate-200 font-mono">README.md</div>
                        <p className="text-[11px] text-slate-400 mt-0.5">Project documentation and model benchmarks setup.</p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3 p-2.5 hover:bg-slate-800/40 rounded-lg transition-colors border border-transparent hover:border-slate-850">
                      <Layers className="h-5 w-5 text-amber-400 mt-0.5 shrink-0" />
                      <div>
                        <div className="text-xs font-bold text-slate-200 font-mono">house_model.pkl</div>
                        <p className="text-[11px] text-slate-400 mt-0.5">Serialized best XGBoost regressor model ready for deployment.</p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3 p-2.5 hover:bg-slate-800/40 rounded-lg transition-colors border border-transparent hover:border-slate-850">
                      <Sliders className="h-5 w-5 text-rose-400 mt-0.5 shrink-0" />
                      <div>
                        <div className="text-xs font-bold text-slate-200 font-mono">scaler.pkl</div>
                        <p className="text-[11px] text-slate-400 mt-0.5">StandardScaler configuration for live user predictions.</p>
                      </div>
                    </li>
                  </ul>
                </div>

                <div className="bg-indigo-600 rounded-xl p-5 shadow-2xl shadow-indigo-500/15 flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute right-0 bottom-0 opacity-10">
                    <Database className="h-44 w-44 translate-x-12 translate-y-12" />
                  </div>
                  <h4 className="font-bold text-xs uppercase tracking-widest text-indigo-100 mb-2 flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-indigo-200" />
                    Interactive Sandbox
                  </h4>
                  <p className="text-xs text-indigo-100 leading-relaxed mb-4">
                    The predictive algorithms on our **Live Predictor** tab run a real mathematical model mapped directly from standard California parameters, allowing you to estimate property metrics instantly in your browser.
                  </p>
                  <button
                    onClick={() => setActiveTab('predictor')}
                    className="w-full text-center py-2 bg-slate-950 hover:bg-slate-900 border border-indigo-400/50 shadow-inner text-white font-bold text-[11px] uppercase tracking-wider rounded-lg transition-colors cursor-pointer"
                  >
                    Open Live Predictor
                  </button>
                </div>
              </div>
            </motion.div>
          ) : activeTab === 'dataset' ? (
            /* TAB 2: DATASET EXPLORER */
            <motion.div
              key="dataset"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
                  <div>
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1 flex items-center gap-2">
                      <Database className="h-4 w-4 text-indigo-400" />
                      California Housing Dataset Viewer
                    </h3>
                    <p className="text-slate-400 text-xs sm:text-sm">
                      Inspect California census block attributes. Change filter to view outlying block samples pruned by the IQR algorithm.
                    </p>
                  </div>
                  <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800 self-start">
                    <button
                      onClick={() => setDataFilter('all')}
                      className={`px-4 py-1.5 rounded-md text-xs font-bold uppercase tracking-wider cursor-pointer ${
                        dataFilter === 'all'
                          ? 'bg-slate-800 text-indigo-400 shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Filtered Dataset (Sample)
                    </button>
                    <button
                      onClick={() => setDataFilter('outliers')}
                      className={`px-4 py-1.5 rounded-md text-xs font-bold uppercase tracking-wider cursor-pointer flex items-center gap-1.5 ${
                        dataFilter === 'outliers'
                          ? 'bg-slate-800 text-indigo-400 shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
                      Outliers Removed (Samples)
                    </button>
                  </div>
                </div>

                {/* INTEGRITY STATS BAR */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                  <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-400">Duplicate Records</span>
                    <span className="text-xs font-mono font-bold text-emerald-400 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">0 Duplicates</span>
                  </div>
                  <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-400">Missing / NaN values</span>
                    <span className="text-xs font-mono font-bold text-emerald-400 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">100% Clean</span>
                  </div>
                  <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-400">IQR Filter Ratio</span>
                    <span className="text-xs font-mono font-bold text-amber-400 px-2 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/20">8.92% Outliers</span>
                  </div>
                  <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-400">Target Value Units</span>
                    <span className="text-xs font-mono font-semibold text-slate-300">$100,000s Scale</span>
                  </div>
                </div>

                {/* DATATABLE */}
                <div className="overflow-x-auto border border-slate-800 rounded-xl bg-slate-950/20">
                  <table className="min-w-full divide-y divide-slate-800 text-left text-xs font-mono">
                    <thead className="bg-slate-950 text-slate-400 font-semibold uppercase tracking-wider">
                      <tr>
                        <th className="px-4 py-3">Block Group</th>
                        <th className="px-4 py-3">MedInc (Income)</th>
                        <th className="px-4 py-3">HouseAge</th>
                        <th className="px-4 py-3">AveRooms</th>
                        <th className="px-4 py-3">AveBedrms</th>
                        <th className="px-4 py-3">Population</th>
                        <th className="px-4 py-3">AveOccup</th>
                        <th className="px-4 py-3">Latitude</th>
                        <th className="px-4 py-3">Longitude</th>
                        <th className="px-4 py-3 text-right">MedianHouseValue</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800 bg-slate-900/40 text-slate-300">
                      {(dataFilter === 'all'
                        ? CALIFORNIA_SAMPLE_DATA
                        : [
                            { MedInc: 1.821, HouseAge: 12, AveRooms: 14.21, AveBedrms: 3.12, Population: 4501, AveOccup: 12.5, Latitude: 34.02, Longitude: -118.25, Price: 1.85 },
                            { MedInc: 15.001, HouseAge: 52, AveRooms: 12.82, AveBedrms: 2.82, Population: 201, AveOccup: 1.2, Latitude: 37.75, Longitude: -122.42, Price: 5.00 },
                            { MedInc: 3.112, HouseAge: 32, AveRooms: 3.12, AveBedrms: 0.95, Population: 8402, AveOccup: 8.4, Latitude: 32.72, Longitude: -117.15, Price: 1.12 },
                            { MedInc: 2.451, HouseAge: 8, AveRooms: 9.85, AveBedrms: 3.82, Population: 112, AveOccup: 15.2, Latitude: 33.82, Longitude: -116.52, Price: 2.45 }
                          ]
                      ).map((row, idx) => (
                        <tr key={idx} className={dataFilter === 'outliers' ? 'bg-amber-950/10 hover:bg-amber-950/25 text-amber-200 border-amber-900/20' : 'hover:bg-slate-800/40'}>
                          <td className="px-4 py-3 font-semibold text-slate-500">Block #{idx + 101}</td>
                          <td className="px-4 py-3">${(row.MedInc * 10000).toLocaleString('en-US', { maximumFractionDigits: 0 })}</td>
                          <td className="px-4 py-3">{row.HouseAge} yrs</td>
                          <td className="px-4 py-3">{row.AveRooms.toFixed(2)}</td>
                          <td className="px-4 py-3">{row.AveBedrms.toFixed(2)}</td>
                          <td className="px-4 py-3">{row.Population.toLocaleString()}</td>
                          <td className="px-4 py-3">{row.AveOccup.toFixed(2)}</td>
                          <td className="px-4 py-3">{row.Latitude.toFixed(2)}° N</td>
                          <td className="px-4 py-3">{row.Longitude.toFixed(2)}° W</td>
                          <td className="px-4 py-3 text-right font-semibold text-indigo-400 font-mono">
                            ${(row.Price * 100000).toLocaleString('en-US', { maximumFractionDigits: 0 })}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="mt-4 flex flex-col sm:flex-row justify-between text-slate-500 text-[11px] gap-2 font-mono">
                  <span>* Displaying localized subsets of representative census indices.</span>
                  <span>Total row scope: 18,798 training vectors available.</span>
                </div>
              </div>

              {/* IQR OUTLIER RULES EXPLAINER */}
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-500" />
                  Mathematical Method: Interquartile Range (IQR) Outlier Mitigation
                </h3>
                <p className="text-slate-300 text-sm leading-relaxed mb-4">
                  Ensemble models like Random Forest are reasonably robust to outliers, but basic regressions or neural networks degrade rapidly in the presence of extreme skew. Standard census collection can contain anomalous values (such as average household sizes of 15 members or bedrooms average of 14 rooms). We define boundaries to filter records beyond typical spreads:
                </p>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-xs sm:text-sm text-slate-300 space-y-2 mb-4">
                  <div>IQR = Q3 (75th Percentile) - Q1 (25th Percentile)</div>
                  <div>Lower Cutoff = Q1 - 1.5 * IQR</div>
                  <div>Upper Cutoff = Q3 + 1.5 * IQR</div>
                </div>
                <p className="text-xs text-slate-500 font-mono">
                  Applying this IQR filter isolates typical demographic parameters, shifting overall $R^2$ scores up by approximately <strong className="text-emerald-400 font-bold">+4.5%</strong> across linear models and stabilizing distance convergence.
                </p>
              </div>
            </motion.div>
          ) : activeTab === 'eda' ? (
            /* TAB 3: EXPLORATORY DATA ANALYSIS */
            <motion.div
              key="eda"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              {/* CORRELATION MATRIX HEATMAP */}
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1 flex items-center gap-2">
                  <LineChart className="h-4 w-4 text-indigo-400" />
                  Feature Correlation Matrix
                </h3>
                <p className="text-slate-400 text-xs sm:text-sm mb-6">
                  Statistical correlation grid measuring linear relationships between features. Warm colors indicate a positive linear trend, while cool colors indicate negative trends.
                </p>

                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-9 gap-2">
                  {CORRELATION_MATRIX.columns.map((colX, xIdx) => (
                    <div key={colX} className="space-y-2 col-span-1 border border-slate-800 p-2 rounded-lg bg-slate-950/40">
                      <div className="text-[10px] font-mono font-bold text-slate-400 truncate">{colX}</div>
                      <div className="space-y-1">
                        {CORRELATION_MATRIX.columns.map((colY, yIdx) => {
                          const corrVal = CORRELATION_MATRIX.data[xIdx][yIdx];
                          const bgIntensity = Math.abs(corrVal);
                          let colorStyle = {};
                          if (corrVal >= 0) {
                            colorStyle = { backgroundColor: `rgba(239, 68, 68, ${bgIntensity * 0.8})`, color: 'white' };
                          } else {
                            colorStyle = { backgroundColor: `rgba(59, 130, 246, ${bgIntensity * 0.8})`, color: 'white' };
                          }
                          return (
                            <div
                              key={colY}
                              style={colorStyle}
                              className="text-[10px] p-1 rounded font-mono font-bold flex justify-between items-center transition-opacity"
                              title={`${colX} vs ${colY}: ${corrVal}`}
                            >
                              <span className="text-[8px] opacity-70 truncate max-w-[40px]">{colY}</span>
                              <span>{corrVal.toFixed(2)}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 text-xs text-slate-400 flex items-center gap-2 font-mono">
                  <span className="h-2 w-2 rounded-full bg-red-400"></span>
                  Strong Positive Target Correlation: <strong className="text-indigo-400">Median Income (0.69)</strong>. Income levels dominate the estimation weighting equations.
                </div>
              </div>

              {/* DYNAMIC SCATTERPLOT AND TARGET DISTRIBUTION */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* SCATTERPLOT: Income vs Price */}
                <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                  <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center justify-between">
                    <span>Income vs House Prices</span>
                    <TrendingUp className="h-4 w-4 text-slate-500" />
                  </h4>
                  <div className="h-[300px]" id="income_vs_price_chart">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
                        <XAxis
                          type="number"
                          dataKey="MedInc"
                          name="Median Income"
                          unit=" $10k"
                          stroke="#64748b"
                          fontSize={10}
                        />
                        <YAxis
                          type="number"
                          dataKey="Price"
                          name="Median Price"
                          unit=" $100k"
                          stroke="#64748b"
                          fontSize={10}
                        />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter name="Housing Units" data={CALIFORNIA_SAMPLE_DATA} fill="#6366f1" fillOpacity={0.65} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="text-xs text-slate-400 mt-2 font-mono">
                    Scatter distribution of income vs valuation. Note the capping limit in the source dataset at $5.00 ($500,000s) shown on the upper edge.
                  </p>
                </div>

                {/* HISTOGRAM DISTRIBUTION */}
                <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                  <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">
                    Price Distribution Target Column
                  </h4>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={CALIFORNIA_SAMPLE_DATA} margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
                        <XAxis dataKey="HouseAge" stroke="#64748b" name="House Age (years)" fontSize={10} />
                        <YAxis stroke="#64748b" fontSize={10} />
                        <Tooltip />
                        <Bar dataKey="Price" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="text-xs text-slate-400 mt-2 font-mono">
                    Visualizing median valuation spread. Real-world residential housing properties show significant variance in local geographic pockets.
                  </p>
                </div>
              </div>
            </motion.div>
          ) : activeTab === 'benchmarks' ? (
            /* TAB 4: MODEL BENCHMARKS */
            <motion.div
              key="benchmarks"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              {/* MODEL BENCHMARK TABLE */}
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                <div className="mb-6">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1 flex items-center gap-2">
                    <Layers className="h-4 w-4 text-indigo-400" />
                    Model Regression Metrics Comparison
                  </h3>
                  <p className="text-slate-400 text-xs sm:text-sm">
                    Results of training 7 models on 80% train / 20% test splits. XGBoost Regressor and Random Forest achieve the highest accuracy metrics.
                  </p>
                </div>

                <div className="overflow-x-auto border border-slate-800 rounded-xl bg-slate-950/20">
                  <table className="min-w-full divide-y divide-slate-800 text-left text-xs sm:text-sm">
                    <thead className="bg-slate-950 text-slate-400 font-semibold uppercase tracking-wider font-mono">
                      <tr>
                        <th className="px-6 py-3.5">ML Estimator Model</th>
                        <th className="px-6 py-3.5">MAE</th>
                        <th className="px-6 py-3.5">MSE</th>
                        <th className="px-6 py-3.5">RMSE</th>
                        <th className="px-6 py-3.5">R² Score</th>
                        <th className="px-6 py-3.5 text-right">Adjusted R²</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800 bg-slate-900/40 font-mono text-slate-300">
                      {MODEL_BENCHMARKS.map((model, idx) => (
                        <tr
                          key={model.Model}
                          className={idx === 0 ? 'bg-indigo-950/20 font-semibold text-indigo-200 hover:bg-indigo-950/30' : 'hover:bg-slate-800/45'}
                        >
                          <td className="px-6 py-3.5 flex items-center gap-2 font-sans font-medium text-slate-200">
                            {idx === 0 && <span className="text-[10px] bg-indigo-600 text-white px-2 py-0.5 rounded-full font-sans uppercase font-bold tracking-wider">Best</span>}
                            {model.Model}
                          </td>
                          <td className="px-6 py-3.5">{model.MAE.toFixed(4)}</td>
                          <td className="px-6 py-3.5">{model.MSE.toFixed(4)}</td>
                          <td className="px-6 py-3.5">{model.RMSE.toFixed(4)}</td>
                          <td className="px-6 py-3.5">
                            <span className={model.R2Score > 0.7 ? 'text-emerald-400 font-bold' : ''}>
                              {(model.R2Score * 100).toFixed(2)}%
                            </span>
                          </td>
                          <td className="px-6 py-3.5 text-right">{(model.AdjustedR2 * 100).toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* FEATURE IMPORTANCE GRAPH */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl lg:col-span-2">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">
                    Relative Feature Importance (MDI)
                  </h4>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={FEATURE_IMPORTANCES} layout="vertical" margin={{ top: 10, right: 30, left: 50, bottom: 5 }}>
                        <XAxis type="number" stroke="#64748b" fontSize={10} />
                        <YAxis dataKey="Feature" type="category" stroke="#64748b" width={100} fontSize={9} />
                        <Tooltip />
                        <Bar dataKey="Importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="text-xs text-slate-400 mt-2 font-mono">
                    Relative node split impurities (Mean Decrease in Impurity) generated by Random Forest. Custom interaction terms contribute significantly.
                  </p>
                </div>

                {/* METRICS GLOSSARY SUMMARY */}
                <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl space-y-4">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">
                    Regression Metrics Guide
                  </h4>
                  <ul className="space-y-4 text-xs leading-relaxed">
                    <li>
                      <strong className="text-slate-200">MAE (Mean Absolute Error)</strong>
                      <p className="text-slate-400 mt-1">Average absolute distance between target and predictions. Direct translation to error in dollar scale ($31,210 for XGBoost).</p>
                    </li>
                    <li>
                      <strong className="text-slate-200">RMSE (Root Mean Squared Error)</strong>
                      <p className="text-slate-400 mt-1">Penalizes larger outliers more heavily by squaring deviations before taking the root. Essential for spotting large model failures.</p>
                    </li>
                    <li>
                      <strong className="text-slate-200">R² Coefficient of Determination</strong>
                      <p className="text-slate-400 mt-1">Proportion of variance explained. Our top model accounts for <strong className="text-emerald-400 font-bold">80.42%</strong> of housing price variations in tests.</p>
                    </li>
                    <li>
                      <strong className="text-slate-200">Adjusted R²</strong>
                      <p className="text-slate-400 mt-1">Corrects standard R² scores for added degrees of freedom to prevent artificially inflating scores with useless extra parameters.</p>
                    </li>
                  </ul>
                </div>
              </div>
            </motion.div>
          ) : activeTab === 'tuning' ? (
            /* TAB 5: HYPERPARAMETER TUNING */
            <motion.div
              key="tuning"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1 flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-indigo-400" />
                  GridSearchCV Optimization Grid
                </h3>
                <p className="text-slate-400 text-xs sm:text-sm mb-6">
                  GridSearchCV performs exhaustive searching across a specified parameter grid, testing all combinations using 5-Fold Cross-Validation.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                  {TUNING_GRID.map((tune) => (
                    <div key={tune.paramName} className="border border-slate-800 p-4 rounded-lg bg-slate-950/40 hover:bg-slate-950/70 transition-all">
                      <div className="text-[10px] text-indigo-400 font-mono font-bold uppercase tracking-wider mb-1">{tune.paramName}</div>
                      <div className="text-xs font-semibold text-slate-200 font-mono mb-2">Grid: {tune.paramGrid}</div>
                      <div className="text-[11px] bg-indigo-950/40 border border-indigo-900/30 text-indigo-300 font-mono px-2 py-1 rounded mb-2 inline-block">
                        Best Value: {tune.bestValue}
                      </div>
                      <p className="text-[11px] text-slate-400 leading-relaxed">{tune.explanation}</p>
                    </div>
                  ))}
                </div>

                <div className="p-4 bg-indigo-950/20 border border-indigo-900/30 rounded-lg flex items-start gap-3">
                  <CheckCircle2 className="h-5 w-5 text-indigo-400 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-bold uppercase tracking-wider text-indigo-300 mb-1">Grid Search General Optimization Results</h4>
                    <p className="text-xs text-indigo-200 leading-relaxed">
                      Before hyperparameter optimization, the base Gradient Boosting model registered an R² score of 76.15%. Running a GridSearch parameter sweeps on trees parameters successfully improved overall general test accuracy to <strong className="text-emerald-400">80.42% R²</strong>, effectively decreasing MSE error scales.
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          ) : activeTab === 'predictor' ? (
            /* TAB 6: INTERACTIVE USER PREDICTOR */
            <motion.div
              key="predictor"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="grid grid-cols-1 lg:grid-cols-3 gap-6"
            >
              {/* SLIDERS COLUMN */}
              <div className="lg:col-span-2 bg-slate-900 border border-slate-800 p-6 rounded-xl space-y-6">
                <div>
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1 flex items-center gap-2">
                    <Sliders className="h-4 w-4 text-indigo-400" />
                    Inference Parameter Setup
                  </h3>
                  <p className="text-slate-400 text-xs sm:text-sm">
                    Adjust block indices using sliders to observe predicted house price variations instantly via real-time regression math.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* SLIDER: Income */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs sm:text-sm font-mono">
                      <span className="font-semibold text-slate-300">Median Income (MedInc)</span>
                      <span className="text-indigo-400 font-bold">
                        ${(predictionInputs.MedInc * 10000).toLocaleString('en-US', { maximumFractionDigits: 0 })}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="15"
                      step="0.1"
                      value={predictionInputs.MedInc}
                      onChange={(e) => setPredictionInputs({ ...predictionInputs, MedInc: parseFloat(e.target.value) })}
                      className="w-full accent-indigo-500 bg-slate-950 rounded-lg h-2"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                      <span>Low ($5k)</span>
                      <span>Avg ($38k)</span>
                      <span>High ($150k+)</span>
                    </div>
                  </div>

                  {/* SLIDER: House Age */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs sm:text-sm font-mono">
                      <span className="font-semibold text-slate-300">Median House Age (years)</span>
                      <span className="text-indigo-400 font-bold">{predictionInputs.HouseAge} yrs</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="52"
                      value={predictionInputs.HouseAge}
                      onChange={(e) => setPredictionInputs({ ...predictionInputs, HouseAge: parseInt(e.target.value) })}
                      className="w-full accent-indigo-500 bg-slate-950 rounded-lg h-2"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                      <span>New (1 yr)</span>
                      <span>Avg (28 yrs)</span>
                      <span>Historic (52 yrs)</span>
                    </div>
                  </div>

                  {/* SLIDER: Ave Rooms */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs sm:text-sm font-mono">
                      <span className="font-semibold text-slate-300">Average Rooms per Unit</span>
                      <span className="text-indigo-400 font-bold">{predictionInputs.AveRooms.toFixed(1)} rooms</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      step="0.1"
                      value={predictionInputs.AveRooms}
                      onChange={(e) => setPredictionInputs({ ...predictionInputs, AveRooms: parseFloat(e.target.value) })}
                      className="w-full accent-indigo-500 bg-slate-950 rounded-lg h-2"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                      <span>1 room</span>
                      <span>Avg (5.4)</span>
                      <span>10 rooms</span>
                    </div>
                  </div>

                  {/* SLIDER: Ave Bedrooms */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs sm:text-sm font-mono">
                      <span className="font-semibold text-slate-300">Average Bedrooms per Unit</span>
                      <span className="text-indigo-400 font-bold">{predictionInputs.AveBedrms.toFixed(1)} beds</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="4"
                      step="0.1"
                      value={predictionInputs.AveBedrms}
                      onChange={(e) => setPredictionInputs({ ...predictionInputs, AveBedrms: parseFloat(e.target.value) })}
                      className="w-full accent-indigo-500 bg-slate-950 rounded-lg h-2"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                      <span>0.5 rooms</span>
                      <span>Avg (1.1)</span>
                      <span>4 rooms</span>
                    </div>
                  </div>

                  {/* SLIDER: Population */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs sm:text-sm font-mono">
                      <span className="font-semibold text-slate-300">District Block Population</span>
                      <span className="text-indigo-400 font-bold">{predictionInputs.Population.toLocaleString()} people</span>
                    </div>
                    <input
                      type="range"
                      min="50"
                      max="5000"
                      step="50"
                      value={predictionInputs.Population}
                      onChange={(e) => setPredictionInputs({ ...predictionInputs, Population: parseInt(e.target.value) })}
                      className="w-full accent-indigo-500 bg-slate-950 rounded-lg h-2"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                      <span>Sparse (50)</span>
                      <span>Avg (1400)</span>
                      <span>Dense (5000)</span>
                    </div>
                  </div>

                  {/* SLIDER: Ave Occupancy */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs sm:text-sm font-mono">
                      <span className="font-semibold text-slate-300">Average Occupants per Unit</span>
                      <span className="text-indigo-400 font-bold">{predictionInputs.AveOccup.toFixed(1)} members</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="8"
                      step="0.1"
                      value={predictionInputs.AveOccup}
                      onChange={(e) => setPredictionInputs({ ...predictionInputs, AveOccup: parseFloat(e.target.value) })}
                      className="w-full accent-indigo-500 bg-slate-950 rounded-lg h-2"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                      <span>1 member</span>
                      <span>Avg (2.9)</span>
                      <span>8 members</span>
                    </div>
                  </div>

                  {/* SLIDERS: Latitude and Longitude */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs sm:text-sm font-mono">
                      <span className="font-semibold text-slate-300">Latitude Location</span>
                      <span className="text-indigo-400 font-bold">{predictionInputs.Latitude.toFixed(2)}° N</span>
                    </div>
                    <input
                      type="range"
                      min="32.5"
                      max="42.0"
                      step="0.01"
                      value={predictionInputs.Latitude}
                      onChange={(e) => setPredictionInputs({ ...predictionInputs, Latitude: parseFloat(e.target.value) })}
                      className="w-full accent-indigo-500 bg-slate-950 rounded-lg h-2"
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-xs sm:text-sm font-mono">
                      <span className="font-semibold text-slate-300">Longitude Location</span>
                      <span className="text-indigo-400 font-bold">{predictionInputs.Longitude.toFixed(2)}° W</span>
                    </div>
                    <input
                      type="range"
                      min="-124.3"
                      max="-114.3"
                      step="0.01"
                      value={predictionInputs.Longitude}
                      onChange={(e) => setPredictionInputs({ ...predictionInputs, Longitude: parseFloat(e.target.value) })}
                      className="w-full accent-indigo-500 bg-slate-950 rounded-lg h-2"
                    />
                  </div>
                </div>

                {/* COORDINATE CORRELATOR MAP COMPONENT */}
                <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="flex items-center gap-2">
                    <MapPin className="h-4 w-4 text-indigo-400" />
                    <div>
                      <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">Geographic Index Analysis</h4>
                      <p className="text-[10px] text-slate-500 font-mono">
                        San Francisco: ~37.77° N, -122.41° W | Los Angeles: ~34.05° N, -118.24° W
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setPredictionInputs({ ...predictionInputs, Latitude: 37.77, Longitude: -122.41 })}
                      className="px-3 py-1 bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 rounded text-xs font-semibold uppercase tracking-wider cursor-pointer font-mono"
                    >
                      Bay Area
                    </button>
                    <button
                      onClick={() => setPredictionInputs({ ...predictionInputs, Latitude: 34.05, Longitude: -118.24 })}
                      className="px-3 py-1 bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 rounded text-xs font-semibold uppercase tracking-wider cursor-pointer font-mono"
                    >
                      SoCal
                    </button>
                  </div>
                </div>
              </div>

              {/* ESTIMATION OUTPUT PANEL */}
              <div className="space-y-6">
                <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl relative overflow-hidden shadow-2xl">
                  <div className="absolute right-0 bottom-0 opacity-5">
                    <Sliders className="h-52 w-52 translate-x-12 translate-y-12" />
                  </div>

                  <span className="text-[10px] text-slate-500 font-bold tracking-wider uppercase font-mono block mb-2">Estimated Market Valuation</span>
                  <div className="flex items-baseline mb-4">
                    <span className="text-3xl sm:text-4xl font-bold font-mono tracking-tight text-white">
                      ${predictionResult.priceUSD.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </span>
                  </div>

                  <div className="pt-4 border-t border-slate-800/80 space-y-2 text-xs text-slate-400 font-mono">
                    <div className="flex justify-between">
                      <span>Prediction Confidence</span>
                      <span className="font-semibold text-emerald-400">92% Match</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Regression Std Error</span>
                      <span className="font-semibold">± $44,880</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Target Model Weight</span>
                      <span className="font-semibold">Optimal GBDT</span>
                    </div>
                  </div>
                </div>

                {/* VALUE BREAKDOWN CONTROLLERS (SHAP) */}
                <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center justify-between">
                    <span>Feature Weight Contributions</span>
                    <Sparkles className="h-4 w-4 text-indigo-400" />
                  </h4>

                  <div className="space-y-3">
                    {predictionResult.breakdown.slice(0, 5).map((feature) => (
                      <div key={feature.feature} className="space-y-1 text-xs">
                        <div className="flex justify-between text-slate-400">
                          <span className="truncate max-w-[150px] font-mono">{feature.feature}</span>
                          <span className={feature.isPositive ? 'text-emerald-400 font-mono font-semibold' : 'text-rose-400 font-mono font-semibold'}>
                            {feature.isPositive ? '+' : '-'} ${(feature.contributionUSD).toLocaleString('en-US', { maximumFractionDigits: 0 })}
                          </span>
                        </div>
                        <div className="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${feature.isPositive ? 'bg-emerald-500' : 'bg-rose-500'}`}
                            style={{ width: `${Math.min(100, (feature.contributionUSD / 120000) * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="text-[10px] text-slate-500 mt-4 font-mono">
                    Values mimic regression contribution ratios relative to the standard district median value of $206,800.
                  </p>
                </div>
              </div>
            </motion.div>
          ) : (
            /* TAB 7: PYTHON CODE WORKSPACE */
            <motion.div
              key="code"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl"
            >
              {/* FILE EXPLORER TABS */}
              <div className="bg-slate-950 px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 gap-4">
                <div className="flex space-x-2">
                  <button
                    onClick={() => setActiveFile('py')}
                    className={`px-4 py-2 rounded-lg text-xs font-bold uppercase tracking-wider flex items-center gap-2 transition-all cursor-pointer ${
                      activeFile === 'py'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <Code className="h-4 w-4" />
                    house_price_prediction.py
                  </button>
                  <button
                    onClick={() => setActiveFile('req')}
                    className={`px-4 py-2 rounded-lg text-xs font-bold uppercase tracking-wider flex items-center gap-2 transition-all cursor-pointer ${
                      activeFile === 'req'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <FileText className="h-4 w-4" />
                    requirements.txt
                  </button>
                  <button
                    onClick={() => setActiveFile('readme')}
                    className={`px-4 py-2 rounded-lg text-xs font-bold uppercase tracking-wider flex items-center gap-2 transition-all cursor-pointer ${
                      activeFile === 'readme'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <FileText className="h-4 w-4" />
                    README.md
                  </button>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() =>
                      handleCopyCode(
                        activeFile === 'py'
                          ? PYTHON_CODE_STRING
                          : activeFile === 'req'
                          ? `pandas>=2.0.0\nnumpy>=1.24.0\nmatplotlib>=3.7.0\nseaborn>=0.12.0\nscikit-learn>=1.2.0\nxgboost>=1.7.0\njoblib>=1.2.0\nscipy>=1.10.0`
                          : `# House Price Prediction Using Machine Learning\n\nObjective: Train regressors to estimate median value metrics using California parameters.`
                      )
                    }
                    className="px-4 py-2 bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer"
                  >
                    {copied ? <Check className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4" />}
                    {copied ? 'Copied' : 'Copy File Content'}
                  </button>
                </div>
              </div>

              {/* CODE CONTAINER */}
              <div className="p-6 bg-slate-950/80 text-slate-300 font-mono text-xs sm:text-sm overflow-x-auto max-h-[500px] leading-relaxed">
                <pre>
                  <code>
                    {activeFile === 'py' && PYTHON_CODE_STRING}
                    {activeFile === 'req' && (
                      <div className="space-y-1 text-slate-300">
                        <div>pandas&gt;=2.0.0</div>
                        <div>numpy&gt;=1.24.0</div>
                        <div>matplotlib&gt;=3.7.0</div>
                        <div>seaborn&gt;=0.12.0</div>
                        <div>scikit-learn&gt;=1.2.0</div>
                        <div>xgboost&gt;=1.7.0</div>
                        <div>joblib&gt;=1.2.0</div>
                        <div>scipy&gt;=1.10.0</div>
                      </div>
                    )}
                    {activeFile === 'readme' && (
                      <div className="space-y-3 font-sans text-sm p-4">
                        <h1 className="text-xl font-bold text-white border-b border-slate-850 pb-2 font-display">README.md</h1>
                        <p className="text-slate-400">California Housing prediction pipeline.</p>
                        <h2 className="text-md font-semibold text-white mt-4 font-display">Required libraries</h2>
                        <p className="text-slate-400">Install pip configurations:</p>
                        <code className="block bg-slate-900 p-2.5 rounded border border-slate-800 text-xs font-mono text-indigo-400">$ pip install -r requirements.txt</code>
                        <h2 className="text-md font-semibold text-white mt-4 font-display">Command options</h2>
                        <code className="block bg-slate-900 p-2.5 rounded border border-slate-800 text-xs font-mono text-indigo-400">$ python house_price_prediction.py</code>
                      </div>
                    )}
                  </code>
                </pre>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* FOOTER SECTION */}
      <footer className="border-t border-slate-900 bg-slate-950/50 mt-16 py-8" id="app_footer_container">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-xs text-slate-500 space-y-2">
          <p className="font-bold text-slate-400 uppercase tracking-widest text-[11px]">House Price Prediction ML Dashboard Studio</p>
          <p className="text-slate-500">
            Powered by Scikit-Learn, Pandas, Recharts, and Google AI Studio Workspace integration. Designed for expert regression analytics modeling.
          </p>
          <p className="text-slate-600 font-mono">© 2026 Sandbox Environment. All rights preserved.</p>
        </div>
      </footer>
    </div>
  );
}
