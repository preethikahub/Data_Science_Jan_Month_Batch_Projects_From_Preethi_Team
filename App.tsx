import { useState } from 'react';
import {
  FileSpreadsheet,
  Sliders,
  BookOpen,
  Terminal,
  Activity,
  Heart,
  Briefcase,
} from 'lucide-react';

// Import subcomponents
import Dashboard from './components/Dashboard';
import PredictionForm from './components/PredictionForm';
import ResultView from './components/ResultView';
import AboutView from './components/AboutView';
import CodeHub from './components/CodeHub';

// Types and predictor
import { CustomerRecord, PredictionResult } from './types';
import { calculateChurnProbability } from './utils';

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'predict' | 'about' | 'code'>('dashboard');
  const [activePrediction, setActivePrediction] = useState<{
    result: PredictionResult;
    customer: CustomerRecord;
  } | null>(null);

  const handlePredictSubmit = (customerData: CustomerRecord) => {
    const output = calculateChurnProbability(customerData);
    setActivePrediction({
      result: output,
      customer: customerData,
    });
  };

  const handlePredictReset = () => {
    setActivePrediction(null);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans flex flex-col antialiased">
      {/* Upper Navigation Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            {/* Branding Logo */}
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-indigo-600 text-white rounded-lg flex items-center justify-center shadow-md">
                <Activity className="w-5.5 h-5.5" />
              </div>
              <div>
                <h1 className="text-base font-bold text-slate-900 tracking-tight leading-none">
                  ChurnShield AI
                </h1>
                <p className="text-[10px] text-slate-500 font-medium mt-1">
                  Production ML Dashboard • v2.1.0
                </p>
              </div>
            </div>

            {/* Model deployment badge & Desktop navigation */}
            <div className="flex items-center space-x-6">
              <div className="hidden sm:flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="text-[10px] font-bold text-emerald-600 uppercase tracking-wider">Model Deployed</span>
              </div>
              
              <div className="hidden sm:block h-8 w-px bg-slate-200"></div>

              {/* Desktop Navigation Link Tabs */}
              <nav className="hidden md:flex space-x-1 bg-slate-100 p-1.5 rounded-xl border border-slate-200/50">
                <button
                  onClick={() => setActiveTab('dashboard')}
                  className={`inline-flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
                    activeTab === 'dashboard'
                      ? 'bg-white text-indigo-600 shadow-sm'
                      : 'text-slate-500 hover:text-slate-900'
                  }`}
                >
                  <FileSpreadsheet className="w-4 h-4" />
                  <span>Dashboard</span>
                </button>
                <button
                  onClick={() => {
                    setActiveTab('predict');
                    handlePredictReset();
                  }}
                  className={`inline-flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
                    activeTab === 'predict'
                      ? 'bg-white text-indigo-600 shadow-sm'
                      : 'text-slate-500 hover:text-slate-900'
                  }`}
                >
                  <Sliders className="w-4 h-4" />
                  <span>Predict Churn</span>
                </button>
                <button
                  onClick={() => setActiveTab('about')}
                  className={`inline-flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
                    activeTab === 'about'
                      ? 'bg-white text-indigo-600 shadow-sm'
                      : 'text-slate-500 hover:text-slate-900'
                  }`}
                >
                  <BookOpen className="w-4 h-4" />
                  <span>Pipeline</span>
                </button>
                <button
                  onClick={() => setActiveTab('code')}
                  className={`inline-flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
                    activeTab === 'code'
                      ? 'bg-white text-indigo-600 shadow-sm'
                      : 'text-slate-500 hover:text-slate-900'
                  }`}
                >
                  <Terminal className="w-4 h-4" />
                  <span>Python Workspace</span>
                </button>
              </nav>
            </div>
          </div>
        </div>
      </header>

      {/* Main Panel Area */}
      <main className="flex-grow max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full flex flex-col">
        {/* Mobile Tab Fallback Header bar */}
        <div className="flex md:hidden overflow-x-auto space-x-2 pb-4 mb-2 scrollbar-none">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`px-4 py-2.5 rounded-lg text-xs font-bold shrink-0 border ${
              activeTab === 'dashboard'
                ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
            }`}
          >
            Dashboard
          </button>
          <button
            onClick={() => {
              setActiveTab('predict');
              handlePredictReset();
            }}
            className={`px-4 py-2.5 rounded-lg text-xs font-bold shrink-0 border ${
              activeTab === 'predict'
                ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
            }`}
          >
            Predict Churn
          </button>
          <button
            onClick={() => setActiveTab('about')}
            className={`px-4 py-2.5 rounded-lg text-xs font-bold shrink-0 border ${
              activeTab === 'about'
                ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
            }`}
          >
            Pipeline
          </button>
          <button
            onClick={() => setActiveTab('code')}
            className={`px-4 py-2.5 rounded-lg text-xs font-bold shrink-0 border ${
              activeTab === 'code'
                ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
            }`}
          >
            Python Workspace
          </button>
        </div>

        {/* Tab Routing Router */}
        <div className="flex-1">
          {activeTab === 'dashboard' && <Dashboard />}

          {activeTab === 'predict' && (
            <div>
              {activePrediction ? (
                <ResultView
                  result={activePrediction.result}
                  customer={activePrediction.customer}
                  onReset={handlePredictReset}
                />
              ) : (
                <PredictionForm onSubmit={handlePredictSubmit} />
              )}
            </div>
          )}

          {activeTab === 'about' && <AboutView />}

          {activeTab === 'code' && <CodeHub />}
        </div>
      </main>

      {/* Footer Branding Area */}
      <footer className="h-12 bg-white border-t border-slate-200 flex items-center justify-between shrink-0 px-4 sm:px-8 mt-12">
        <div className="flex gap-4 text-[10px] text-slate-500 font-bold uppercase tracking-widest">
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
            Session: Active
          </span>
          <span className="text-slate-300">|</span>
          <span>Runtime: 42ms</span>
          <span className="text-slate-300">|</span>
          <span>Engine: Scikit-learn + XGBoost</span>
        </div>
        <div className="hidden sm:block text-[10px] text-slate-400 font-medium">
          Developed by AI Software Architects • Professional Data Suite © 2026
        </div>
      </footer>
    </div>
  );
}
