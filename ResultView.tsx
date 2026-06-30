import { ArrowLeft, CheckCircle2, AlertTriangle, Play, HelpCircle, User, Activity } from 'lucide-react';
import { PredictionResult, CustomerRecord } from '../types';

interface ResultViewProps {
  result: PredictionResult;
  customer: CustomerRecord;
  onReset: () => void;
}

export default function ResultView({ result, customer, onReset }: ResultViewProps) {
  const probPercentage = Math.round(result.churn_probability * 100);

  const riskStyles = {
    High: {
      border: 'border-rose-200 bg-rose-50/40 text-rose-800',
      badge: 'bg-rose-500 text-white',
      banner: 'bg-rose-50 text-rose-700 border-rose-200',
      iconColor: 'text-rose-500',
      text: 'Extremely high risk of churn. Immediate action advised.',
    },
    Medium: {
      border: 'border-amber-200 bg-amber-50/40 text-amber-800',
      badge: 'bg-amber-500 text-white',
      banner: 'bg-amber-50 text-amber-700 border-amber-200',
      iconColor: 'text-amber-500',
      text: 'Moderate risk of churn. Targeted outreach recommended.',
    },
    Low: {
      border: 'border-emerald-200 bg-emerald-50/40 text-emerald-800',
      badge: 'bg-emerald-500 text-white',
      banner: 'bg-emerald-50 text-emerald-700 border-emerald-200',
      iconColor: 'text-emerald-500',
      text: 'Healthy retention index. Account is stable.',
    },
  }[result.risk_level];

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6" id="prediction-result-root">
      {/* Upper Navigation Action */}
      <div className="flex items-center justify-between border-b border-slate-200 pb-5">
        <button
          onClick={onReset}
          className="inline-flex items-center space-x-1.5 text-xs font-bold text-slate-500 hover:text-slate-800 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Go Back / Reset Form</span>
        </button>
        <span className="text-[10px] font-bold font-mono text-slate-400 bg-slate-50 px-2.5 py-1 rounded-full border border-slate-200">
          Engine: {result.model_used}
        </span>
      </div>

      {/* Probability Gauge and Severity Panel */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-center">
        {/* Risk Gauge */}
        <div className="md:col-span-5 flex flex-col items-center justify-center p-4 border border-slate-200 rounded-xl bg-slate-50/40">
          <div className="relative w-44 h-44 flex items-center justify-center">
            {/* SVG Arc Gauge */}
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="42"
                stroke="#e2e8f0"
                strokeWidth="7"
                fill="none"
              />
              <circle
                cx="50"
                cy="50"
                r="42"
                stroke={result.risk_level === 'High' ? '#f43f5e' : result.risk_level === 'Medium' ? '#f59e0b' : '#10b981'}
                strokeWidth="7"
                fill="none"
                strokeDasharray="263.89"
                strokeDashoffset={263.89 - (263.89 * probPercentage) / 100}
                strokeLinecap="round"
                className="transition-all duration-1000 ease-out"
              />
            </svg>
            <div className="absolute text-center">
              <span className="text-4xl font-extrabold text-slate-800">{probPercentage}%</span>
              <p className="text-[10px] font-bold text-slate-400 mt-0.5 uppercase tracking-wider">Churn Risk</p>
            </div>
          </div>
          <span className={`inline-flex items-center px-3 py-1 rounded-full text-[10px] font-bold uppercase mt-4 tracking-wider ${riskStyles.badge}`}>
            {result.risk_level} Risk Level
          </span>
        </div>

        {/* Severity Banner and Quick Details */}
        <div className="md:col-span-7 space-y-4">
          <div className={`border rounded-xl p-5 ${riskStyles.border}`}>
            <h3 className="text-xs font-bold uppercase tracking-wider flex items-center gap-2">
              {result.risk_level === 'Low' ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
              ) : (
                <AlertTriangle className="w-5 h-5 shrink-0" />
              )}
              Retention Status: {result.prediction === 'Churn' ? 'At Risk' : 'Secure'}
            </h3>
            <p className="text-xs mt-1.5 opacity-90">{riskStyles.text}</p>
          </div>

          <div className="bg-slate-50 rounded-xl p-5 border border-slate-200 grid grid-cols-2 gap-4 text-[11px]">
            <div>
              <span className="text-slate-400 font-bold uppercase tracking-wider block text-[9px]">Contract Term</span>
              <span className="text-slate-800 font-bold block mt-0.5">{customer.Contract}</span>
            </div>
            <div>
              <span className="text-slate-400 font-bold uppercase tracking-wider block text-[9px]">Tenure Loyalty</span>
              <span className="text-slate-800 font-bold block mt-0.5">{customer.tenure} Months</span>
            </div>
            <div>
              <span className="text-slate-400 font-bold uppercase tracking-wider block text-[9px]">Monthly charges</span>
              <span className="text-slate-800 font-bold block mt-0.5">${customer.MonthlyCharges.toFixed(2)}</span>
            </div>
            <div>
              <span className="text-slate-400 font-bold uppercase tracking-wider block text-[9px]">Payment Method</span>
              <span className="text-slate-800 font-bold block mt-0.5 truncate">{customer.PaymentMethod}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Local Explainability Factors */}
      <div className="space-y-4 pt-2">
        <h3 className="text-xs font-bold text-slate-800 flex items-center gap-2 border-b border-slate-200 pb-2">
          <Activity className="w-4 h-4 text-indigo-600" />
          Explainable AI (XAI) Risk Drivers
        </h3>
        <p className="text-xs text-slate-500 mt-1">
          Simulated Shapley/LIME values outlining exactly which features increased or decreased this subscriber's risk.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {result.explainability_factors.map((factor, index) => {
            const impactStyle = {
              'High Risk': 'bg-rose-50 text-rose-700 border-rose-100',
              'Moderate Risk': 'bg-amber-50 text-amber-700 border-amber-100',
              'Low Risk': 'bg-yellow-50 text-yellow-700 border-yellow-100',
              'Healthy (No Risk Drivers)': 'bg-emerald-50 text-emerald-700 border-emerald-100',
            }[factor.impact];

            return (
              <div
                key={index}
                className="border border-slate-200 rounded-xl p-4 bg-slate-50/20 flex flex-col justify-between"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wide">
                      {factor.feature}
                    </h4>
                    <span className="text-xs font-bold text-slate-800 mt-1 block">{factor.value}</span>
                  </div>
                  <span className={`px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider border ${impactStyle}`}>
                    {factor.impact}
                  </span>
                </div>
                <p className="text-xs text-slate-500 mt-3.5 bg-white p-2.5 rounded border border-slate-200 leading-relaxed">
                  {factor.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Recommended Retention Playbooks */}
      <div className="space-y-4 pt-2">
        <h3 className="text-xs font-bold text-slate-800 flex items-center gap-2 border-b border-slate-200 pb-2">
          <Play className="w-4 h-4 text-indigo-600 fill-indigo-600" />
          Customer Success & Retention Playbook
        </h3>
        <p className="text-xs text-slate-500 mt-1">
          Prescriptive analytics outlining targeted marketing recommendations to mitigate churn risk.
        </p>

        <div className="space-y-3">
          {result.retention_playbook.map((play, index) => (
            <div
              key={index}
              className="flex items-start space-x-3 bg-indigo-50/20 border border-indigo-50/60 rounded-xl p-3.5"
            >
              <div className="bg-indigo-100 text-indigo-700 rounded-lg p-2 font-bold text-[10px] shrink-0">
                0{index + 1}
              </div>
              <p className="text-xs text-slate-600 font-semibold leading-relaxed self-center">{play}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
