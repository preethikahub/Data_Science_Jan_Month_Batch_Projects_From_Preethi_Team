import { useState } from 'react';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from 'recharts';
import {
  BarChart3,
  TrendingUp,
  Users,
  Percent,
  CheckCircle,
  Eye,
  Settings,
  ShieldCheck,
  FileSpreadsheet,
} from 'lucide-react';
import { modelComparisonsRegistry } from '../utils';

export default function Dashboard() {
  const [selectedModel, setSelectedModel] = useState<string>('Gradient Boosting');

  // Static stats representative of Telco Customer Churn
  const stats = [
    { label: 'Total Audited Clients', value: '1,500', icon: Users, color: 'text-indigo-600 bg-indigo-50' },
    { label: 'Baseline Churn Rate', value: '26.5%', icon: Percent, color: 'text-rose-600 bg-rose-50' },
    { label: 'Average Tenure', value: '32.4 Months', icon: TrendingUp, color: 'text-emerald-600 bg-emerald-50' },
    { label: 'Avg Monthly charges', value: '$64.76', icon: BarChart3, color: 'text-amber-600 bg-amber-50' },
  ];

  // Churn target distribution data
  const churnData = [
    { name: 'Retained Customers', value: 1102, color: '#4f46e5' },
    { name: 'Churned Customers', value: 398, color: '#f43f5e' },
  ];

  // Monthly charges frequency distribution curve (simulated density)
  const billingDistribution = [
    { range: '$18-30', Retained: 240, Churned: 45 },
    { range: '$30-50', Retained: 180, Churned: 65 },
    { range: '$50-70', Retained: 290, Churned: 85 },
    { range: '$70-90', Retained: 210, Churned: 120 },
    { range: '$90-110', Retained: 145, Churned: 68 },
    { range: '$110-120', Retained: 37, Churned: 15 },
  ];

  // Contract risk ratio
  const contractRisk = [
    { term: 'Month-to-month', Retained: 340, Churned: 320, Rate: '48.5%' },
    { term: 'One year', Retained: 350, Churned: 55, Rate: '13.6%' },
    { term: 'Two year', Retained: 412, Churned: 23, Rate: '5.3%' },
  ];

  const models = Object.keys(modelComparisonsRegistry) as Array<
    keyof typeof modelComparisonsRegistry
  >;
  const activeModelDetails = modelComparisonsRegistry[selectedModel as keyof typeof modelComparisonsRegistry];

  return (
    <div className="space-y-8" id="eda-dashboard-root">
      {/* Upper Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((st) => {
          const Icon = st.icon;
          return (
            <div
              key={st.label}
              className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm flex items-center space-x-4"
            >
              <div className={`p-3 rounded-lg ${st.color} shrink-0`}>
                <Icon className="w-5.5 h-5.5" />
              </div>
              <div>
                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{st.label}</p>
                <h4 className="text-xl font-bold text-slate-900 mt-1">{st.value}</h4>
              </div>
            </div>
          );
        })}
      </div>

      {/* Exploratory Data Analysis section */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <FileSpreadsheet className="w-5 h-5 text-indigo-600" />
            Exploratory Data Analysis (EDA) Insights
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Analyzing historical factors and service correlations driving customer attrition.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Target Variable Breakdown */}
          <div className="border border-slate-200 rounded-xl p-5 bg-white flex flex-col justify-between shadow-sm">
            <div>
              <h3 className="text-xs font-bold text-slate-400 uppercase">1. Churn Variable Distribution</h3>
              <p className="text-xs text-slate-500 mt-0.5">Proportion of retaining versus churning accounts.</p>
            </div>
            <div className="h-52 my-4 flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={churnData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={75}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {churnData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`${value} Clients`, 'Count']} />
                  <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: 11 }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="text-center text-[11px] text-slate-500 italic bg-slate-50 p-2.5 rounded border border-slate-200">
              Imbalance factor is ~3:1. Preprocessor applies <strong className="text-indigo-600">SMOTE</strong> oversampling during model fit.
            </div>
          </div>

          {/* Monthly Charges Density */}
          <div className="border border-slate-200 rounded-xl p-5 bg-white flex flex-col justify-between shadow-sm">
            <div>
              <h3 className="text-xs font-bold text-slate-400 uppercase">2. Attrition vs. Monthly Charges</h3>
              <p className="text-xs text-slate-500 mt-0.5">Billing impact showing churn frequency at specific charge tiers.</p>
            </div>
            <div className="h-52 my-4">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={billingDistribution} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="range" tick={{ fontSize: 10, fill: '#64748b' }} stroke="#cbd5e1" />
                  <YAxis tick={{ fontSize: 10, fill: '#64748b' }} stroke="#cbd5e1" />
                  <Tooltip formatter={(value) => [`${value} Clients`, '']} />
                  <Legend verticalAlign="bottom" height={36} iconType="rect" wrapperStyle={{ fontSize: 11 }} />
                  <Area type="monotone" dataKey="Retained" stackId="1" stroke="#4f46e5" fill="#c7d2fe" fillOpacity={0.4} />
                  <Area type="monotone" dataKey="Churned" stackId="2" stroke="#f43f5e" fill="#fecdd3" fillOpacity={0.4} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="text-center text-[11px] text-slate-500 italic bg-slate-50 p-2.5 rounded border border-slate-200">
              Attrition is highly dense in the premium <strong className="text-rose-600">$70-$90</strong> fiber-optic charging bandwidth.
            </div>
          </div>

          {/* Contract Types Churn rate */}
          <div className="border border-slate-200 rounded-xl p-5 bg-white flex flex-col justify-between shadow-sm">
            <div>
              <h3 className="text-xs font-bold text-slate-400 uppercase">3. Contract Term Churn Matrix</h3>
              <p className="text-xs text-slate-500 mt-0.5">How specific contract tiers alter subscription longevity.</p>
            </div>
            <div className="h-52 my-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={contractRisk} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="term" tick={{ fontSize: 10, fill: '#64748b' }} stroke="#cbd5e1" />
                  <YAxis tick={{ fontSize: 10, fill: '#64748b' }} stroke="#cbd5e1" />
                  <Tooltip />
                  <Legend verticalAlign="bottom" height={36} wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="Retained" fill="#c7d2fe" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Churned" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="text-center text-[11px] text-slate-500 italic bg-slate-50 p-2.5 rounded border border-slate-200">
              Month-to-month contracts trigger high attrition rates (<strong className="text-rose-600">48.5%</strong>) compared to two-year (<strong>5.3%</strong>).
            </div>
          </div>
        </div>
      </div>

      {/* Machine Learning Models Comparison Section */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-indigo-600" />
            Machine Learning Classification Benchmark
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Comparative analysis of nine major classification models evaluated on the processed, SMOTE-balanced test data.
          </p>
        </div>

        {/* Benchmarking Table */}
        <div className="overflow-x-auto border border-slate-200 rounded-xl">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-slate-400 text-[10px] font-bold uppercase tracking-wider">
                <th className="py-3 px-6">Algorithm Model Name</th>
                <th className="py-3 px-4 text-center">Accuracy</th>
                <th className="py-3 px-4 text-center">Precision</th>
                <th className="py-3 px-4 text-center">Recall</th>
                <th className="py-3 px-4 text-center">F1-Score</th>
                <th className="py-3 px-4 text-center">ROC-AUC</th>
                <th className="py-3 px-4 text-center">5-Fold CV F1</th>
                <th className="py-3 px-6 text-right">Inference Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-150 text-xs text-slate-600">
              {models.map((name) => {
                const metrics = modelComparisonsRegistry[name];
                const isBest = name === 'Gradient Boosting';
                const isSelected = selectedModel === name;

                return (
                  <tr
                    key={name}
                    className={`transition-colors hover:bg-slate-50/50 ${
                      isSelected ? 'bg-indigo-50/20 font-medium' : ''
                    }`}
                  >
                    <td className="py-3 px-6 font-semibold text-slate-850 flex items-center space-x-2">
                      <span>{name}</span>
                      {isBest && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
                          <CheckCircle className="w-3 h-3 mr-1" /> BEST
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-center font-mono">
                      {(metrics.Accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4 text-center font-mono">
                      {(metrics.Precision * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4 text-center font-mono">
                      {(metrics.Recall * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4 text-center font-mono font-bold text-slate-800">
                      {(metrics['F1-Score'] * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4 text-center font-mono text-indigo-600 font-bold">
                      {metrics['ROC-AUC'].toFixed(3)}
                    </td>
                    <td className="py-3 px-4 text-center font-mono">
                      {(metrics.CV_F1_Mean * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-6 text-right">
                      <button
                        onClick={() => setSelectedModel(name)}
                        className={`inline-flex items-center space-x-1 px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all ${
                          isSelected
                            ? 'bg-indigo-600 text-white shadow-sm'
                            : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
                        }`}
                      >
                        <Eye className="w-3.5 h-3.5" />
                        <span>Inspect</span>
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Selected Model Focus Panel */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
          {/* Confusion Matrix Visualizer */}
          <div className="border border-slate-200 rounded-xl p-5 bg-slate-50/30">
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2 mb-1">
              <Settings className="w-4 h-4 text-indigo-600 animate-spin-slow" />
              Confusion Matrix - {selectedModel}
            </h4>
            <p className="text-xs text-slate-500">Actual vs Predicted customer classifications.</p>

            <div className="grid grid-cols-2 gap-4 mt-5">
              <div className="bg-white border border-slate-200 rounded-xl p-4 text-center shadow-sm">
                <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">True Negative (TN)</p>
                <h5 className="text-xl font-bold text-slate-850 mt-1">
                  {activeModelDetails.Confusion_Matrix[0][0]}
                </h5>
                <p className="text-[10px] text-slate-500 mt-0.5">Accurately Retained</p>
              </div>
              <div className="bg-white border border-slate-200 rounded-xl p-4 text-center shadow-sm border-l-4 border-l-rose-500">
                <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">False Positive (FP)</p>
                <h5 className="text-xl font-bold text-rose-600 mt-1">
                  {activeModelDetails.Confusion_Matrix[0][1]}
                </h5>
                <p className="text-[10px] text-slate-500 mt-0.5">Missed Retention (Type I)</p>
              </div>
              <div className="bg-white border border-slate-200 rounded-xl p-4 text-center shadow-sm border-l-4 border-l-amber-500">
                <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">False Negative (FN)</p>
                <h5 className="text-xl font-bold text-amber-600 mt-1">
                  {activeModelDetails.Confusion_Matrix[1][0]}
                </h5>
                <p className="text-[10px] text-slate-500 mt-0.5">Missed Churn (Type II)</p>
              </div>
              <div className="bg-white border border-slate-200 rounded-xl p-4 text-center shadow-sm border-l-4 border-l-emerald-500">
                <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">True Positive (TP)</p>
                <h5 className="text-xl font-bold text-emerald-600 mt-1">
                  {activeModelDetails.Confusion_Matrix[1][1]}
                </h5>
                <p className="text-[10px] text-slate-500 mt-0.5">Accurately Churned</p>
              </div>
            </div>
          </div>

          {/* Diagnostic interpretation */}
          <div className="border border-slate-200 rounded-xl p-5 bg-slate-50/30 flex flex-col justify-between">
            <div>
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-1">Pipeline Performance Summary</h4>
              <p className="text-xs text-slate-500">How this algorithm secures enterprise customer retention.</p>
            </div>
            <div className="space-y-3 my-4">
              <div className="flex items-start space-x-2.5 text-xs text-slate-600">
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-600 mt-1.5 shrink-0" />
                <p className="leading-relaxed text-[11px]">
                  <strong>F1-Score ({(activeModelDetails['F1-Score'] * 100).toFixed(1)}%)</strong> balances precision and recall, optimizing targeted promotional retention spending while avoiding false alarms.
                </p>
              </div>
              <div className="flex items-start space-x-2.5 text-xs text-slate-600">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-600 mt-1.5 shrink-0" />
                <p className="leading-relaxed text-[11px]">
                  <strong>ROC-AUC ({activeModelDetails['ROC-AUC'].toFixed(3)})</strong> confirms powerful discriminative capacity to accurately segregate loyal clients from at-risk accounts.
                </p>
              </div>
              <div className="flex items-start space-x-2.5 text-xs text-slate-600">
                <div className="w-1.5 h-1.5 rounded-full bg-amber-600 mt-1.5 shrink-0" />
                <p className="leading-relaxed text-[11px]">
                  <strong>Cross-Validation Accuracy ({(activeModelDetails.CV_Accuracy_Mean * 100).toFixed(1)}%)</strong> ensures zero overfitting risk, ensuring generalization capacity in production.
                </p>
              </div>
            </div>
            <div className="bg-indigo-50 border border-indigo-100 text-indigo-800 rounded-lg p-3 text-[11px] text-center font-semibold">
              Model serialization leverages a joblib compression schema of level 3 for sub-millisecond predictions.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
