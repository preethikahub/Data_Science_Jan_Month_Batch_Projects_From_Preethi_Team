import { BookOpen, Server, Zap, Shield, HelpCircle, GraduationCap } from 'lucide-react';

export default function AboutView() {
  const steps = [
    {
      title: 'Exploratory Data Analysis',
      icon: Zap,
      desc: 'Executing statistical summaries, target imbalance mapping, and univariate histograms. Inspecting service dependencies (Fiber optic vs DSL) and high-density attrition zones.',
    },
    {
      title: 'Advanced Preprocessing',
      icon: GraduationCap,
      desc: 'Imputing missing data, conducting IQR outlier isolation, applying Label Encoding to binary variables, One-Hot encoding to multi-categorical values, and fitting StandardScalers to continuous features.',
    },
    {
      title: 'Class Imbalance Correction',
      icon: Shield,
      desc: 'Applying Synthetic Minority Over-sampling Technique (SMOTE) to mathematically generate synthetic minority class records, neutralizing prediction bias toward loyal customers.',
    },
    {
      title: 'Benchmarked Classifiers Suite',
      icon: Server,
      desc: 'Fitting nine separate algorithms including tree ensembles, linear models, and SVMs. Running 5-fold cross-validation to assess robustness and prevent overfitting.',
    },
  ];

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-8" id="about-view-root">
      {/* Intro */}
      <div className="space-y-3">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-indigo-600" />
          Customer Churn Machine Learning Pipeline
        </h2>
        <p className="text-xs text-slate-600 leading-relaxed">
          Customer retention is a core driver of recurring revenue for Telecom, SaaS, and subscription-based companies. 
          By combining advanced data engineering, classification modeling, and prescriptive decision systems, 
          organizations can predict churn before it occurs and deploy optimized retention interventions.
        </p>
      </div>

      {/* Structured pipeline card blocks */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {steps.map((st, idx) => {
          const Icon = st.icon;
          return (
            <div
              key={st.title}
              className="border border-slate-200 rounded-xl p-5 bg-slate-50/30 flex items-start space-x-4"
            >
              <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-lg shrink-0 mt-0.5">
                <Icon className="w-5 h-5" />
              </div>
              <div className="space-y-1">
                <span className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">
                  Phase 0{idx + 1}
                </span>
                <h3 className="text-xs font-bold text-slate-800">{st.title}</h3>
                <p className="text-xs text-slate-500 leading-relaxed mt-1">{st.desc}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Analytical Metric guides */}
      <div className="border border-slate-200 rounded-xl p-5 bg-slate-50/50 space-y-4">
        <h3 className="text-xs font-bold text-slate-800 flex items-center gap-2">
          <HelpCircle className="w-4.5 h-4.5 text-indigo-600" />
          Analytical Evaluation Metrics Guide
        </h3>
        <p className="text-xs text-slate-500 leading-relaxed">
          While simple accuracy measures overall predictions correctness, churn modeling requires robust multi-metric evaluations:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 text-xs">
          <div className="bg-white p-4 rounded-lg border border-slate-200 space-y-1.5 shadow-sm">
            <h4 className="font-bold text-rose-700 uppercase tracking-wide text-[10px]">Recall (Sensitivity)</h4>
            <p className="text-slate-500 leading-relaxed text-xs">
              Measures the proportion of actual churned customers correctly identified. Maximizing recall is paramount to ensure high-risk accounts are never missed.
            </p>
          </div>
          <div className="bg-white p-4 rounded-lg border border-slate-200 space-y-1.5 shadow-sm">
            <h4 className="font-bold text-indigo-700 uppercase tracking-wide text-[10px]">Precision</h4>
            <p className="text-slate-500 leading-relaxed text-xs">
              Measures predictions correctness (of all predicted churners, how many actually churn). High precision prevents unnecessary discount spending on loyal clients.
            </p>
          </div>
          <div className="bg-white p-4 rounded-lg border border-slate-200 space-y-1.5 shadow-sm">
            <h4 className="font-bold text-emerald-700 uppercase tracking-wide text-[10px]">F1-Score</h4>
            <p className="text-slate-500 leading-relaxed text-xs">
              The harmonic mean of Precision and Recall. This serves as the primary metric to optimize our SMOTE threshold, balancing targeted promotional expenditures.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
