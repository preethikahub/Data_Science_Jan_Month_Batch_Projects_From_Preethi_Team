import React, { useState, useEffect } from 'react';
import { UserCheck, Sliders, Sparkles, HelpCircle, UserPlus2 } from 'lucide-react';
import { CustomerRecord } from '../types';

interface PredictionFormProps {
  onSubmit: (data: CustomerRecord) => void;
}

const HIGH_RISK_PRESET: CustomerRecord = {
  gender: 'Female',
  SeniorCitizen: 1,
  Partner: 'No',
  Dependents: 'No',
  tenure: 3,
  PhoneService: 'Yes',
  MultipleLines: 'No',
  InternetService: 'Fiber optic',
  OnlineSecurity: 'No',
  OnlineBackup: 'No',
  DeviceProtection: 'No',
  TechSupport: 'No',
  StreamingTV: 'Yes',
  StreamingMovies: 'No',
  Contract: 'Month-to-month',
  PaperlessBilling: 'Yes',
  PaymentMethod: 'Electronic check',
  MonthlyCharges: 84.85,
  TotalCharges: 254.55,
};

const LOYAL_PRESET: CustomerRecord = {
  gender: 'Male',
  SeniorCitizen: 0,
  Partner: 'Yes',
  Dependents: 'Yes',
  tenure: 64,
  PhoneService: 'Yes',
  MultipleLines: 'Yes',
  InternetService: 'DSL',
  OnlineSecurity: 'Yes',
  OnlineBackup: 'Yes',
  DeviceProtection: 'Yes',
  TechSupport: 'Yes',
  StreamingTV: 'Yes',
  StreamingMovies: 'Yes',
  Contract: 'Two year',
  PaperlessBilling: 'No',
  PaymentMethod: 'Credit card (automatic)',
  MonthlyCharges: 89.9,
  TotalCharges: 5753.6,
};

export default function PredictionForm({ onSubmit }: PredictionFormProps) {
  const [formData, setFormData] = useState<CustomerRecord>({ ...HIGH_RISK_PRESET });
  const [autoCalc, setAutoCalc] = useState<boolean>(true);

  // Auto-calculate TotalCharges as MonthlyCharges * tenure
  useEffect(() => {
    if (autoCalc) {
      const calculated = parseFloat((formData.MonthlyCharges * formData.tenure).toFixed(2));
      setFormData((prev) => ({ ...prev, TotalCharges: calculated }));
    }
  }, [formData.MonthlyCharges, formData.tenure, autoCalc]);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    let finalValue: string | number = value;

    if (name === 'tenure') {
      finalValue = parseInt(value, 10) || 0;
    } else if (name === 'MonthlyCharges' || name === 'TotalCharges') {
      finalValue = parseFloat(value) || 0.0;
    } else if (name === 'SeniorCitizen') {
      finalValue = parseInt(value, 10) as 0 | 1;
    }

    setFormData((prev) => ({ ...prev, [name]: finalValue }));
  };

  const handleLoadPreset = (presetType: 'high' | 'loyal') => {
    const selected = presetType === 'high' ? HIGH_RISK_PRESET : LOYAL_PRESET;
    setFormData({ ...selected });
    setAutoCalc(true);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6" id="prediction-form-root">
      {/* Header and Preset Shortcuts */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Sliders className="w-5 h-5 text-indigo-600" />
            Customer Feature Evaluation
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Input subscriber specifications to evaluate churn risk probabilities.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 shrink-0">
          <button
            type="button"
            onClick={() => handleLoadPreset('high')}
            className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-lg text-xs font-bold bg-rose-50 border border-rose-100 text-rose-700 hover:bg-rose-100 transition-colors"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Load High Risk Preset</span>
          </button>
          <button
            type="button"
            onClick={() => handleLoadPreset('loyal')}
            className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-lg text-xs font-bold bg-indigo-50 border border-indigo-100 text-indigo-700 hover:bg-indigo-100 transition-colors"
          >
            <UserCheck className="w-3.5 h-3.5" />
            <span>Load Loyal Preset</span>
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Section 1: Demographics */}
        <div className="space-y-4">
          <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <UserPlus2 className="w-4 h-4 text-slate-400" />
            1. Customer Demographics
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Gender</label>
              <select
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none"
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Senior Citizen</label>
              <select
                name="SeniorCitizen"
                value={formData.SeniorCitizen}
                onChange={handleChange}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none"
              >
                <option value={0}>No (Younger/Adult)</option>
                <option value={1}>Yes (Senior Citizen)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Has Partner</label>
              <select
                name="Partner"
                value={formData.Partner}
                onChange={handleChange}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Has Dependents</label>
              <select
                name="Dependents"
                value={formData.Dependents}
                onChange={handleChange}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
          </div>
        </div>

        {/* Section 2: Services Subscriptions */}
        <div className="space-y-4 pt-2">
          <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-slate-400" />
            2. Active Service Subscriptions
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Phone Service</label>
              <select
                name="PhoneService"
                value={formData.PhoneService}
                onChange={handleChange}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Multiple Lines</label>
              <select
                name="MultipleLines"
                value={formData.MultipleLines}
                onChange={handleChange}
                disabled={formData.PhoneService === 'No'}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none disabled:opacity-50"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No phone service">No phone service</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Internet Connection</label>
              <select
                name="InternetService"
                value={formData.InternetService}
                onChange={handleChange}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none"
              >
                <option value="DSL">DSL Connection</option>
                <option value="Fiber optic">Fiber optic Connection</option>
                <option value="No">No Internet Service</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Cyber Security</label>
              <select
                name="OnlineSecurity"
                value={formData.OnlineSecurity}
                onChange={handleChange}
                disabled={formData.InternetService === 'No'}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none disabled:opacity-50"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Online Backup</label>
              <select
                name="OnlineBackup"
                value={formData.OnlineBackup}
                onChange={handleChange}
                disabled={formData.InternetService === 'No'}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none disabled:opacity-50"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Device Protection</label>
              <select
                name="DeviceProtection"
                value={formData.DeviceProtection}
                onChange={handleChange}
                disabled={formData.InternetService === 'No'}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none disabled:opacity-50"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Tech Support</label>
              <select
                name="TechSupport"
                value={formData.TechSupport}
                onChange={handleChange}
                disabled={formData.InternetService === 'No'}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none disabled:opacity-50"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Streaming TV</label>
              <select
                name="StreamingTV"
                value={formData.StreamingTV}
                onChange={handleChange}
                disabled={formData.InternetService === 'No'}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none disabled:opacity-50"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Streaming Movies</label>
              <select
                name="StreamingMovies"
                value={formData.StreamingMovies}
                onChange={handleChange}
                disabled={formData.InternetService === 'No'}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none disabled:opacity-50"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>
          </div>
        </div>

        {/* Section 3: Billing and Financials */}
        <div className="space-y-4 pt-2">
          <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <HelpCircle className="w-4 h-4 text-slate-400" />
            3. Account Terms & Billing Metrics
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Contract Term</label>
              <select
                name="Contract"
                value={formData.Contract}
                onChange={handleChange}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none"
              >
                <option value="Month-to-month">Month-to-month</option>
                <option value="One year">One year</option>
                <option value="Two year">Two year</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Paperless Billing</label>
              <select
                name="PaperlessBilling"
                value={formData.PaperlessBilling}
                onChange={handleChange}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Payment Method</label>
              <select
                name="PaymentMethod"
                value={formData.PaymentMethod}
                onChange={handleChange}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none appearance-none"
              >
                <option value="Electronic check">Electronic check</option>
                <option value="Mailed check">Mailed check</option>
                <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                <option value="Credit card (automatic)">Credit card (automatic)</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-5 items-end">
            <div className="md:col-span-1">
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Tenure (Months): {formData.tenure}</label>
              <input
                type="range"
                name="tenure"
                min={1}
                max={72}
                value={formData.tenure}
                onChange={handleChange}
                className="w-full accent-indigo-600 h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                <span>1 mo</span>
                <span>72 mo</span>
              </div>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Monthly Charges ($)</label>
              <input
                type="number"
                name="MonthlyCharges"
                step="0.01"
                min="18.0"
                max="120.0"
                value={formData.MonthlyCharges}
                onChange={handleChange}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Total Charges ($)</label>
              <input
                type="number"
                name="TotalCharges"
                step="0.01"
                value={formData.TotalCharges}
                onChange={handleChange}
                disabled={autoCalc}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 px-3 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 outline-none disabled:opacity-50 disabled:bg-slate-100"
              />
            </div>
            <div className="flex items-center space-x-2 pb-2.5">
              <input
                type="checkbox"
                id="autoCalcCheck"
                checked={autoCalc}
                onChange={(e) => setAutoCalc(e.target.checked)}
                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 w-4 h-4"
              />
              <label htmlFor="autoCalcCheck" className="text-xs font-bold text-slate-600 cursor-pointer select-none">
                Auto-calculate Total
              </label>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="pt-4 border-t border-slate-200 flex justify-end">
          <button
            type="submit"
            className="w-full sm:w-auto bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-6 py-3 rounded-lg text-xs shadow-sm hover:shadow transition-all flex items-center justify-center space-x-2"
          >
            <span>Run Churn Diagnostics</span>
          </button>
        </div>
      </form>
    </div>
  );
}
