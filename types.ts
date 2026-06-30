export interface CustomerRecord {
  gender: 'Male' | 'Female';
  SeniorCitizen: 0 | 1;
  Partner: 'Yes' | 'No';
  Dependents: 'Yes' | 'No';
  tenure: number;
  PhoneService: 'Yes' | 'No';
  MultipleLines: 'Yes' | 'No' | 'No phone service';
  InternetService: 'DSL' | 'Fiber optic' | 'No';
  OnlineSecurity: 'Yes' | 'No' | 'No internet service';
  OnlineBackup: 'Yes' | 'No' | 'No internet service';
  DeviceProtection: 'Yes' | 'No' | 'No internet service';
  TechSupport: 'Yes' | 'No' | 'No internet service';
  StreamingTV: 'Yes' | 'No' | 'No internet service';
  StreamingMovies: 'Yes' | 'No' | 'No internet service';
  Contract: 'Month-to-month' | 'One year' | 'Two year';
  PaperlessBilling: 'Yes' | 'No';
  PaymentMethod:
    | 'Electronic check'
    | 'Mailed check'
    | 'Bank transfer (automatic)'
    | 'Credit card (automatic)';
  MonthlyCharges: number;
  TotalCharges: number;
}

export interface ModelMetrics {
  Accuracy: number;
  Precision: number;
  Recall: number;
  'F1-Score': number;
  'ROC-AUC': number;
  CV_Accuracy_Mean: number;
  CV_F1_Mean: number;
  Confusion_Matrix: number[][];
  Feature_Importances: number[];
}

export interface ExplanationFactor {
  feature: string;
  value: string;
  impact: 'High Risk' | 'Moderate Risk' | 'Low Risk' | 'Healthy (No Risk Drivers)';
  description: string;
}

export interface PredictionResult {
  prediction: 'Churn' | 'Retain';
  prediction_binary: number;
  churn_probability: number;
  risk_level: 'High' | 'Medium' | 'Low';
  model_used: string;
  explainability_factors: ExplanationFactor[];
  retention_playbook: string[];
}
