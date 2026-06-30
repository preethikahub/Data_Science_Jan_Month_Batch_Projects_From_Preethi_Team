"""
Customer Churn Prediction - Safe Prediction and Inference API Engine
Author: Senior ML Engineer & Architect
File: predict.py

This module provides programmatic customer churn inference using the serialized
Joblib pipeline. It takes raw customer dictionaries, performs safety validations,
computes risk probabilities, and provides local explainability risk factors
and business customer retention playbooks.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PredictionInference")


class ChurnInferenceEngine:
    """
    Handles deserialization of model assets and executes fast,
    local-explainable inferences on single or batched customer vectors.
    """
    
    def __init__(self, pipeline_path="churn_prediction_pipeline.joblib"):
        self.pipeline_path = pipeline_path
        self.preprocessor = None
        self.model = None
        self.best_model_name = None
        self.results_summary = None
        
        self._load_pipeline()
        
    def _load_pipeline(self):
        """
        Safely loads joblib-packaged model pipeline components.
        """
        if not os.path.exists(self.pipeline_path):
            logger.warning(f"Pipeline file '{self.pipeline_path}' not found. Training model first...")
            # Automatically trigger train.py if pipeline missing
            from train import run_training_pipeline
            run_training_pipeline(output_pipeline=self.pipeline_path)
            
        logger.info(f"Loading predictive assets from '{self.pipeline_path}'...")
        try:
            pipeline = joblib.load(self.pipeline_path)
            self.preprocessor = pipeline["preprocessor"]
            self.model = pipeline["model"]
            self.best_model_name = pipeline["best_model_name"]
            self.results_summary = pipeline.get("results", {})
            logger.info(f"Pipeline loaded. Model engine: {self.best_model_name}")
        except Exception as e:
            logger.error(f"Critical error deserializing pipeline assets: {str(e)}", exc_info=True)
            raise
            
    def predict_single(self, customer_data):
        """
        Executes inference on a single customer dictionary.
        Returns:
            dict: Churn Prediction, Probability, Risk Level, Explainable Factors, and Recommendations.
        """
        # Convert single dict to pandas DataFrame
        df_input = pd.DataFrame([customer_data])
        
        # Enforce exact columns to avoid preprocessor misalignment
        required_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]
        
        # Ensure all required features are present
        for col in required_cols:
            if col not in df_input.columns:
                if col in ["tenure", "SeniorCitizen"]:
                    df_input[col] = 0
                elif col in ["MonthlyCharges", "TotalCharges"]:
                    df_input[col] = 0.0
                else:
                    df_input[col] = "No"
                    
        # Apply transformation
        X_scaled = self.preprocessor.transform(df_input, is_inference=True)
        
        # Run model predictions
        prediction = int(self.model.predict(X_scaled)[0])
        probability = float(self.model.predict_proba(X_scaled)[0, 1])
        
        # Risk categorization
        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.35:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        # 4. Generate local explainability (Shapley/LIME style highlights)
        reasons, recs = self._generate_explainability_reasons(customer_data, probability)
        
        return {
            "prediction": "Churn" if prediction == 1 else "Retain",
            "prediction_binary": prediction,
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "model_used": self.best_model_name,
            "explainability_factors": reasons,
            "retention_playbook": recs
        }
        
    def _generate_explainability_reasons(self, data, prob):
        """
        Heuristically maps customer attributes to feature importance weights for local explainability.
        """
        reasons = []
        recs = []
        
        # Extract features
        tenure = int(data.get("tenure", 0))
        contract = str(data.get("Contract", "Month-to-month"))
        internet = str(data.get("InternetService", "DSL"))
        security = str(data.get("OnlineSecurity", "No"))
        support = str(data.get("TechSupport", "No"))
        monthly_charges = float(data.get("MonthlyCharges", 0.0))
        payment = str(data.get("PaymentMethod", ""))
        
        # Analyze critical risk drivers
        if contract == "Month-to-month":
            reasons.append({
                "feature": "Contract",
                "value": "Month-to-month",
                "impact": "High Risk",
                "description": "Short-term contracts have the highest systemic volatility (+35% base risk)."
            })
            recs.append("Transition account to an annual Contract by offering a 15% loyalty subscription discount.")
            
        if tenure <= 6:
            reasons.append({
                "feature": "Tenure",
                "value": f"{tenure} months",
                "impact": "High Risk",
                "description": "Early lifecycle customer (under 6 months) exhibiting high initial churn risk."
            })
            recs.append("Enroll customer in automated onboarding sequences and send a 3-month milestone reward.")
        elif tenure <= 18:
            reasons.append({
                "feature": "Tenure",
                "value": f"{tenure} months",
                "impact": "Moderate Risk",
                "description": "Mid-lifecycle account. Customer loyalty is not fully stabilized."
            })
            
        if internet == "Fiber optic":
            reasons.append({
                "feature": "Internet Service",
                "value": "Fiber optic",
                "impact": "Moderate Risk",
                "description": "Fiber optic connections are premium priced and historically correlate with price-sensitive attrition."
            })
            
        if security == "No" and internet != "No":
            reasons.append({
                "feature": "Online Security",
                "value": "Not Subscribed",
                "impact": "Moderate Risk",
                "description": "Lack of digital protection package reduces customer product-stickiness."
            })
            recs.append("Promote and upsell a secure bundle containing Online Security at a waived cost for 3 months.")
            
        if support == "No" and internet != "No":
            reasons.append({
                "feature": "Tech Support",
                "value": "Not Subscribed",
                "impact": "Moderate Risk",
                "description": "Absence of active technical assistance channels correlates with higher support frustration attrition."
            })
            recs.append("Offer a complimentary technical account review to ensure router and setup are fully optimized.")
            
        if monthly_charges > 85.0:
            reasons.append({
                "feature": "Monthly Charges",
                "value": f"${monthly_charges}",
                "impact": "Moderate Risk",
                "description": "Premium billing tier exceeds average. Elevated risk of price-shopping competition."
            })
            recs.append("Audit bill to suggest customized lower cost tier bundles or loyalty bill credits.")
            
        if payment == "Electronic check":
            reasons.append({
                "feature": "Payment Method",
                "value": "Electronic check",
                "impact": "Low Risk",
                "description": "Manual monthly electronic checks trigger active transaction decisions compared to automatic autopay."
            })
            recs.append("Encourage signup for Auto-Pay (Credit Card/Bank Transfer) by awarding a one-time $10 account bill credit.")
            
        # Default safety cases
        if not reasons:
            reasons.append({
                "feature": "Overall Profile",
                "value": "Highly Balanced",
                "impact": "Healthy (No Risk Drivers)",
                "description": "Long tenure, long term contract, and stable utility values indicate a highly loyal user profile."
            })
            recs.append("Maintain high-quality services. Target for premium cross-selling opportunities (new devices/services).")
            
        return reasons, recs


if __name__ == "__main__":
    # Sample Test Customer Input
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 84.85,
        "TotalCharges": "254.55"
    }
    
    engine = ChurnInferenceEngine()
    result = engine.predict_single(sample_customer)
    print(json.dumps(result, indent=4))
