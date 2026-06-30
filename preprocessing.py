"""
Customer Churn Prediction - Preprocessing Pipeline
Author: Senior ML Engineer & Architect
File: preprocessing.py

This module contains the ChurnPreprocessor class, which encapsulates the entire 
data preprocessing and feature engineering pipeline for customer churn prediction.
It adheres strictly to PEP 8 standards, utilizes robust exception handling, 
and includes structured logging.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PreprocessingPipeline")


def generate_synthetic_data(num_samples=1000, random_state=42):
    """
    Generates a highly realistic synthetic Telco Customer Churn dataset matching
    the schema and statistical properties of the Kaggle Telco Churn dataset.
    This guarantees that the project is runnable out-of-the-box.
    """
    logger.info(f"Generating {num_samples} realistic synthetic customer records...")
    np.random.seed(random_state)
    
    # Generate Customer IDs
    customer_ids = [f"{np.random.randint(1000, 9999)}-{np.random.choice(['AAAAA', 'BBBBB', 'CCCCC', 'DDDDD', 'EEEEE'])}" for _ in range(num_samples)]
    
    # Generate demographics
    gender = np.random.choice(["Male", "Female"], size=num_samples)
    senior_citizen = np.random.choice([0, 1], p=[0.84, 0.16], size=num_samples)
    partner = np.random.choice(["Yes", "No"], p=[0.48, 0.52], size=num_samples)
    dependents = np.random.choice(["Yes", "No"], p=[0.30, 0.70], size=num_samples)
    
    # Generate tenure (correlated with contracts)
    contract_choices = ["Month-to-month", "One year", "Two year"]
    contracts = np.random.choice(contract_choices, p=[0.55, 0.21, 0.24], size=num_samples)
    
    tenure = []
    for contract in contracts:
        if contract == "Month-to-month":
            tenure.append(int(np.random.exponential(scale=10) + 1))
        elif contract == "One year":
            tenure.append(int(np.random.normal(loc=36, scale=12)))
        else:
            tenure.append(int(np.random.normal(loc=60, scale=10)))
            
    tenure = np.clip(tenure, 1, 72).astype(int)
    
    # Services
    phone_service = np.random.choice(["Yes", "No"], p=[0.90, 0.10], size=num_samples)
    
    multiple_lines = []
    for phone in phone_service:
        if phone == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], p=[0.34, 0.44, 0.22], size=num_samples)
    
    # Internet-dependent services
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []
    
    for internet in internet_service:
        if internet == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
            streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))
            
    paperless_billing = np.random.choice(["Yes", "No"], p=[0.59, 0.41], size=num_samples)
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        p=[0.34, 0.23, 0.21, 0.22],
        size=num_samples
    )
    
    # Charges (dependent on internet service)
    monthly_charges = []
    for internet in internet_service:
        if internet == "No":
            monthly_charges.append(float(np.random.normal(loc=21, scale=3)))
        elif internet == "DSL":
            monthly_charges.append(float(np.random.normal(loc=58, scale=12)))
        else:  # Fiber optic
            monthly_charges.append(float(np.random.normal(loc=91, scale=14)))
            
    monthly_charges = np.clip(monthly_charges, 18, 120).round(2)
    
    total_charges = []
    for m, t in zip(monthly_charges, tenure):
        # Total charge is tenure * monthly charges with slight billing noise
        noise = np.random.normal(loc=0, scale=0.02 * m * t)
        total_charges.append(round(max(m, (m * t) + noise), 2))
        
    # Introduce some missing value strings in TotalCharges to test preprocessing robustness (5 records)
    total_charges_str = [str(tc) for tc in total_charges]
    for idx in np.random.choice(range(num_samples), size=5, replace=False):
        total_charges_str[idx] = " "  # Standard missing value representation in the actual dataset
        
    # Generate Churn based on risk rules
    # Factors increasing churn: high monthly charges, Fiber optic, month-to-month contract, no security/support, short tenure
    churn_probs = []
    for i in range(num_samples):
        prob = 0.1  # Baseline risk
        
        # Tenure effect (Logarithmic decrease in risk as loyalty increases)
        prob += max(0, 0.45 - (np.log1p(tenure[i]) / 10))
        
        # Contract effect
        if contracts[i] == "Month-to-month":
            prob += 0.35
        elif contracts[i] == "One year":
            prob += 0.05
            
        # Internet service effect
        if internet_service[i] == "Fiber optic":
            prob += 0.15
        elif internet_service[i] == "No":
            prob -= 0.05
            
        # Support services
        if internet_service[i] != "No":
            if online_security[i] == "No":
                prob += 0.08
            if tech_support[i] == "No":
                prob += 0.08
                
        # Senior status
        if senior_citizen[i] == 1:
            prob += 0.05
            
        # Payment method
        if payment_method[i] == "Electronic check":
            prob += 0.10
            
        # Total charges ratio check
        if monthly_charges[i] > 80:
            prob += 0.05
            
        # Clip to valid probability range
        prob = np.clip(prob, 0.02, 0.98)
        churn_probs.append(prob)
        
    churn = [("Yes" if np.random.rand() < p else "No") for p in churn_probs]
    
    # Package into DataFrame
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contracts,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges_str,
        "Churn": churn
    })
    
    return df


class ChurnPreprocessor:
    """
    A professional, reproducible Preprocessing Pipeline designed for
    classifying customer churn. Supports encoding, scaling, engineered feature addition,
    missing value handling, and custom oversampling to solve class imbalance.
    """
    
    def __init__(self, scaling_method="standard"):
        """
        Initializes preprocessor settings.
        :param scaling_method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
        """
        self.scaling_method = scaling_method
        self.scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
        self.label_encoders = {}
        self.one_hot_columns = []
        self.numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        self.categorical_cols = []
        self.binary_cols = []
        self.feature_columns = []
        self.is_fitted = False
        
    def perform_eda_summary(self, df):
        """
        Performs programmatic Exploratory Data Analysis.
        Returns statistical and structural insights.
        """
        logger.info("Executing Exploratory Data Analysis Programmatic Summary...")
        
        # Clean temporary copies to get precise statistics
        df_clean = df.copy()
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median())
        
        summary = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "empty_string_charges": int((df["TotalCharges"].astype(str).str.strip() == "").sum()),
            "duplicates": int(df.duplicated().sum()),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "target_distribution": df["Churn"].value_counts(normalize=True).to_dict(),
            "churn_count": df["Churn"].value_counts().to_dict(),
            "numerical_summary": df_clean[self.numerical_cols].describe().to_dict(),
        }
        
        # Correlation Matrix
        num_df = df_clean[self.numerical_cols].copy()
        num_df["Churn_Numeric"] = df_clean["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
        summary["correlation_matrix"] = num_df.corr().to_dict()
        
        return summary
        
    def fit(self, df):
        """
        Identifies and fits scaling and encoders based on training dataset.
        """
        logger.info("Fitting Preprocessing Pipeline on dataset...")
        df_clean = df.copy()
        
        # 1. Clean customer ID and target variables if present
        if "customerID" in df_clean.columns:
            df_clean = df_clean.drop(columns=["customerID"])
            
        # 2. Convert TotalCharges to float and handle empty strings
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
        total_charges_median = df_clean["TotalCharges"].median()
        self.total_charges_median_val = total_charges_median if not pd.isna(total_charges_median) else 0.0
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(self.total_charges_median_val)
        
        # 3. Handle categorical column types
        target_col = "Churn"
        all_cols = df_clean.columns.tolist()
        if target_col in all_cols:
            all_cols.remove(target_col)
            
        # Automatically separate binary features (Yes/No, Male/Female) from multi-class categorical features
        self.categorical_cols = []
        self.binary_cols = []
        
        for col in all_cols:
            if col not in self.numerical_cols:
                unique_vals = df_clean[col].dropna().unique()
                if len(unique_vals) <= 2:
                    self.binary_cols.append(col)
                else:
                    self.categorical_cols.append(col)
                    
        logger.info(f"Detected Numerical: {self.numerical_cols}")
        logger.info(f"Detected Binary: {self.binary_cols}")
        logger.info(f"Detected Categorical: {self.categorical_cols}")
        
        # 4. Fit Label Encoders for binary and target columns
        for col in self.binary_cols:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            self.label_encoders[col] = le
            
        # 5. Handle categorical features with One-Hot encoding schema definitions
        # Store categorical category names for reproducible one-hot encoding
        self.categories_map = {}
        self.one_hot_columns = []
        for col in self.categorical_cols:
            unique_cats = sorted(df_clean[col].astype(str).unique())
            self.categories_map[col] = unique_cats
            for cat in unique_cats[1:]:  # Drop first to avoid multi-collinearity
                self.one_hot_columns.append(f"{col}_{cat}")
                
        # 6. Fit Scaler on numerical features and engineered numerical columns
        # To engineer features during fit:
        df_engineered = self._engineer_features_internal(df_clean)
        engineered_numeric_cols = self.numerical_cols + ["ChargesRatio", "TenureToCharges"]
        
        self.scaler.fit(df_engineered[engineered_numeric_cols])
        
        # Track final features
        self.feature_columns = (
            self.binary_cols + 
            self.one_hot_columns + 
            ["ChargesRatio", "TenureToCharges"] + 
            self.numerical_cols
        )
        self.is_fitted = True
        logger.info(f"Pipeline fitted successfully with {len(self.feature_columns)} final features.")
        return self
        
    def _engineer_features_internal(self, df):
        """
        Internal helper to generate customer metrics safely.
        """
        df_eng = df.copy()
        
        # Charges Ratio: How much the monthly charges weigh relative to total charges
        # Avoid division by zero
        df_eng["ChargesRatio"] = df_eng["MonthlyCharges"] / (df_eng["TotalCharges"] + 1.0)
        
        # Tenure to Charges Ratio: Customer longevity index
        df_eng["TenureToCharges"] = df_eng["tenure"] / (df_eng["MonthlyCharges"] + 1.0)
        
        return df_eng
        
    def transform(self, df, is_inference=False):
        """
        Transforms raw data into a fully-numeric scaled format ready for ML models.
        """
        if not self.is_fitted:
            raise ValueError("The preprocessor must be fitted first!")
            
        df_trans = df.copy()
        
        # Drop customer ID
        if "customerID" in df_trans.columns:
            df_trans = df_trans.drop(columns=["customerID"])
            
        # Clean target variable if it exists and we're not running inference
        y = None
        if "Churn" in df_trans.columns:
            # Check if Churn is yes/no
            if not is_inference:
                y = df_trans["Churn"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0).values
            df_trans = df_trans.drop(columns=["Churn"])
            
        # Convert TotalCharges to float and handle empty strings
        df_trans["TotalCharges"] = pd.to_numeric(df_trans["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
        df_trans["TotalCharges"] = df_trans["TotalCharges"].fillna(self.total_charges_median_val)
        
        # Transform binary columns
        for col in self.binary_cols:
            if col in df_trans.columns:
                le = self.label_encoders[col]
                # Safe mapping for unseen categories
                df_trans[col] = df_trans[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else 0
                )
            else:
                df_trans[col] = 0
                
        # Engineer features
        df_trans = self._engineer_features_internal(df_trans)
        
        # Transform categorical columns via manual reproducible one-hot encoding
        for col in self.categorical_cols:
            cats = self.categories_map[col]
            for cat in cats[1:]:
                dummy_col = f"{col}_{cat}"
                if col in df_trans.columns:
                    df_trans[dummy_col] = (df_trans[col].astype(str) == cat).astype(int)
                else:
                    df_trans[dummy_col] = 0
                    
        # Scale all numerical columns
        engineered_numeric_cols = self.numerical_cols + ["ChargesRatio", "TenureToCharges"]
        scaled_vals = self.scaler.transform(df_trans[engineered_numeric_cols])
        
        for idx, col in enumerate(engineered_numeric_cols):
            df_trans[col] = scaled_vals[:, idx]
            
        # Align features perfectly
        X = df_trans[self.feature_columns].copy().values
        
        if is_inference:
            return X
        return X, y
        
    def handle_class_imbalance_smote(self, X, y, random_state=42):
        """
        Handles severe customer churn class imbalance using SMOTE
        (Synthetic Minority Over-sampling Technique).
        This is a robust custom Python vector implementation of SMOTE to guarantee
        no external dependencies like imbalanced-learn fail during compile.
        """
        logger.info("Executing SMOTE oversampling to balance Churn target class...")
        np.random.seed(random_state)
        
        # Find minority and majority indexes
        minority_idx = np.where(y == 1)[0]
        majority_idx = np.where(y == 0)[0]
        
        n_minority = len(minority_idx)
        n_majority = len(majority_idx)
        
        if n_minority >= n_majority:
            logger.info("Class imbalance is not severe. No oversampling performed.")
            return X, y
            
        # Get minority class samples
        X_min = X[minority_idx]
        
        # Calculate how many synthetic samples to generate
        n_synthetic_needed = n_majority - n_minority
        logger.info(f"Oversampling minority class by generating {n_synthetic_needed} synthetic samples...")
        
        # Build a basic k-NN algorithm to find nearest neighbors in minority class
        # k=5 is standard
        k = 5
        synthetic_samples = []
        
        for i in range(n_synthetic_needed):
            # Select random minority sample
            target_idx = np.random.choice(n_minority)
            target = X_min[target_idx]
            
            # Find distances to other minority samples
            distances = np.sum((X_min - target) ** 2, axis=1)
            # Sort and select k nearest neighbors (excluding self, index 0 is always distance 0)
            nearest_neighbors_idx = np.argsort(distances)[1:k+1]
            
            # Pick a random neighbor
            neighbor_idx = np.random.choice(nearest_neighbors_idx)
            neighbor = X_min[neighbor_idx]
            
            # Linearly interpolate
            diff = neighbor - target
            gap = np.random.rand()
            synthetic = target + (gap * diff)
            synthetic_samples.append(synthetic)
            
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.ones(n_synthetic_needed)
        
        # Merge datasets
        X_balanced = np.vstack([X, X_synthetic])
        y_balanced = np.hstack([y, y_synthetic])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_balanced))
        X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]
        
        logger.info(f"Oversampling completed. Balanced dataset shape: {X_balanced.shape}")
        return X_balanced, y_balanced


# Unit testing module trigger
if __name__ == "__main__":
    df = generate_synthetic_data(100)
    preprocessor = ChurnPreprocessor()
    eda = preprocessor.perform_eda_summary(df)
    preprocessor.fit(df)
    X, y = preprocessor.transform(df)
    X_bal, y_bal = preprocessor.handle_class_imbalance_smote(X, y)
    print("EDA Shape:", eda["shape"])
    print("X preprocessed shape:", X.shape)
    print("X balanced shape:", X_bal.shape)
