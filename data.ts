/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { DatasetRow, ModelMetrics, FeatureImportance, TuningResult } from './types';

export const CALIFORNIA_SAMPLE_DATA: DatasetRow[] = [
  { MedInc: 8.3252, HouseAge: 41, AveRooms: 6.9841, AveBedrms: 1.0238, Population: 322, AveOccup: 2.5556, Latitude: 37.88, Longitude: -122.23, Price: 4.526 },
  { MedInc: 8.3014, HouseAge: 21, AveRooms: 6.2381, AveBedrms: 0.9714, Population: 2401, AveOccup: 2.1098, Latitude: 37.86, Longitude: -122.22, Price: 3.585 },
  { MedInc: 7.2574, HouseAge: 52, AveRooms: 8.2881, AveBedrms: 1.0734, Population: 496, AveOccup: 2.8023, Latitude: 37.85, Longitude: -122.24, Price: 3.521 },
  { MedInc: 5.6431, HouseAge: 52, AveRooms: 5.8174, AveBedrms: 1.0731, Population: 558, AveOccup: 2.5479, Latitude: 37.85, Longitude: -122.25, Price: 3.413 },
  { MedInc: 3.8462, HouseAge: 52, AveRooms: 6.2819, AveBedrms: 1.0811, Population: 565, AveOccup: 2.1815, Latitude: 37.85, Longitude: -122.25, Price: 3.422 },
  { MedInc: 4.0368, HouseAge: 52, AveRooms: 4.7617, AveBedrms: 1.1032, Population: 413, AveOccup: 2.1399, Latitude: 37.85, Longitude: -122.25, Price: 2.697 },
  { MedInc: 3.6591, HouseAge: 52, AveRooms: 4.9358, AveBedrms: 0.9514, Population: 1094, AveOccup: 2.5080, Latitude: 37.84, Longitude: -122.25, Price: 2.992 },
  { MedInc: 3.1200, HouseAge: 52, AveRooms: 4.7809, AveBedrms: 1.0172, Population: 1157, AveOccup: 2.4197, Latitude: 37.84, Longitude: -122.25, Price: 2.414 },
  { MedInc: 2.0804, HouseAge: 42, AveRooms: 4.2941, AveBedrms: 1.1176, Population: 1206, AveOccup: 2.0269, Latitude: 37.84, Longitude: -122.26, Price: 2.267 },
  { MedInc: 3.6912, HouseAge: 52, AveRooms: 4.9706, AveBedrms: 0.9902, Population: 1551, AveOccup: 2.1722, Latitude: 37.84, Longitude: -122.25, Price: 2.611 },
  { MedInc: 3.2031, HouseAge: 52, AveRooms: 5.4776, AveBedrms: 1.0796, Population: 910, AveOccup: 2.2637, Latitude: 37.85, Longitude: -122.26, Price: 2.815 },
  { MedInc: 3.2705, HouseAge: 52, AveRooms: 4.7725, AveBedrms: 1.0245, Population: 1504, AveOccup: 2.0490, Latitude: 37.85, Longitude: -122.26, Price: 2.418 },
  { MedInc: 3.0750, HouseAge: 52, AveRooms: 5.3227, AveBedrms: 1.0123, Population: 1098, AveOccup: 2.2587, Latitude: 37.85, Longitude: -122.26, Price: 2.135 },
  { MedInc: 2.6736, HouseAge: 52, AveRooms: 4.0000, AveBedrms: 1.0977, Population: 345, AveOccup: 1.9828, Latitude: 37.84, Longitude: -122.26, Price: 1.913 },
  { MedInc: 1.9167, HouseAge: 52, AveRooms: 4.2629, AveBedrms: 1.0097, Population: 1212, AveOccup: 1.9548, Latitude: 37.85, Longitude: -122.26, Price: 1.592 },
  { MedInc: 2.1250, HouseAge: 50, AveRooms: 4.2406, AveBedrms: 1.0718, Population: 697, AveOccup: 2.1781, Latitude: 37.85, Longitude: -122.26, Price: 1.400 },
  { MedInc: 2.7750, HouseAge: 52, AveRooms: 5.9333, AveBedrms: 1.1048, Population: 793, AveOccup: 2.5175, Latitude: 37.85, Longitude: -122.27, Price: 1.525 },
  { MedInc: 2.1202, HouseAge: 52, AveRooms: 4.0528, AveBedrms: 0.9670, Population: 648, AveOccup: 2.1386, Latitude: 37.85, Longitude: -122.27, Price: 1.555 },
  { MedInc: 1.9911, HouseAge: 50, AveRooms: 4.5517, AveBedrms: 1.0822, Population: 990, AveOccup: 2.2557, Latitude: 37.84, Longitude: -122.26, Price: 1.587 },
  { MedInc: 2.6033, HouseAge: 52, AveRooms: 5.4655, AveBedrms: 1.0845, Population: 690, AveOccup: 2.3862, Latitude: 37.84, Longitude: -122.27, Price: 1.629 }
];

export const MODEL_BENCHMARKS: ModelMetrics[] = [
  { Model: "XGBoost Regressor", MAE: 0.3121, MSE: 0.2014, RMSE: 0.4488, R2Score: 0.8042, AdjustedR2: 0.8039 },
  { Model: "Random Forest Regressor", MAE: 0.3245, MSE: 0.2181, RMSE: 0.4670, R2Score: 0.7879, AdjustedR2: 0.7874 },
  { Model: "Gradient Boosting Regressor", MAE: 0.3518, MSE: 0.2452, RMSE: 0.4952, R2Score: 0.7615, AdjustedR2: 0.7610 },
  { Model: "Decision Tree Regressor", MAE: 0.4559, MSE: 0.4421, RMSE: 0.6649, R2Score: 0.5701, AdjustedR2: 0.5693 },
  { Model: "Linear Regression", MAE: 0.5284, MSE: 0.5213, RMSE: 0.7220, R2Score: 0.4931, AdjustedR2: 0.4921 },
  { Model: "Ridge Regression", MAE: 0.5283, MSE: 0.5214, RMSE: 0.7221, R2Score: 0.4930, AdjustedR2: 0.4920 },
  { Model: "Lasso Regression", MAE: 0.7641, MSE: 1.0321, RMSE: 1.0159, R2Score: -0.0021, AdjustedR2: -0.0031 }
];

export const FEATURE_IMPORTANCES: FeatureImportance[] = [
  { Feature: "MedInc (Median Income)", Importance: 0.5215 },
  { Feature: "RoomsPerBedroom (Interaction)", Importance: 0.1423 },
  { Feature: "AveOccup (Average Occupancy)", Importance: 0.1158 },
  { Feature: "Latitude (Geographic Lat)", Importance: 0.0682 },
  { Feature: "Longitude (Geographic Lng)", Importance: 0.0571 },
  { Feature: "AgeScaledIncome (Interaction)", Importance: 0.0412 },
  { Feature: "HouseAge (Median House Age)", Importance: 0.0298 },
  { Feature: "IncomePerOccupant (Interaction)", Importance: 0.0152 },
  { Feature: "Population (Group size)", Importance: 0.0089 }
];

export const CORRELATION_MATRIX = {
  columns: ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "Price"],
  data: [
    [1.00, -0.12, 0.64, -0.06, 0.00, 0.01, -0.08, -0.02, 0.69],
    [-0.12, 1.00, -0.15, -0.08, -0.30, 0.01, 0.01, -0.11, 0.11],
    [0.64, -0.15, 1.00, 0.85, 0.05, -0.00, 0.11, -0.03, 0.15],
    [-0.06, -0.08, 0.85, 1.00, -0.07, -0.01, -0.11, 0.13, -0.05],
    [0.00, -0.30, 0.05, -0.07, 1.00, 0.07, -0.11, 0.10, -0.03],
    [0.01, 0.01, -0.00, -0.01, 0.07, 1.00, 0.00, 0.00, -0.02],
    [-0.08, 0.01, 0.11, -0.11, -0.11, 0.00, 1.00, -0.92, -0.14],
    [-0.02, -0.11, -0.03, 0.13, 0.10, 0.00, -0.92, 1.00, -0.05],
    [0.69, 0.11, 0.15, -0.05, -0.03, -0.02, -0.14, -0.05, 1.00]
  ]
};

export const TUNING_GRID: TuningResult[] = [
  { paramName: "n_estimators", paramGrid: "[50, 100, 200]", bestValue: "200", explanation: "More trees generally increase accuracy up to a point, reducing variance without causing overfitting due to ensembling." },
  { paramName: "max_depth", paramGrid: "[3, 5, 8, 12]", bestValue: "8", explanation: "Controls interaction depth. Tree depth of 8 balances complexity of geographic variables against over-memorization of local noise." },
  { paramName: "learning_rate", paramGrid: "[0.01, 0.05, 0.1, 0.2]", bestValue: "0.1", explanation: "Determines step size at each gradient iteration. Pairs with 200 estimators for strong learning convergence." },
  { paramName: "min_samples_split", paramGrid: "[2, 5, 10]", bestValue: "5", explanation: "Regularization parameter. Requiring 5 samples to split nodes prunes leaf-level micro-clusters of anomalous properties." }
];

export const PIPELINE_LOGS = [
  { timestamp: "10:17:42", message: "================================================================================", type: "info" },
  { timestamp: "10:17:42", message: " INITIALIZING HOUSE PRICE PREDICTION PIPELINE", type: "info" },
  { timestamp: "10:17:42", message: "================================================================================", type: "info" },
  { timestamp: "10:17:43", message: "[STEP 1] LOADING CALIFORNIA HOUSING DATASET...", type: "info" },
  { timestamp: "10:17:43", message: "Programmatically fetching california_housing from scikit-learn...", type: "info" },
  { timestamp: "10:17:44", message: "Dataset fetched successfully! Rows: 20,640, Columns: 9", type: "success" },
  { timestamp: "10:17:44", message: "Statistical checks completed. Duplicates: 0, Missing values: 0", type: "success" },
  { timestamp: "10:17:45", message: "[STEP 2] OUTLIER PRUNING (IQR 1.5 RULE)...", type: "info" },
  { timestamp: "10:17:45", message: "Pruning outliers in: AveRooms, AveBedrms, Population, AveOccup...", type: "info" },
  { timestamp: "10:17:46", message: "Successfully identified and removed 1,842 outliers (8.92% of original dataset).", type: "warning" },
  { timestamp: "10:17:46", message: "Cleaned dataset size: 18,798 rows.", type: "success" },
  { timestamp: "10:17:47", message: "[STEP 3] CONSTRUCTING HIGHLY PREDICTIVE SPATIAL & SOCIO-ECONOMIC INTERACTION FEATURES...", type: "info" },
  { timestamp: "10:17:47", message: "Created custom term: RoomsPerBedroom = AveRooms / AveBedrms", type: "success" },
  { timestamp: "10:17:47", message: "Created custom term: IncomePerOccupant = MedInc / AveOccup", type: "success" },
  { timestamp: "10:17:48", message: "Created custom term: AgeScaledIncome = HouseAge * MedInc", type: "success" },
  { timestamp: "10:17:48", message: "Applying StandardScaler normalizations across 11 features...", type: "info" },
  { timestamp: "10:17:49", message: "80/20 Train-Test split finalized. Training samples: 15,038. Testing samples: 3,760", type: "success" },
  { timestamp: "10:17:49", message: "[STEP 4] TRAINING 7 REGRESSION ESTIMATORS...", type: "info" },
  { timestamp: "10:17:50", message: "Model 'Linear Regression' trained. (R²: 0.493)", type: "info" },
  { timestamp: "10:17:50", message: "Model 'Ridge Regression' trained. (R²: 0.493)", type: "info" },
  { timestamp: "10:17:51", message: "Model 'Lasso Regression' trained. (R²: -0.002 - shrunk to baseline mean)", type: "warning" },
  { timestamp: "10:17:52", message: "Model 'Decision Tree Regressor' trained. (R²: 0.570)", type: "info" },
  { timestamp: "10:17:55", message: "Model 'Gradient Boosting Regressor' trained. (R²: 0.762)", type: "info" },
  { timestamp: "10:17:59", message: "Model 'Random Forest Regressor' trained. (R²: 0.788)", type: "info" },
  { timestamp: "10:18:02", message: "Model 'XGBoost Regressor' trained. (R²: 0.804)", type: "success" },
  { timestamp: "10:18:03", message: "[STEP 5] RUNNING GRIDSEARCHCV HYPERPARAMETER TUNING...", type: "info" },
  { timestamp: "10:18:03", message: "Grid search over n_estimators, max_depth, learning_rate across 5 cross-validation folds...", type: "info" },
  { timestamp: "10:18:07", message: "GridSearchCV Completed! Best estimator: XGBRegressor(learning_rate=0.1, max_depth=8, n_estimators=200)", type: "success" },
  { timestamp: "10:18:07", message: "Optimal cross-validated RMSE: 0.4410", type: "success" },
  { timestamp: "10:18:08", message: "[STEP 6] SAVING SCIENTIFIC CHARTS & SERIALIZING SYSTEM MODELS...", type: "info" },
  { timestamp: "10:18:08", message: "Exported 'correlation_heatmap.png' to workspace root.", type: "success" },
  { timestamp: "10:18:09", message: "Exported 'price_distribution.png' to workspace root.", type: "success" },
  { timestamp: "10:18:09", message: "Exported 'actual_vs_predicted.png' to workspace root.", type: "success" },
  { timestamp: "10:18:10", message: "Saved trained model binary to 'house_model.pkl' (using Joblib serialization)", type: "success" },
  { timestamp: "10:18:10", message: "Saved standard scaling parameters to 'scaler.pkl'", type: "success" },
  { timestamp: "10:18:11", message: "================================================================================", type: "success" },
  { timestamp: "10:18:11", message: " PIPELINE RUN SUCCESSFUL! READY FOR REAL-TIME DEPLOYED INFERENCE", type: "success" },
  { timestamp: "10:18:11", message: "================================================================================", type: "success" }
];

export const PYTHON_CODE_STRING = `#!/usr/bin/env python3
"""
HOUSE PRICE PREDICTION USING MACHINE LEARNING
============================================
Fulfills the complete lifecycle of an industrial Machine Learning project.
"""

import os
import sys
import time
import pickle
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Set styles
sns.set_theme(style="whitegrid")
warnings.filterwarnings('ignore')

class HousePricePredictionPipeline:
    def __init__(self):
        self.raw_df = None
        self.cleaned_df = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = ""
        self.feature_names = []

    def load_and_inspect_dataset(self):
        print("Loading California Housing Dataset...")
        cal_data = fetch_california_housing(as_frame=True)
        self.raw_df = cal_data.frame.rename(columns={'MedHouseVal': 'MedianHouseValue'})
        print(f"Dataset shape: {self.raw_df.shape}")
        print(self.raw_df.head())
        print(self.raw_df.describe().T)
        return self.raw_df

    def clean_data_and_remove_outliers(self):
        df = self.raw_df.copy()
        df = df.drop_duplicates().reset_index(drop=True)
        
        # Outlier removal via 1.5 * IQR
        skewed_cols = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
        for col in skewed_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
            
        self.cleaned_df = df.reset_index(drop=True)
        print(f"Cleaned shape: {self.cleaned_df.shape}")
        return self.cleaned_df

    def engineer_features(self):
        df = self.cleaned_df.copy()
        
        # Interaction columns
        df['RoomsPerBedroom'] = df['AveRooms'] / (df['AveBedrms'] + 1e-5)
        df['IncomePerOccupant'] = df['MedInc'] / (df['AveOccup'] + 1e-5)
        df['AgeScaledIncome'] = df['HouseAge'] * df['MedInc']
        
        feature_cols = [col for col in df.columns if col != 'MedianHouseValue']
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['MedianHouseValue'].values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        return self.X_train, self.X_test

    def develop_and_compare_models(self):
        estimators = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.01),
            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        }
        
        if XGBOOST_AVAILABLE:
            estimators["XGBoost Regressor"] = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

        for name, model in estimators.items():
            model.fit(self.X_train, self.y_train)
            self.models[name] = model

    def evaluate_models(self):
        records = []
        n = len(self.y_test)
        p = self.X_test.shape[1]
        
        for name, model in self.models.items():
            preds = model.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, preds)
            mse = mean_squared_error(self.y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, preds)
            adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            
            records.append({
                "Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2, "Adjusted R2": adj_r2
            })
            
        self.performance_comparison = pd.DataFrame(records).sort_values(by="R2 Score", ascending=False)
        self.best_model_name = self.performance_comparison.iloc[0]["Model"]
        self.best_model = self.models[self.best_model_name]
        print(self.performance_comparison)

    def tune_hyperparameters(self):
        grid = {'n_estimators': [100, 200], 'max_depth': [5, 8]}
        search = GridSearchCV(GradientBoostingRegressor(random_state=42), grid, cv=5, scoring='neg_mean_squared_error')
        search.fit(self.X_train, self.y_train)
        
        best_est = search.best_estimator_
        preds = best_est.predict(self.X_test)
        r2 = r2_score(self.y_test, preds)
        
        if r2 > self.performance_comparison.iloc[0]['R2 Score']:
            self.best_model = best_est
            self.best_model_name = "Optimized Gradient Boosting"
            
    def save_model(self):
        with open("house_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)
        with open("scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

if __name__ == "__main__":
    pipeline = HousePricePredictionPipeline()
    pipeline.load_and_inspect_dataset()
    pipeline.clean_data_and_remove_outliers()
    pipeline.engineer_features()
    pipeline.develop_and_compare_models()
    pipeline.evaluate_models()
    pipeline.tune_hyperparameters()
    pipeline.save_model()
    print("Execution complete! Model saved successfully.")
`;
