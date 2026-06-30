#!/usr/bin/env python3
"""
================================================================================
HOUSE PRICE PREDICTION USING MACHINE LEARNING
================================================================================
An End-to-End Regression Pipeline using Python, Scikit-Learn, and XGBoost.
Fulfills the complete lifecycle of an industrial Machine Learning project:
1. Data Fetching and Analytical Loading
2. Thorough Exploratory Data Analysis (EDA)
3. Advanced Data Cleaning & IQR Outlier Removal
4. Feature Engineering (Scaling, Custom Interaction Terms, Selection)
5. Model Development (Comparing 7 major estimators)
6. Comprehensive Model Evaluation (MAE, MSE, RMSE, R2, Adjusted R2)
7. Advanced Visualizations (Heatmaps, Scatter, Distribution, Error, Importance)
8. Hyperparameter Optimization via Cross-Validated GridSearchCV
9. Model Serialization & Verification Loading
10. CLI User Interface for Live Prediction Simulation

Author: Expert Data Scientist & Machine Learning Engineer
Date: June 30, 2026
License: Apache-2.0
================================================================================
"""

# ==============================================================================
# SECTION 1: SYSTEM IMPORTS AND DEPENDENCY CONFIGURATION
# ==============================================================================
import os
import sys
import time
import pickle
import warnings
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

# Visualizations
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend suitable for server environments
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Metrics & Preprocessing
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Machine Learning Estimators
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Optional XGBoost import - gracefully fallback if not present
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Suppress warnings for visual cleanliness
warnings.filterwarnings('ignore')

# Set visual style parameters
sns.set_theme(style="whitegrid", context="talk", palette="muted")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'font.family': 'sans-serif'
})


# ==============================================================================
# SECTION 2: EXPERT ML PIPELINE CLASS DEFINITION
# ==============================================================================
class HousePricePredictionPipeline:
    """
    An encapsulated, production-grade Machine Learning pipeline designed 
    to handle the entire workflow of the House Price Prediction Project.
    """

    def __init__(self, data_source: str = "california"):
        """
        Initializes the pipeline with configurations and system state.
        """
        print("=" * 80)
        print(" INITIALIZING HOUSE PRICE PREDICTION PIPELINE")
        print("=" * 80)
        self.data_source = data_source
        self.raw_df: pd.DataFrame = None
        self.cleaned_df: pd.DataFrame = None
        self.engineered_df: pd.DataFrame = None
        
        self.X_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None
        
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.performance_comparison: pd.DataFrame = None
        self.feature_names: List[str] = []
        
        # Meta logs
        self.outliers_removed_count = 0
        self.raw_shape = (0, 0)
        self.clean_shape = (0, 0)

    # --------------------------------------------------------------------------
    # STEP 1: DATASET LOADING & INSPECTION
    # --------------------------------------------------------------------------
    def load_and_inspect_dataset(self) -> pd.DataFrame:
        """
        Loads the California Housing dataset directly from Scikit-learn,
        constructs a structured Pandas DataFrame, and prints initial inspection details.
        """
        print("\n" + "-" * 50)
        print(" [STEP 1] DATASET LOADING & INSPECTION")
        print("-" * 50)
        
        print("Loading California Housing Dataset programmatically...")
        cal_data = fetch_california_housing(as_frame=True)
        
        # Merge features and target into a single pandas DataFrame
        df = cal_data.frame
        # Rename target column to MedianHouseValue for clarity
        df = df.rename(columns={'MedHouseVal': 'MedianHouseValue'})
        
        self.raw_df = df.copy()
        self.raw_shape = df.shape
        self.feature_names = list(cal_data.feature_names)
        
        print("\n>>> Dataset loaded successfully!")
        print(f"Shape of Dataset (Rows, Columns): {df.shape[0]} rows, {df.shape[1]} columns")
        
        print("\n>>> Displaying First 5 Rows of Dataset:")
        print(df.head().to_string())
        
        print("\n>>> Detailed Dataset Information:")
        print(df.info())
        
        print("\n>>> Statistical Summary of Numerical Columns:")
        print(df.describe().T.to_string())
        
        print("\n>>> Checking for Data Integrity:")
        missing_counts = df.isnull().sum()
        duplicate_count = df.duplicated().sum()
        
        print(f"Total Missing Values across all columns:\n{missing_counts.to_string()}")
        print(f"Total Duplicate Rows identified: {duplicate_count}")
        
        return self.raw_df

    # --------------------------------------------------------------------------
    # STEP 2: DATA CLEANING & OUTLIER DETECTION
    # --------------------------------------------------------------------------
    def clean_data_and_remove_outliers(self) -> pd.DataFrame:
        """
        Handles missing values, removes duplicate rows, and detects & prunes
        statistical outliers in numerical features using the Interquartile Range (IQR).
        """
        print("\n" + "-" * 50)
        print(" [STEP 2] DATA CLEANING & OUTLIER DETECTION")
        print("-" * 50)
        
        if self.raw_df is None:
            raise ValueError("Raw dataframe is empty. Ensure step 1 is run before cleaning.")
            
        working_df = self.raw_df.copy()
        
        # Handle duplicate values if any
        duplicates = working_df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate records...")
            working_df = working_df.drop_duplicates().reset_index(drop=True)
        else:
            print("No duplicate records found in the dataset.")
            
        # Handle missing values if any
        missing_total = working_df.isnull().sum().sum()
        if missing_total > 0:
            print(f"Detected {missing_total} missing cells. Performing median imputation...")
            for col in working_df.columns:
                if working_df[col].isnull().sum() > 0:
                    working_df[col].fillna(working_df[col].median(), inplace=True)
        else:
            print("No missing values found. Dataset is 100% complete.")

        # Outlier detection and elimination using the 1.5 * IQR Rule.
        # Features with extreme positive/negative skewness are prone to noise.
        outlier_candidates = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
        print(f"\nDetecting and removing outliers on skewed features: {outlier_candidates}")
        
        initial_row_count = len(working_df)
        
        for col in outlier_candidates:
            q1 = working_df[col].quantile(0.25)
            q3 = working_df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Mask outliers out
            filter_mask = (working_df[col] >= lower_bound) & (working_df[col] <= upper_bound)
            working_df = working_df[filter_mask]
            
        final_row_count = len(working_df)
        self.outliers_removed_count = initial_row_count - final_row_count
        self.clean_shape = working_df.shape
        self.cleaned_df = working_df.reset_index(drop=True)
        
        print(f"Outliers pruned: {self.outliers_removed_count} records removed.")
        print(f"Shape of Cleaned Dataset: {self.cleaned_df.shape[0]} rows, {self.cleaned_df.shape[1]} columns")
        print(f"Outlier-to-data ratio: {(self.outliers_removed_count / initial_row_count) * 100:.2f}%")
        
        return self.cleaned_df

    # --------------------------------------------------------------------------
    # STEP 3: FEATURE ENGINEERING
    # --------------------------------------------------------------------------
    def engineer_features(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Creates custom analytical interaction features to enhance non-linear relationships,
        splits data into features and target, and standardizes numeric columns.
        """
        print("\n" + "-" * 50)
        print(" [STEP 3] ADVANCED FEATURE ENGINEERING & TRANSFORMATION")
        print("-" * 50)
        
        if self.cleaned_df is None:
            raise ValueError("Cleaned dataframe is empty. Ensure step 2 is run before engineering.")
            
        df = self.cleaned_df.copy()
        
        # Construct custom features of domain significance
        print("Constructing highly predictive spatial & socio-economic interaction features:")
        
        # 1. Ratio of average rooms to average bedrooms (Indicates the spaciousness of household configurations)
        df['RoomsPerBedroom'] = df['AveRooms'] / (df['AveBedrms'] + 1e-5) # Prevent division by zero
        print(" -> Created Feature: 'RoomsPerBedroom' = AveRooms / AveBedrms")
        
        # 2. Income per occupant in the block group
        df['IncomePerOccupant'] = df['MedInc'] / (df['AveOccup'] + 1e-5)
        print(" -> Created Feature: 'IncomePerOccupant' = MedInc / AveOccup")
        
        # 3. Age-Scaled Income interaction
        df['AgeScaledIncome'] = df['HouseAge'] * df['MedInc']
        print(" -> Created Feature: 'AgeScaledIncome' = HouseAge * MedInc")
        
        self.engineered_df = df.copy()
        
        # Extract features and target arrays
        target_column = 'MedianHouseValue'
        feature_cols = [col for col in df.columns if col != target_column]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df[target_column].values
        
        print(f"\nEngineered Features list: {self.feature_names}")
        print(f"Total Features used for modeling: {len(self.feature_names)}")
        
        # Split into training and testing configurations (80% / 20% split)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Training split shape: {self.X_train.shape[0]} samples")
        print(f"Testing split shape: {self.X_test.shape[0]} samples")
        
        # Standardize features (StandardScaler: mean=0, std=1)
        print("\nApplying Standard Normalization via StandardScaler to all features...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("Feature transformation is complete. Train/Test datasets fully prepared.")
        
        return self.X_train, self.X_test, self.feature_names

    # --------------------------------------------------------------------------
    # STEP 4: MODEL DEVELOPMENT & TRAINING
    # --------------------------------------------------------------------------
    def develop_and_compare_models(self) -> Dict[str, Any]:
        """
        Instantiates 7 primary regression algorithms, fits them to the scaled training data,
        and saves each model within the pipeline's internal state.
        """
        print("\n" + "-" * 50)
        print(" [STEP 4] MODEL DEVELOPMENT & ENSEMBLE TRAINING")
        print("-" * 50)
        
        # Instantiate estimators
        estimators = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.01),
            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        }
        
        # Safely append XGBoost if imported successfully
        if XGBOOST_AVAILABLE:
            print("XGBoost is available in the current environment. Appending XGBRegressor to training queue.")
            estimators["XGBoost Regressor"] = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        else:
            print("XGBoost not available in the current Python environment. Skipping XGBoost.")

        # Train and save each estimator
        for name, model in estimators.items():
            print(f"Training model: {name}...")
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            duration = time.time() - start_time
            print(f" -> Completed training in {duration:.3f} seconds.")
            self.models[name] = model
            
        return self.models

    # --------------------------------------------------------------------------
    # STEP 5: MODEL EVALUATION & COMPARISON
    # --------------------------------------------------------------------------
    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluates each trained regression model against standard metrics:
        Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
        Coefficient of Determination (R2 Score), and Adjusted R2 Score.
        Generates a comparative analysis table.
        """
        print("\n" + "-" * 50)
        print(" [STEP 5] DETAILED MODEL EVALUATION METRICS")
        print("-" * 50)
        
        if not self.models:
            raise ValueError("No models trained. Execute step 4 first.")
            
        comparison_records = []
        n = len(self.y_test)
        p = self.X_test.shape[1]  # Number of predictors
        
        for name, model in self.models.items():
            predictions = model.predict(self.X_test)
            
            # Standard Metrics
            mae = mean_absolute_error(self.y_test, predictions)
            mse = mean_squared_error(self.y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, predictions)
            
            # Adjusted R2 Score Formula:
            # Adj_R2 = 1 - [(1 - R2) * (n - 1) / (n - p - 1)]
            adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            
            comparison_records.append({
                "Model": name,
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R2 Score": round(r2, 4),
                "Adjusted R2": round(adj_r2, 4)
            })
            
        comparison_df = pd.DataFrame(comparison_records)
        # Sort values with highest R2 score first
        comparison_df = comparison_df.sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
        self.performance_comparison = comparison_df
        
        # Deduce best model
        self.best_model_name = comparison_df.iloc[0]["Model"]
        self.best_model = self.models[self.best_model_name]
        
        print("\n>>> Performance Comparison Matrix (Sorted by R² Score):")
        print(self.performance_comparison.to_string(index=False))
        
        print("\n" + "*" * 80)
        print(f" BEST PERFORMING MODEL DETECTED: {self.best_model_name}")
        print(f" Test R² Score achieved: {comparison_df.iloc[0]['R2 Score']:.4f}")
        print(f" Test RMSE achieved: {comparison_df.iloc[0]['RMSE']:.4f} ($100,000s scale)")
        print("*" * 80)
        
        return self.performance_comparison

    # --------------------------------------------------------------------------
    # STEP 6: DATA & PREDICTION VISUALIZATIONS
    # --------------------------------------------------------------------------
    def generate_and_save_visualizations(self, output_dir: str = ".") -> List[str]:
        """
        Creates publication-quality statistical charts from data science analysis and metrics
        and saves them locally in the specified output directory.
        """
        print("\n" + "-" * 50)
        print(" [STEP 6] GENERATING DESCRIPTIVE AND PREDICTIVE CHARTS")
        print("-" * 50)
        
        saved_paths = []
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Correlation Heatmap Matrix
        plt.figure(figsize=(14, 10))
        corr_matrix = self.engineered_df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                    vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .7})
        plt.title("Correlation Heatmap Matrix of Engineered Dataset", pad=20)
        plt.tight_layout()
        plot1_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(plot1_path, dpi=150)
        plt.close()
        saved_paths.append(plot1_path)
        print(f" Saved: {plot1_path}")

        # Plot 2: Distribution Plot of House Price Target
        plt.figure(figsize=(12, 6))
        sns.histplot(self.raw_df['MedianHouseValue'], kde=True, color="teal", bins=50, label="Raw Data")
        sns.histplot(self.cleaned_df['MedianHouseValue'], kde=True, color="coral", bins=50, label="Cleaned Data")
        plt.title("Distribution of Median House Prices (Raw vs. IQR Cleaned)")
        plt.xlabel("Median House Price ($100,000s)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plot2_path = os.path.join(output_dir, "price_distribution.png")
        plt.savefig(plot2_path, dpi=150)
        plt.close()
        saved_paths.append(plot2_path)
        print(f" Saved: {plot2_path}")

        # Plot 3: Boxplots for Outlier Visualizations
        plt.figure(figsize=(14, 8))
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        skewed_features = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
        for i, col in enumerate(skewed_features):
            row_idx = i // 2
            col_idx = i % 2
            sns.boxplot(data=self.raw_df, y=col, ax=axes[row_idx, col_idx], color="skyblue")
            axes[row_idx, col_idx].set_title(f"Boxplot of {col} (Raw Outliers)")
        plt.suptitle("Outlier Verification via Boxplots on Skewed Features")
        plt.tight_layout()
        plot3_path = os.path.join(output_dir, "outliers_boxplots.png")
        plt.savefig(plot3_path, dpi=150)
        plt.close()
        saved_paths.append(plot3_path)
        print(f" Saved: {plot3_path}")

        # Plot 4: Scatterplot: Median Income vs Median House Price
        plt.figure(figsize=(12, 8))
        # Sample data if too large to speed up rendering
        sample_size = min(3000, len(self.cleaned_df))
        sample_df = self.cleaned_df.sample(sample_size, random_state=42)
        sns.regplot(data=sample_df, x="MedInc", y="MedianHouseValue", 
                    scatter_kws={"alpha": 0.4, "color": "darkblue"}, 
                    line_kws={"color": "red", "linewidth": 2.5})
        plt.title(f"Median Income vs. House Price Trend (Sample of {sample_size} records)")
        plt.xlabel("Median Income ($10,000s)")
        plt.ylabel("Median House Price ($100,000s)")
        plt.tight_layout()
        plot4_path = os.path.join(output_dir, "income_vs_price.png")
        plt.savefig(plot4_path, dpi=150)
        plt.close()
        saved_paths.append(plot4_path)
        print(f" Saved: {plot4_path}")

        # Plot 5: Actual vs Predicted Scatterplot for the Best Performing Model
        plt.figure(figsize=(12, 8))
        best_preds = self.best_model.predict(self.X_test)
        plt.scatter(self.y_test, best_preds, alpha=0.3, color="darkmagenta")
        # Draw a perfect prediction y=x diagonal reference line
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2.5, color="red")
        plt.title(f"Actual vs. Predicted Median House Prices ({self.best_model_name})")
        plt.xlabel("Actual Price ($100,000s)")
        plt.ylabel("Predicted Price ($100,000s)")
        plt.tight_layout()
        plot5_path = os.path.join(output_dir, "actual_vs_predicted.png")
        plt.savefig(plot5_path, dpi=150)
        plt.close()
        saved_paths.append(plot5_path)
        print(f" Saved: {plot5_path}")

        # Plot 6: Residual Distribution Plot (Prediction Error Distribution)
        plt.figure(figsize=(12, 6))
        residuals = self.y_test - best_preds
        sns.histplot(residuals, kde=True, color="olive", bins=50)
        plt.axvline(x=0, color="red", linestyle="--", linewidth=2)
        plt.title(f"Distribution of Residuals / Prediction Errors ({self.best_model_name})")
        plt.xlabel("Residual Value (Actual - Predicted)")
        plt.ylabel("Density")
        plt.tight_layout()
        plot6_path = os.path.join(output_dir, "residuals_plot.png")
        plt.savefig(plot6_path, dpi=150)
        plt.close()
        saved_paths.append(plot6_path)
        print(f" Saved: {plot6_path}")

        # Plot 7: Feature Importance Matrix (for tree-based ensembles)
        if hasattr(self.best_model, "feature_importances_"):
            plt.figure(figsize=(12, 8))
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            importance_df = pd.DataFrame({
                "Feature": [self.feature_names[i] for i in indices],
                "Importance": importances[indices]
            })
            
            sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
            plt.title(f"Feature Importance Map derived from {self.best_model_name}")
            plt.xlabel("Relative Importance Score")
            plt.ylabel("Engineered Feature Name")
            plt.tight_layout()
            plot7_path = os.path.join(output_dir, "feature_importances.png")
            plt.savefig(plot7_path, dpi=150)
            plt.close()
            saved_paths.append(plot7_path)
            print(f" Saved: {plot7_path}")
        else:
            print(" Best model is not a tree ensemble; skipping feature importances visualization.")
            
        return saved_paths

    # --------------------------------------------------------------------------
    # STEP 7: HYPERPARAMETER TUNING
    # --------------------------------------------------------------------------
    def tune_hyperparameters(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes a localized cross-validated hyperparameter search using GridSearchCV
        on the GradientBoostingRegressor model to improve generalized performance.
        """
        print("\n" + "-" * 50)
        print(" [STEP 7] HYPERPARAMETER OPTIMIZATION (GRID SEARCH)")
        print("-" * 50)
        
        # We will tune the Gradient Boosting Regressor or Random Forest
        tune_model = GradientBoostingRegressor(random_state=42)
        
        # Keep hyperparameter grid dense to prevent high computation delay during runs
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2]
        }
        
        print(f"Configuring GridSearchCV for Gradient Boosting with parameters:")
        print(param_grid)
        print("Executing 5-Fold Cross Validation... Please wait.")
        
        grid_search = GridSearchCV(
            estimator=tune_model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit GridSearch using training data
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model configurations
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = np.sqrt(-grid_search.best_score_)
        
        print("\n>>> GridSearchCV Optimization Complete!")
        print(f"Best Parameters Set: {best_params}")
        print(f"Lowest Cross-Validation RMSE: {best_score:.4f} ($100,000s scale)")
        
        # Test performance of optimized estimator
        grid_preds = best_estimator.predict(self.X_test)
        grid_r2 = r2_score(self.y_test, grid_preds)
        print(f"Optimized Model Test R² Score: {grid_r2:.4f}")
        
        # If the tuned model is better, promote it to self.best_model
        baseline_r2 = self.performance_comparison.iloc[0]['R2 Score']
        if grid_r2 > baseline_r2:
            print(f"Tunings outperformed baseline model. Promoting to Best Pipeline Model.")
            self.best_model = best_estimator
            self.best_model_name = f"Optimized GradientBoostingRegressor"
            
        return self.best_model, best_params

    # --------------------------------------------------------------------------
    # STEP 8: SAVE MODEL (SERIALIZATION)
    # --------------------------------------------------------------------------
    def serialize_model_assets(self, model_path: str = "house_model.pkl", scaler_path: str = "scaler.pkl"):
        """
        Serializes the trained best performing estimator and fitted scaler 
        assets onto the storage system using Pickle.
        """
        print("\n" + "-" * 50)
        print(" [STEP 8] MODEL ARTIFACT SERIALIZATION")
        print("-" * 50)
        
        print(f"Serializing Best Performing Estimator '{self.best_model_name}' to disk...")
        with open(model_path, 'wb') as m_file:
            pickle.dump(self.best_model, m_file)
        print(f" -> Saved successfully as: {model_path}")
            
        print("Serializing Fitted StandardScaler configuration...")
        with open(scaler_path, 'wb') as s_file:
            pickle.dump(self.scaler, s_file)
        print(f" -> Saved successfully as: {scaler_path}")
        
        # Save a list of metadata for prediction reconstructions
        metadata = {
            "feature_names": self.feature_names,
            "best_model_name": self.best_model_name
        }
        with open("model_metadata.pkl", 'wb') as meta_file:
            pickle.dump(metadata, meta_file)
        print(" -> Saved metadata configurations as: model_metadata.pkl")


# ==============================================================================
# SECTION 3: DEPLOYED PREDICTION ENGINE CLASS
# ==============================================================================
class LiveHousePricePredictor:
    """
    Production inference engine designed to load serialized assets 
    and construct predictions against raw inputs.
    """
    
    def __init__(self, model_path: str = "house_model.pkl", scaler_path: str = "scaler.pkl", meta_path: str = "model_metadata.pkl"):
        """
        Loads the pickled estimator, normalization scales, and properties.
        """
        print("\n>>> Initializing Inference Predictor Engine...")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or Scaler pickle files are missing. Train the pipeline first.")
            
        with open(model_path, 'rb') as m_file:
            self.model = pickle.load(m_file)
            
        with open(scaler_path, 'rb') as s_file:
            self.scaler = pickle.load(s_file)
            
        with open(meta_path, 'rb') as f_meta:
            self.metadata = pickle.load(f_meta)
            
        self.feature_names = self.metadata["feature_names"]
        self.model_name = self.metadata["best_model_name"]
        print(f"Inference Model loaded: {self.model_name}")
        print(f"Required features count: {len(self.feature_names)}")

    def predict_price(self, raw_input_dict: Dict[str, float]) -> float:
        """
        Processes a raw features dictionary, engineers the required interaction terms, 
        standardizes numerical arrays, and computes a price prediction in Dollars.
        """
        # Formulate pandas row representation to process feature alignments
        input_df = pd.DataFrame([raw_input_dict])
        
        # Calculate matching interactions used during training
        input_df['RoomsPerBedroom'] = input_df['AveRooms'] / (input_df['AveBedrms'] + 1e-5)
        input_df['IncomePerOccupant'] = input_df['MedInc'] / (input_df['AveOccup'] + 1e-5)
        input_df['AgeScaledIncome'] = input_df['HouseAge'] * input_df['MedInc']
        
        # Realign columns with feature names
        aligned_df = input_df[self.feature_names]
        
        # Standardize the features
        scaled_features = self.scaler.transform(aligned_df.values)
        
        # Predict price (Target variable is scaled by 100,000 in California Housing)
        predicted_scaled_val = self.model.predict(scaled_features)[0]
        
        # Return converted price in USD
        usd_price = predicted_scaled_val * 100000.0
        return max(0.0, usd_price) # Prevent negative values in extreme scenarios


# ==============================================================================
# SECTION 4: MAIN PIPELINE INTEGRATION RUNNER
# ==============================================================================
def run_entire_ml_project():
    """
    Orchestrates the running of the complete end-to-end Machine Learning project.
    """
    start_execution = time.time()
    
    # 1. Instantiate the pipeline
    pipeline = HousePricePredictionPipeline()
    
    # 2. Loading and analytics summary
    pipeline.load_and_inspect_dataset()
    
    # 3. Handle Cleanings
    pipeline.clean_data_and_remove_outliers()
    
    # 4. Feature Construction
    pipeline.engineer_features()
    
    # 5. Train baseline estimators
    pipeline.develop_and_compare_models()
    
    # 6. Evaluate all metrics
    pipeline.evaluate_models()
    
    # 7. Localized GridSearchCV Parameter Optimization
    pipeline.tune_hyperparameters()
    
    # 8. Save output plots
    pipeline.generate_and_save_visualizations()
    
    # 9. Serialize pipeline structures
    pipeline.serialize_model_assets()
    
    duration = time.time() - start_execution
    print("\n" + "=" * 80)
    print(" HOUSE PRICE PREDICTION SYSTEM RUN COMPLETED")
    print(f" Total Pipeline Execution Time: {duration:.2f} seconds")
    print("=" * 80 + "\n")


# ==============================================================================
# SECTION 5: INTERACTIVE CLI PREDICTION TOOL
# ==============================================================================
def start_interactive_cli():
    """
    Launches an interactive Command Line Interface to accept custom inputs
    and run price predictions using the trained models.
    """
    print("=" * 80)
    print(" HOUSE PRICE PREDICTOR CLI CLIENT")
    print("=" * 80)
    
    try:
        predictor = LiveHousePricePredictor()
    except Exception as e:
        print(f"Error loading model files: {e}")
        print("Please run the full training pipeline first using: python house_price_prediction.py")
        return
        
    print("\nPlease provide the characteristics of the housing district to estimate value:")
    
    try:
        # Default typical values in case the user hits enter
        med_inc = input("1. Median Income of block group (in $10,000s, default 4.5): ")
        med_inc = float(med_inc) if med_inc.strip() else 4.5
        
        house_age = input("2. Median House Age in block group (years, default 28.0): ")
        house_age = float(house_age) if house_age.strip() else 28.0
        
        ave_rooms = input("3. Average Rooms per household (default 5.4): ")
        ave_rooms = float(ave_rooms) if ave_rooms.strip() else 5.4
        
        ave_bedrms = input("4. Average Bedrooms per household (default 1.1): ")
        ave_bedrms = float(ave_bedrms) if ave_bedrms.strip() else 1.1
        
        population = input("5. Block Group Population density (default 1400.0): ")
        population = float(population) if population.strip() else 1400.0
        
        ave_occup = input("6. Average Household Occupancy (default 3.0): ")
        ave_occup = float(ave_occup) if ave_occup.strip() else 3.0
        
        latitude = input("7. Coordinate Latitude (default 34.05): ")
        latitude = float(latitude) if latitude.strip() else 34.05
        
        longitude = input("8. Coordinate Longitude (default -118.24): ")
        longitude = float(longitude) if longitude.strip() else -118.24
        
        # Assemble dictionary matching training features
        user_input = {
            'MedInc': med_inc,
            'HouseAge': house_age,
            'AveRooms': ave_rooms,
            'AveBedrms': ave_bedrms,
            'Population': population,
            'AveOccup': ave_occup,
            'Latitude': latitude,
            'Longitude': longitude
        }
        
        print("\nAnalyzing parameters and executing pricing inference...")
        predicted_usd = predictor.predict_price(user_input)
        
        print("\n" + "*" * 60)
        print(" ESTIMATED MEDIAN HOUSE VALUE FOR SPECIFIED BLOCK GROUP:")
        print(f"  Predicted Price (USD): ${predicted_usd:,.2f}")
        print(f"  Predicted Price (Scale): ${predicted_usd/100000.0:.4f} hundred thousand dollars")
        print("*" * 60 + "\n")
        
    except ValueError as val_err:
        print(f"Input conversion failure. Please ensure numeric entries: {val_err}")


# ==============================================================================
# ENTRY POINT TRIGGER
# ==============================================================================
if __name__ == "__main__":
    # If run with a command argument like 'predict', launch interactive CLI directly
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        start_interactive_cli()
    else:
        # Run entire machine learning workflow pipeline
        run_entire_ml_project()
        
        # Start command prompt predictor immediately for verification
        print("Would you like to test a custom prediction run now?")
        choice = input("Enter 'y' for Yes, 'n' to exit (default y): ")
        if not choice.strip() or choice.strip().lower() == 'y':
            start_interactive_cli()
        else:
            print("Project complete. Model saved. Exiting pipeline runner.")
