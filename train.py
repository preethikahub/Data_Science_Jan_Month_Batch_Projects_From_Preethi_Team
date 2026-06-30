"""
Customer Churn Prediction - End-to-End Training Pipeline Runner
Author: Senior ML Engineer & Architect
File: train.py

This script implements the full end-to-end Machine Learning training workflow.
It performs dataset generation/loading, executes EDA summaries, applies pre-processing,
handles SMOTE balancing, evaluates 9 classification models, plots the metrics,
and serializes the production assets (joblib pipeline and JSON outputs).
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import custom components
from preprocessing import generate_synthetic_data, ChurnPreprocessor
from model import ChurnClassifierSuite

# Setup Matplotlib for headless server execution (Critical for production runtimes)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TrainingRunner")


def save_visualization_plots(suite, preprocessor, df, plot_dir="static/plots"):
    """
    Generates and saves the required industry-standard ML visuals:
    1. Churn Distribution Plot
    2. Correlation Heatmap
    3. Confusion Matrix Heatmap (for the best model)
    4. ROC Curves (comparing top models)
    5. Precision-Recall Curves
    6. Feature Importances Plot
    7. Learning Curves
    """
    os.makedirs(plot_dir, exist_ok=True)
    best_name = suite.best_model_name
    best_res = suite.results[best_name]
    
    logger.info(f"Generating diagnostic visual plots inside '{plot_dir}'...")
    
    # 1. Churn Distribution (Pie and Bar)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    df["Churn"].value_counts().plot(kind="bar", color=["#4f46e5", "#f43f5e"], edgecolor="black")
    plt.title("Customer Churn Count")
    plt.ylabel("Count")
    plt.xlabel("Churn Status")
    
    plt.subplot(1, 2, 2)
    df["Churn"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["#4f46e5", "#f43f5e"], startangle=90, explode=[0, 0.1])
    plt.title("Churn Proportion")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/churn_distribution.png", dpi=150)
    plt.close()
    
    # 2. Correlation Heatmap (for numerical variables)
    plt.figure(figsize=(8, 6))
    df_clean = df.copy()
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median())
    num_df = df_clean[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
    num_df["Churn_Numeric"] = df_clean["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
    
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    
    # Annotate correlation values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.6 else "white")
            
    plt.title("Numerical Feature Correlation")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/correlation_heatmap.png", dpi=150)
    plt.close()
    
    # 3. Confusion Matrix Heatmap
    cm = np.array(best_res["Confusion_Matrix"])
    plt.figure(figsize=(5, 4))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Retained (0)", "Churned (1)"])
    ax.set_yticklabels(["Retained (0)", "Churned (1)"])
    
    # Annotate CM
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
            
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/confusion_matrix_heatmap.png", dpi=150)
    plt.close()
    
    # 4. ROC Curve (Top 4 models comparison)
    plt.figure(figsize=(7, 5))
    models_to_plot = sorted(suite.results.keys(), key=lambda k: suite.results[k]["ROC-AUC"], reverse=True)[:4]
    for name in models_to_plot:
        curve = suite.results[name]["ROC_Curve"]
        plt.plot(curve["fpr"], curve["tpr"], label=f"{name} (AUC = {suite.results[name]['ROC-AUC']:.3f})")
        
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/roc_curve.png", dpi=150)
    plt.close()
    
    # 5. Precision-Recall Curve
    plt.figure(figsize=(7, 5))
    for name in models_to_plot:
        curve = suite.results[name]["PR_Curve"]
        plt.plot(curve["recall"], curve["precision"], label=f"{name}")
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/precision_recall_curve.png", dpi=150)
    plt.close()
    
    # 6. Feature Importance Plot
    importances = best_res["Feature_Importances"]
    if importances:
        features = preprocessor.feature_columns
        # Pair and sort
        paired = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:12]
        sorted_feats, sorted_imps = zip(*paired)
        
        plt.figure(figsize=(8, 5))
        plt.barh(np.arange(len(sorted_feats)), sorted_imps, color="#4f46e5", edgecolor="black")
        plt.yticks(np.arange(len(sorted_feats)), sorted_feats)
        plt.gca().invert_yaxis()  # Best feature at the top
        plt.xlabel("Relative Importance")
        plt.title(f"Top 12 Features Importance ({best_name})")
        plt.grid(True, axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/feature_importance.png", dpi=150)
        plt.close()
        
    # 7. Learning Curve
    lc = best_res["Learning_Curve"]
    plt.figure(figsize=(7, 4))
    plt.plot(lc["train_sizes"], lc["train_scores_mean"], "o-", color="#f43f5e", label="Training Score")
    plt.plot(lc["train_sizes"], lc["val_scores_mean"], "o-", color="#4f46e5", label="Cross-Validation Score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Score Accuracy")
    plt.title(f"Learning Curve ({best_name})")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/learning_curve.png", dpi=150)
    plt.close()
    
    logger.info("Visual Diagnostic plots generated and saved successfully.")


def run_training_pipeline(data_path="telco_churn_data.csv", output_pipeline="churn_prediction_pipeline.joblib"):
    """
    Main execution loop.
    """
    logger.info("Initializing ML training pipeline...")
    
    # 1. Load or Generate Dataset
    if not os.path.exists(data_path):
        logger.info(f"Dataset '{data_path}' not found.")
        df = generate_synthetic_data(num_samples=1500, random_state=42)
        df.to_csv(data_path, index=False)
        logger.info(f"Created a realistic mock Customer Churn dataset of 1500 records saved at '{data_path}'.")
    else:
        logger.info(f"Loading existing customer dataset from '{data_path}'...")
        df = pd.read_csv(data_path)
        
    # 2. EDA analysis
    preprocessor = ChurnPreprocessor()
    eda_summary = preprocessor.perform_eda_summary(df)
    
    # 3. Fit preprocessor
    preprocessor.fit(df)
    X, y = preprocessor.transform(df)
    
    # 4. Oversample to handle class imbalance
    X_balanced, y_balanced = preprocessor.handle_class_imbalance_smote(X, y)
    
    # 5. Split into training and testing partitions
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    logger.info(f"Training partition shape: {X_train.shape}")
    logger.info(f"Test partition shape: {X_test.shape}")
    
    # 6. Train all models
    suite = ChurnClassifierSuite()
    all_results = suite.train_and_evaluate_all(X_train, y_train, X_test, y_test)
    
    # Print terminal-friendly comparison table
    print("\n" + "="*80)
    print("CUSTOMER CHURN MACHINE LEARNING MODEL BENCHMARK".center(80))
    print("="*80)
    headers = ["Model Name", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    print(f"{headers[0]:<25} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10}")
    print("-"*80)
    for name, r in all_results.items():
        print(f"{name:<25} {r['Accuracy']:<10.4f} {r['Precision']:<10.4f} {r['Recall']:<10.4f} {r['F1-Score']:<10.4f} {r['ROC-AUC']:<10.4f}")
    print("="*80)
    print(f"Optimal Model Selected: {suite.best_model_name} (F1: {all_results[suite.best_model_name]['F1-Score']:.4f})")
    print("="*80 + "\n")
    
    # 7. Generate diagnostic plots
    save_visualization_plots(suite, preprocessor, df)
    
    # 8. Save production-ready binary using Joblib
    suite.save_pipeline(preprocessor, output_pipeline)
    
    # 9. Save JSON results for front-end visual mapping
    cleaned_results = {}
    for model_name, metrics in all_results.items():
        cleaned_results[model_name] = {
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "ROC-AUC": metrics["ROC-AUC"],
            "CV_Accuracy_Mean": metrics["CV_Accuracy_Mean"],
            "CV_F1_Mean": metrics["CV_F1_Mean"],
            "Confusion_Matrix": metrics["Confusion_Matrix"],
            "Feature_Importances": metrics["Feature_Importances"],
            "Classification_Report": metrics["Classification_Report"]
        }
        
    training_summary = {
        "eda_summary": {
            "total_customers": eda_summary["shape"][0],
            "total_features": eda_summary["shape"][1],
            "duplicates": eda_summary["duplicates"],
            "missing_values_charges": eda_summary["empty_string_charges"],
            "churn_rate": eda_summary["target_distribution"].get("Yes", 0.0),
            "retained_count": eda_summary["churn_count"].get("No", 0),
            "churn_count": eda_summary["churn_count"].get("Yes", 0),
            "averages": {
                "tenure": eda_summary["numerical_summary"]["tenure"]["mean"],
                "monthly_charges": eda_summary["numerical_summary"]["MonthlyCharges"]["mean"],
                "total_charges": eda_summary["numerical_summary"]["TotalCharges"]["mean"]
            }
        },
        "best_model": suite.best_model_name,
        "features": preprocessor.feature_columns,
        "model_comparisons": cleaned_results
    }
    
    os.makedirs("static", exist_ok=True)
    with open("static/results.json", "w") as f:
        json.dump(training_summary, f, indent=4)
    logger.info("Saved visual summary payload 'static/results.json' successfully.")


if __name__ == "__main__":
    run_training_pipeline()
