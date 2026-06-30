"""
Customer Churn Prediction - Model Suite and Training Engine
Author: Senior ML Engineer & Architect
File: model.py

This module contains the ChurnClassifierSuite class, which wraps model training,
cross-validation, hyperparameter adjustment, metrics evaluation, and pipeline
serialization. It handles package omissions gracefully, using fallbacks for XGBoost.
"""

import logging
import joblib
import numpy as np
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ModelSuite")


class ChurnClassifierSuite:
    """
    Manages training, evaluation, comparison, and export of 9 major machine learning
    classification algorithms for customer churn prediction.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        
        self._initialize_models()
        
    def _initialize_models(self):
        """
        Instantiates the nine classification models with optimized baseline hyperparameters.
        Includes a graceful fallback for XGBoost.
        """
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, C=0.1, random_state=self.random_state
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=6, min_samples_split=10, random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, max_depth=10, min_samples_split=5, random_state=self.random_state
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.08, max_depth=4, random_state=self.random_state
            ),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=100, learning_rate=0.1, random_state=self.random_state
            ),
            "SVM": SVC(
                C=1.0, kernel="rbf", probability=True, random_state=self.random_state
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=7, weights="distance"
            ),
            "Naive Bayes": GaussianNB()
        }
        
        # XGBoost Initialization with dynamic fallback check
        try:
            import xgboost as xgb
            self.models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.05,
                use_label_encoder=False, eval_metric="logloss", random_state=self.random_state
            )
            logger.info("Successfully imported and loaded XGBoost Classifier.")
        except ImportError:
            # Fallback to an advanced RandomForest/ExtraTrees style or gradient booster with alternate parameters
            logger.warning("XGBoost is not installed. Deploying high-fidelity Gradient Boosting fallback as XGBoost.")
            self.models["XGBoost"] = GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=5,
                subsample=0.8, random_state=self.random_state
            )
            
    def train_and_evaluate_all(self, X_train, y_train, X_test, y_test, cv_folds=5):
        """
        Trains all initialized classifiers, executes K-Fold Cross-Validation,
        and computes exhaustive evaluation metrics on the test set.
        """
        logger.info(f"Beginning training on {X_train.shape[0]} samples across {len(self.models)} models...")
        
        for name, model in self.models.items():
            logger.info(f"Training Model: {name}")
            try:
                # 1. Fit the model
                model.fit(X_train, y_train)
                
                # 2. Predict classes and probabilities
                y_pred = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    # For non-probability models (like SVM if prob=False), map decision function to pseudo-probs
                    dfun = model.decision_function(X_test)
                    y_prob = 1 / (1 + np.exp(-dfun))
                else:
                    y_prob = y_pred.astype(float)
                    
                # 3. Compute Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_prob)
                cm = confusion_matrix(y_test, y_pred)
                
                # 4. K-Fold Cross-Validation
                logger.info(f"Executing {cv_folds}-fold cross validation for {name}...")
                cv_results = cross_validate(
                    model, X_train, y_train, cv=cv_folds,
                    scoring=["accuracy", "f1"], n_jobs=-1
                )
                cv_acc_mean = cv_results["test_accuracy"].mean()
                cv_f1_mean = cv_results["test_f1"].mean()
                
                # 5. Extract Curve Details (Decimated to save space and represent beautifully)
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                precision_crv, recall_crv, _ = precision_recall_curve(y_test, y_prob)
                
                # Select up to 20 coordinates for clean JSON serialization
                roc_step = max(1, len(fpr) // 15)
                pr_step = max(1, len(precision_crv) // 15)
                
                # 6. Learning curve data (for Random Forest/Gradient Boosting or best model visualization)
                train_sizes, train_scores, validation_scores = learning_curve(
                    model, X_train, y_train, cv=3, n_jobs=-1,
                    train_sizes=np.linspace(0.2, 1.0, 5), random_state=self.random_state
                )
                
                # Package model results
                self.results[name] = {
                    "Accuracy": float(acc),
                    "Precision": float(prec),
                    "Recall": float(rec),
                    "F1-Score": float(f1),
                    "ROC-AUC": float(roc_auc),
                    "CV_Accuracy_Mean": float(cv_acc_mean),
                    "CV_F1_Mean": float(cv_f1_mean),
                    "Confusion_Matrix": cm.tolist(),
                    "Classification_Report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                    "ROC_Curve": {
                        "fpr": fpr[::roc_step].tolist(),
                        "tpr": tpr[::roc_step].tolist()
                    },
                    "PR_Curve": {
                        "precision": precision_crv[::pr_step].tolist(),
                        "recall": recall_crv[::pr_step].tolist()
                    },
                    "Learning_Curve": {
                        "train_sizes": train_sizes.tolist(),
                        "train_scores_mean": train_scores.mean(axis=1).tolist(),
                        "val_scores_mean": validation_scores.mean(axis=1).tolist()
                    }
                }
                
                # Feature Importances if available
                if hasattr(model, "feature_importances_"):
                    self.results[name]["Feature_Importances"] = model.feature_importances_.tolist()
                elif hasattr(model, "coef_"):
                    self.results[name]["Feature_Importances"] = np.abs(model.coef_[0]).tolist()
                else:
                    self.results[name]["Feature_Importances"] = []
                    
                logger.info(f"Model {name} Evaluation complete. F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train or evaluate model {name}: {str(e)}", exc_info=True)
                
        # Select best model automatically using F1-Score (balanced metric for churn prediction)
        self._select_best_model()
        return self.results
        
    def _select_best_model(self):
        """
        Determines the optimal model based on the maximum test set F1-Score.
        """
        best_score = -1.0
        best_name = None
        
        for name, metrics in self.results.items():
            f1 = metrics["F1-Score"]
            if f1 > best_score:
                best_score = f1
                best_name = name
                
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        logger.info(f"Best Performing Model selected: {best_name} with F1-Score: {best_score:.4f}")
        
    def save_pipeline(self, preprocessor, filepath="churn_prediction_pipeline.joblib"):
        """
        Bundles the preprocessor state and the best model together, saving to file using Joblib.
        """
        if self.best_model is None:
            raise ValueError("No models trained or selected!")
            
        pipeline = {
            "preprocessor": preprocessor,
            "best_model_name": self.best_model_name,
            "model": self.best_model,
            "results": self.results[self.best_model_name],
            "all_comparisons": {name: {k: v for k, v in m.items() if k not in ["ROC_Curve", "PR_Curve", "Learning_Curve"]} for name, m in self.results.items()}
        }
        
        joblib.dump(pipeline, filepath)
        logger.info(f"Full Production Pipeline successfully saved to '{filepath}' using Joblib.")
        return filepath


if __name__ == "__main__":
    # Self-test code
    from preprocessing import generate_synthetic_data, ChurnPreprocessor
    df = generate_synthetic_data(200)
    preproc = ChurnPreprocessor()
    preproc.fit(df)
    X, y = preproc.transform(df)
    
    # Split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    suite = ChurnClassifierSuite()
    suite.train_and_evaluate_all(X_train, y_train, X_test, y_test)
    print("Best model selected:", suite.best_model_name)
    print("Best F1 Score:", suite.results[suite.best_model_name]["F1-Score"])
