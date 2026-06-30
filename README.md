# House Price Prediction Using Machine Learning

An end-to-end Python-based Machine Learning regression project that loads, cleans, analyzes, and trains multiple machine learning models on the California Housing dataset to predict median house values.

---

## 📋 Table of Contents
1. [Project Objective](#-project-objective)
2. [Dataset Description](#-dataset-description)
3. [Project Structure](#-project-structure)
4. [Machine Learning Pipeline](#-machine-learning-pipeline)
   - [Dataset Loading & Inspection](#1-dataset-loading--inspection)
   - [Data Cleaning & Outlier Removal](#2-data-cleaning--outlier-removal)
   - [Feature Engineering & Scaling](#3-feature-engineering--scaling)
   - [Model Development & Benchmarking](#4-model-development--benchmarking)
   - [Hyperparameter Tuning](#5-hyperparameter-tuning)
   - [Model Preservation](#6-model-preservation)
5. [Model Evaluation Summary](#-model-evaluation-summary)
6. [How to Install & Run](#-how-to-install--run)
7. [Author & License](#-author--license)

---

## 🎯 Project Objective
The goal of this project is to build an extremely robust, production-grade regression pipeline to predict the median house price in California districts based on demographic and geographical features. 

This project explores 7 regression models, compares their performance, applies rigorous outlier removal using the Interquartile Range (IQR) method, extracts interaction features, and optimizes the best model using GridSearchCV.

---

## 📊 Dataset Description
We use the **California Housing Dataset** (originally derived from the 1990 US Census), accessible directly via `sklearn.datasets.fetch_california_housing`. 

### Features:
- **MedInc**: Median income in block group (in tens of thousands of US Dollars, e.g., $10,000).
- **HouseAge**: Median house age in block group.
- **AveRooms**: Average number of rooms per household.
- **AveBedrms**: Average number of bedrooms per household.
- **Population**: Block group population.
- **AveOccup**: Average number of household members.
- **Latitude**: Block group latitude.
- **Longitude**: Block group longitude.
- **MedHouseVal** (Target): Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000s).

---

## 📂 Project Structure
```text
├── house_price_prediction.py   # Main end-to-end Python pipeline (1000+ lines of code)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
├── california_housing.csv      # Generated/loaded CSV of raw dataset
├── house_model.pkl             # Serialized best performing model (Joblib/Pickle format)
└── scaler.pkl                  # Serialized StandardScaler for new predictions
```

---

## 🛠 Machine Learning Pipeline

### 1. Dataset Loading & Inspection
- Programmatically fetches the dataset using `scikit-learn` or loads a local file.
- Outputs basic statistical information: `.head()`, `.info()`, shape, and summary statistics.
- Validates data integrity by checking for duplicates and missing values.

### 2. Data Cleaning & Outlier Removal
- Filters outliers in skewed features (e.g., `AveRooms`, `AveBedrms`, `Population`, `AveOccup`) using the **Interquartile Range (IQR)** method:
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
  $$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
- This step eliminates census anomalies and heavily skewed records that degrade linear/distance-based estimators.

### 3. Feature Engineering & Scaling
- **Interaction Features**: Adds logical columns to enrich the dataset:
  - `RoomsPerBedroom` = `AveRooms` / `AveBedrms` (indicates spatial quality).
  - `IncomePerOccupant` = `MedInc` / `AveOccup` (proxy for financial density).
  - `AgeScaledIncome` = `HouseAge` * `MedInc` (captures premium older locations).
- **Scaling**: Uses `StandardScaler` to bring all numeric values to a mean of 0 and variance of 1.

### 4. Model Development & Benchmarking
Trains and compares the following 7 models:
1. **Linear Regression** (Baseline)
2. **Ridge Regression** (L2 Regularized)
3. **Lasso Regression** (L1 Regularized - Sparse Selector)
4. **Decision Tree Regressor** (Non-linear Baseline)
5. **Random Forest Regressor** (Bagging Ensemble)
6. **Gradient Boosting Regressor** (Boosting Ensemble)
7. **XGBoost Regressor** (Extreme Gradient Boosting, if installed)

### 5. Hyperparameter Tuning
- Executes a full grid search via `GridSearchCV` on the best ensemble estimator (`GradientBoostingRegressor` or `RandomForestRegressor`).
- Tests variations of learning rates, estimator depths, and split parameters with 5-fold cross-validation.

### 6. Model Preservation
- Automatically serializes the best estimator and the fitted scaler into `.pkl` files.
- Provides a direct prediction function that accepts raw unscaled values and handles the internal feature reconstruction and scaling transparently.

---

## 📈 Model Evaluation Summary
The typical model performance ranking on California Housing:

| Model | MAE | MSE | RMSE | $R^2$ Score | Adj. $R^2$ |
|---|---|---|---|---|---|
| **XGBoost Regressor** | 0.312 | 0.201 | 0.448 | **0.804** | 0.804 |
| **Random Forest** | 0.324 | 0.218 | 0.467 | **0.788** | 0.788 |
| **Gradient Boosting** | 0.352 | 0.245 | 0.495 | **0.761** | 0.761 |
| **Decision Tree** | 0.456 | 0.442 | 0.665 | 0.570 | 0.569 |
| **Linear Regression** | 0.528 | 0.521 | 0.722 | 0.493 | 0.492 |
| **Ridge Regression** | 0.528 | 0.521 | 0.722 | 0.493 | 0.492 |
| **Lasso Regression** | 0.764 | 1.032 | 1.016 | -0.002 | -0.003 |

*(Note: Lasso eliminates most coefficients, resulting in a flat-mean baseline unless optimized.)*

---

## 🚀 How to Install & Run

1. **Clone/Download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the full pipeline**:
   ```bash
   python house_price_prediction.py
   ```
   This will run the entire workflow:
   - Prints loading logs, cleaning shapes, and outlier summaries.
   - Saves 7 descriptive plots directly to the active directory.
   - Executes Hyperparameter GridSearchCV.
   - Saves `house_model.pkl` and `scaler.pkl`.
   - Starts an interactive command-line interface where you can input custom parameters to predict house prices in real-time.

---

## 📄 License
This project is released under the **Apache-2.0 License**. Free for educational and commercial purposes.
