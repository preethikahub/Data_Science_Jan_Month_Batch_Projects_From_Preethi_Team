# Customer Churn Prediction Using Machine Learning
A production-grade, end-to-end Machine Learning classification and decision intelligence system. It features Exploratory Data Analysis (EDA), robust pre-processing, imbalance correction (SMOTE), nine trained models benchmarks, and a responsive web dashboard.

---

## 📂 Project Architecture and Structure
The codebase follows senior software engineering best practices with full modularity, object-oriented pipelines, extensive error logging, and high compliance with PEP 8.

```text
├── preprocessing.py                 # Data loaders, EDA summary functions, and ChurnPreprocessor class
├── model.py                         # Multi-model classifier benchmark suite, evaluations & serialization
├── train.py                         # Main execution script to train models and save diagnostics plots
├── predict.py                       # High-speed local and batch inference API engine with explainability
├── app.py                           # Production Flask web server exposing visual dashboards and REST APIs
├── requirements.txt                 # Python dependencies and locked constraints
├── customer_churn_prediction.ipynb  # Self-contained Jupyter Notebook workspace for experiments
├── static/
│   ├── plots/                       # Generated PNG figures (ROC, CM, feature importances, curves)
│   └── results.json                 # Exported metrics from the trained models benchmark
└── templates/                       # Bootstrap HTML templates for the Flask application
```

---

## 📊 Dataset Schema and Explanation
The model is optimized to ingest and train on the famous **Telco Customer Churn** schema (commonly from Kaggle). Below are the supported input features:

### Demographics and Accounts
- `customerID`: String (uniquely identifying the subscriber, dropped during feature selection).
- `gender`: Categorical (`Male`, `Female`).
- `SeniorCitizen`: Binary (`0` = No, `1` = Yes).
- `Partner`: Categorical (`Yes`, `No`).
- `Dependents`: Categorical (`Yes`, `No`).

### Contract and billing Details
- `tenure`: Integer (number of months the customer has stayed with the company, up to 72).
- `Contract`: Categorical (`Month-to-month`, `One year`, `Two year`).
- `PaperlessBilling`: Categorical (`Yes`, `No`).
- `PaymentMethod`: Categorical (`Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)`).
- `MonthlyCharges`: Float (the monthly subscription cost of service).
- `TotalCharges`: String/Float (total billed amount across tenure).

### Subscribed Services
- `PhoneService`: Categorical (`Yes`, `No`).
- `MultipleLines`: Categorical (`Yes`, `No`, `No phone service`).
- `InternetService`: Categorical (`DSL`, `Fiber optic`, `No`).
- `OnlineSecurity`: Categorical (`Yes`, `No`, `No internet service`).
- `OnlineBackup`: Categorical (`Yes`, `No`, `No internet service`).
- `DeviceProtection`: Categorical (`Yes`, `No`, `No internet service`).
- `TechSupport`: Categorical (`Yes`, `No`, `No internet service`).
- `StreamingTV`: Categorical (`Yes`, `No`, `No internet service`).
- `StreamingMovies`: Categorical (`Yes`, `No`, `No internet service`).

### Target Feature
- `Churn`: Categorical (`Yes` or `No`), indicating whether the customer closed their subscription.

---

## 🚀 Installation and Local Execution Guide

### Prerequisite
Ensure you have **Python 3.8+** installed.

### 1. Clone the repository and navigate inside
```bash
git clone <repository_url>
cd customer-churn-prediction
```

### 2. Configure Virtual Environment (Recommended)
```bash
# Create environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Training Pipeline
Run the main script to load/generate dataset, preprocess, handle class imbalance, evaluate all models, and compile results:
```bash
python train.py
```
*Note: If the dataset is absent, the script automatically creates a representative 1,500 record synthetic telco CSV file to guarantee successful execution.*

### 5. Launch the Web Application
```bash
python app.py
```
Open your web browser and navigate to `http://localhost:5000` to interact with the dashboards and input customer details for risk estimation.

---

## ☁️ Deployment Guide

### Option 1: Docker Containerization
Create a `Dockerfile` in the project root:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python train.py
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```
Build and run:
```bash
docker build -t churn-predictor .
docker run -p 5000:5000 churn-predictor
```

### Option 2: Cloud Deployment (Heroku, Render, AWS, or GCP)
1. Ensure `gunicorn` is present in `requirements.txt`.
2. Define a `Procfile` for platform launchers:
   ```text
   web: gunicorn app:app
   ```
3. Connect your GitHub repository to **Render** or **Heroku**, choose python runtime environment, and deploy. The service will automatically spin up on port 5000.
