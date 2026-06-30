"""
Customer Churn Prediction - Production Flask Web Application Server
Author: Senior ML Engineer & Architect
File: app.py

This script implements a high-performance Flask web server. It exposes routing
for interactive customer form submissions, metrics dashboards, and API interfaces.
It integrates seamlessly with 'predict.py' and 'preprocessing.py' to serve predictions.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify

# Import prediction engine
try:
    from predict import ChurnInferenceEngine
except ImportError:
    # Adding workspace root to path if needed
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from predict import ChurnInferenceEngine

# Configure Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FlaskApplication")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize the machine learning predictive engine lazily
inference_engine = None


def get_inference_engine():
    """
    Ensures safe, lazy instantiation of ML models on first request
    to prevent memory and startup blockers.
    """
    global inference_engine
    if inference_engine is None:
        logger.info("Initializing ML Model assets inside Flask Server...")
        try:
            inference_engine = ChurnInferenceEngine()
        except Exception as e:
            logger.error(f"Failed to load predictive assets: {str(e)}", exc_info=True)
            raise e
    return inference_engine


@app.route("/")
def home():
    """
    Renders the modern Dashboard Home Page.
    Fetches the pre-computed evaluation results from results.json if available.
    """
    results_data = {}
    results_path = os.path.join(app.static_folder, "results.json")
    
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read static/results.json: {str(e)}")
            
    return render_template("index.html", results=results_data)


@app.route("/about")
def about():
    """
    Renders the ML pipeline and dataset informational page.
    """
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Handles predicting Churn from form inputs.
    - GET: Renders the Customer Evaluation Form.
    - POST: Processes inputs, calculates risk, and displays results.
    """
    if request.method == "POST":
        try:
            # Parse form fields
            form_data = {
                "gender": request.form.get("gender", "Male"),
                "SeniorCitizen": int(request.form.get("SeniorCitizen", 0)),
                "Partner": request.form.get("Partner", "No"),
                "Dependents": request.form.get("Dependents", "No"),
                "tenure": int(request.form.get("tenure", 1)),
                "PhoneService": request.form.get("PhoneService", "Yes"),
                "MultipleLines": request.form.get("MultipleLines", "No"),
                "InternetService": request.form.get("InternetService", "DSL"),
                "OnlineSecurity": request.form.get("OnlineSecurity", "No"),
                "OnlineBackup": request.form.get("OnlineBackup", "No"),
                "DeviceProtection": request.form.get("DeviceProtection", "No"),
                "TechSupport": request.form.get("TechSupport", "No"),
                "StreamingTV": request.form.get("StreamingTV", "No"),
                "StreamingMovies": request.form.get("StreamingMovies", "No"),
                "Contract": request.form.get("Contract", "Month-to-month"),
                "PaperlessBilling": request.form.get("PaperlessBilling", "No"),
                "PaymentMethod": request.form.get("PaymentMethod", "Electronic check"),
                "MonthlyCharges": float(request.form.get("MonthlyCharges", 20.0)),
                "TotalCharges": request.form.get("TotalCharges", "20.0")
            }
            
            logger.info(f"Processing incoming prediction request: Tenure={form_data['tenure']}, Contract={form_data['Contract']}")
            
            # Predict
            engine = get_inference_engine()
            prediction_output = engine.predict_single(form_data)
            
            # Render predictions page with outputs
            return render_template("result.html", result=prediction_output, customer=form_data)
            
        except Exception as e:
            logger.error(f"Error executing customer churn predictions: {str(e)}", exc_info=True)
            return render_template("predict.html", error=f"Prediction Error: {str(e)}")
            
    # GET Request: serve empty inputs form
    return render_template("predict.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Exposes high-speed JSON API interface for batched or external predictions.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "Missing JSON Payload"}), 400
            
        engine = get_inference_engine()
        prediction_output = engine.predict_single(data)
        
        return jsonify({
            "status": "success",
            "data": prediction_output
        })
        
    except Exception as e:
        logger.error(f"JSON prediction API error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500


if __name__ == "__main__":
    # Ensure folders exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/plots", exist_ok=True)
    
    # Warm up model engine to cache artifacts
    try:
        get_inference_engine()
    except Exception:
        logger.warning("Pipeline is missing. Run 'python train.py' to generate model artifacts.")
        
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
