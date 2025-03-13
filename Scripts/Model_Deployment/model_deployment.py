import os
import logging
import pickle
import sqlite3
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load trained models
MODEL_PATH_LOCAL = "models/RandomForest_local.pkl"
MODEL_PATH_KAGGLE = "models/RandomForest_kaggle.pkl"

def load_model(model_path):
    """Loads a trained model from the specified path."""
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        logging.error(f"Model file not found: {model_path}")
        return None

model_local = load_model(MODEL_PATH_LOCAL)
model_kaggle = load_model(MODEL_PATH_KAGGLE)

def get_latest_features(dataset_name):
    """Fetches the latest feature set from the database."""
    db_path = "data_lake/database/customer_churn.db"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {dataset_name} ORDER BY ROWID DESC LIMIT 1", conn)
    conn.close()
    return df.drop(columns=['CustomerId', 'customerID', 'Exited', 'Churn'], errors='ignore')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predicting customer churn."""
    data = request.get_json()
    dataset = data.get("dataset", "local")
    
    if dataset == "local":
        model = model_local
        feature_data = get_latest_features("churn_local")
    elif dataset == "kaggle":
        model = model_kaggle
        feature_data = get_latest_features("churn_kaggle")
    else:
        return jsonify({"error": "Invalid dataset. Choose 'local' or 'kaggle'."}), 400
    
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500
    
    prediction = model.predict(feature_data)
    return jsonify({"dataset": dataset, "prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
