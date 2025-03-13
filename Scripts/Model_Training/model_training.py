import os
import logging
import pickle
import pandas as pd
import sqlite3
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def setup_logging():
    """Configures logging for model training."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'model_training.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)

def load_data():
    """Loads processed datasets from SQLite database."""
    db_path = "data_lake/database/customer_churn.db"
    conn = sqlite3.connect(db_path)
    df_local = pd.read_sql_query("SELECT * FROM churn_local", conn)
    df_kaggle = pd.read_sql_query("SELECT * FROM churn_kaggle", conn)
    conn.close()
    
    return df_local, df_kaggle

def identify_columns(df, dataset_name):
    """Identifies the correct ID and target variable columns for each dataset."""
    if dataset_name == "local":
        id_column = "CustomerId"
        target_column = "Exited"
        surname_column = "Surname"
    elif dataset_name == "kaggle":
        id_column = "customerID"
        target_column = "Churn"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return id_column, target_column

def prepare_data(df, dataset_name):
    """Prepares data for model training by splitting into train and test sets."""
    id_column, target_column = identify_columns(df, dataset_name)
    
    if dataset_name == "kaggle":
        surname_column = "Surname"
    else:
        surname_column = None
    X = df.drop(columns=[id_column, target_column, surname_column], errors='ignore')  # Remove ID & target variable
    y = df[target_column] if target_column in df.columns else None
    
    if y is None:
        raise ValueError(f"Target variable '{target_column}' not found in {dataset_name} dataset.")
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name, dataset_name):
    """Trains the given model and evaluates its performance."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    logging.info(f"{model_name} Performance: {metrics}")
    print(f"{model_name} Performance: {metrics}")
    
    # Save performance report
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"performance_report_{model_name}_{dataset_name}.csv")
    pd.DataFrame([metrics]).to_csv(report_path, index=False)
    logging.info(f"Performance report saved: {report_path}")
    print(f"Performance report saved: {report_path}")
    
    return model, metrics

def save_model(model, model_name):
    """Saves the trained model using MLflow and Pickle."""
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    mlflow.sklearn.log_model(model, model_name)
    logging.info(f"Model saved: {model_path}")
    print(f"Model saved: {model_path}")

def train_models():
    """Trains multiple models on both datasets."""
    df_local, df_kaggle = load_data()
    
    for dataset_name, df in zip(["local", "kaggle"], [df_local, df_kaggle]):
        try:
            X_train, X_test, y_train, y_test = prepare_data(df, dataset_name)
            
            # Train and evaluate models
            rf_model, rf_metrics = train_and_evaluate(X_train, X_test, y_train, y_test, RandomForestClassifier(), f"RandomForest_{dataset_name}", dataset_name)
            lr_model, lr_metrics = train_and_evaluate(X_train, X_test, y_train, y_test, LogisticRegression(max_iter=1000), f"LogisticRegression_{dataset_name}", dataset_name)
            
            # Save models
            save_model(rf_model, f"RandomForest_{dataset_name}")
            save_model(lr_model, f"LogisticRegression_{dataset_name}")
            
        except ValueError as e:
            logging.error(f"Error processing {dataset_name}: {e}")
            print(f"Error processing {dataset_name}: {e}")

if __name__ == "__main__":
    setup_logging()
    train_models()
