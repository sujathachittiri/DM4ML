from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_script(script_name):
    """Executes a Python script as a subprocess."""
    try:
        subprocess.run(["python", script_name], check=True)
        logging.info(f"Successfully executed {script_name}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing {script_name}: {e}")
        raise

def run_data_ingestion():
    run_script("data_ingestion.py")

def run_kaggle_api_ingestion():
    run_script("kaggle_api_ingestion.py")

def run_data_storage():
    run_script("data_storage.py")

def run_dvc_versioning_raw():
    run_script("data_versioning.py --stage raw")

def run_data_validation():
    run_script("data_validation.py")

def run_data_preparation():
    run_script("data_preparation.py")

def run_dvc_versioning_processed():
    run_script("data_versioning.py --stage processed")

def run_data_transformation():
    run_script("data_transformation_storage.py")

def run_feature_store():
    run_script("feature_store.py")

def run_dvc_versioning_transformed():
    run_script("data_versioning.py --stage transformed")

def run_model_training():
    run_script("model_training.py")

def run_model_deployment():
    run_script("model_deployment.py")

# Define default args for Airflow DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
with DAG(
    "ml_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
) as dag:
    
    task_ingestion = PythonOperator(task_id="data_ingestion", python_callable=run_data_ingestion)
    task_kaggle_api_ingestion = PythonOperator(task_id="kaggle_api_ingestion", python_callable=run_kaggle_api_ingestion)
    task_data_storage = PythonOperator(task_id="data_storage", python_callable=run_data_storage)
    task_versioning_raw = PythonOperator(task_id="dvc_versioning_raw", python_callable=run_dvc_versioning_raw)
    task_validation = PythonOperator(task_id="data_validation", python_callable=run_data_validation)
    task_preparation = PythonOperator(task_id="data_preparation", python_callable=run_data_preparation)
    task_versioning_processed = PythonOperator(task_id="dvc_versioning_processed", python_callable=run_dvc_versioning_processed)
    task_transformation = PythonOperator(task_id="data_transformation_storage", python_callable=run_data_transformation)
    task_feature_store = PythonOperator(task_id="feature_store", python_callable=run_feature_store)
    task_versioning_transformed = PythonOperator(task_id="dvc_versioning_transformed", python_callable=run_dvc_versioning_transformed)
    task_training = PythonOperator(task_id="model_training", python_callable=run_model_training)
    task_deployment = PythonOperator(task_id="model_deployment", python_callable=run_model_deployment)

    # Define task dependencies
    task_ingestion >> task_kaggle_api_ingestion >> task_data_storage >> task_versioning_raw >> task_validation >> task_preparation >> task_versioning_processed \
        >> task_transformation >> task_feature_store >> task_versioning_transformed >> task_training >> task_deployment
