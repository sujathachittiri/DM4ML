import os
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

def setup_logging():
    """Configures logging for API data ingestion."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'api_data_ingestion_kaggle.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)

def download_kaggle_dataset(dataset_name, output_folder="data/raw"):
    """Downloads a dataset from Kaggle and stores it in the raw data folder."""
    os.makedirs(output_folder, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    logging.info(f"Downloading Kaggle dataset: {dataset_name}")
    api.dataset_download_files(dataset_name, path=output_folder, unzip=True)
    logging.info(f"Dataset {dataset_name} downloaded successfully.")
    print(f"Dataset {dataset_name} downloaded successfully.")

if __name__ == "__main__":
    setup_logging()
    kaggle_dataset = "blastchar/telco-customer-churn"  # Example dataset (Telco Customer Churn)
    download_kaggle_dataset(kaggle_dataset)
