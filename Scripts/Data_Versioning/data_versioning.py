import os
import logging
import subprocess
import argparse
from datetime import datetime

def setup_logging():
    """Configures logging for data versioning."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'data_versioning.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)

def init_dvc():
    """Initializes DVC if not already initialized."""
    if not os.path.exists(".dvc"):
        subprocess.run(["dvc", "init"], check=True)
        logging.info("Initialized DVC repository.")
        print("Initialized DVC repository.")

def track_dataset(dataset_path, version_message):
    """Adds dataset to DVC tracking and commits a version."""
    subprocess.run(["dvc", "add", dataset_path], check=True)
    subprocess.run(["git", "add", dataset_path + ".dvc"], check=True)
    subprocess.run(["git", "commit", "-m", version_message], check=True)
    subprocess.run(["dvc", "push"], check=True)
    logging.info(f"Tracked dataset: {dataset_path} - {version_message}")
    print(f"Tracked dataset: {dataset_path} - {version_message}")

def save_version_metadata(dataset_path, version_message):
    """Saves metadata about dataset versions."""
    metadata_file = "data_lake/versioning/version_metadata.txt"
    os.makedirs("data_lake/versioning", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(metadata_file, "a") as f:
        f.write(f"{timestamp} | {dataset_path} | {version_message}\n")
    logging.info(f"Version metadata saved for {dataset_path}")

def version_datasets(stage):
    """Handles versioning for datasets based on the pipeline stage."""
    dataset_paths = {
        "raw": ["data_lake/raw/csv/customer_churn.csv", "data_lake/raw/kaggle/kaggle_churn.csv"],
        "processed": ["data_lake/processed/cleaned_data_churn_local.csv", "data_lake/processed/cleaned_data_churn_kaggle.csv"],
        "transformed": ["data_lake/transformed/transformed_data_local.csv", "data_lake/transformed/transformed_data_kaggle.csv"]
    }
    
    if stage not in dataset_paths:
        logging.error(f"Invalid stage: {stage}. Choose from 'raw', 'processed', or 'transformed'.")
        print(f"Invalid stage: {stage}. Choose from 'raw', 'processed', or 'transformed'.")
        return
    
    for dataset in dataset_paths[stage]:
        if os.path.exists(dataset):
            track_dataset(dataset, f"Versioned {stage} dataset: {dataset}")
            save_version_metadata(dataset, f"Versioned {stage} dataset: {dataset}")

def main():
    parser = argparse.ArgumentParser(description="Dataset Versioning using DVC")
    parser.add_argument("--stage", type=str, required=True, help="Pipeline stage: raw, processed, transformed")
    args = parser.parse_args()
    
    setup_logging()
    init_dvc()
    version_datasets(args.stage)

if __name__ == "__main__":
    main()
