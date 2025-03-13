import os
import shutil
from datetime import datetime

def create_storage_structure(base_dir="data_lake"):
    """Creates a structured local storage system for ingested data."""
    sub_dirs = ["raw/csv", "raw/kaggle_api", "processed", "models", "logs"]

    for sub in sub_dirs:
        path = os.path.join(base_dir, sub)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

    return base_dir

def move_to_storage(source_file, storage_type, base_dir="data_lake"):
    """Moves an ingested file to the appropriate storage partition."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dest_dir = os.path.join(base_dir, "raw", storage_type, timestamp)
    os.makedirs(dest_dir, exist_ok=True)

    if storage_type == "csv":
        file_name = os.path.basename(source_file)
    elif storage_type == "kaggle_api":
        file_name = "kaggle_churn.csv"
    #file_name = os.path.basename(source_file)
    dest_path = os.path.join(dest_dir, file_name)

    shutil.copy(source_file, dest_path)
    print(f"Copied {source_file} to {dest_path}")
    return dest_path

if __name__ == "__main__":
    base_dir = create_storage_structure()

    # Example usage (assuming ingestion scripts save files in 'data/raw/')
    sample_csv = "data/raw/customer_churn.csv"
    sample_api = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    move_to_storage(sample_csv, "csv", base_dir)
    move_to_storage(sample_api, "kaggle_api", base_dir)
