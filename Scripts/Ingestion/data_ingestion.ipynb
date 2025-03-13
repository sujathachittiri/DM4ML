"""This script reads the CSV file from Local. For Google Colab upload the input file to Customer_Churn_Dataset_Input.csv /content/sample_data/ """
import pandas as pd
import logging
import os
import sys

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'data_ingestion.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # Print logs to console too

def load_data(file_path):
    """Loads customer churn dataset from a CSV file and logs the process."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}")
        print("Dataset loaded successfully!")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        print("Error loading dataset:", str(e))
        return None

def save_raw_data(df, output_path):
    """Saves raw data in a structured format."""
    try:
        raw_data_dir = "data/raw"
        os.makedirs(raw_data_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Raw data saved to {output_path}")
        print(f"Raw data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving raw data: {e}")
        print("Error saving raw data:", str(e))

if __name__ == "__main__":
    file_path = "/content/sample_data/Customer_Churn_Dataset_Input.csv"
    output_path = "data/raw/customer_churn.csv"
    df = load_data(file_path)
    if df is not None:
        save_raw_data(df, output_path)
