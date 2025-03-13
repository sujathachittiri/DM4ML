import pandas as pd
import numpy as np
import logging
import os
import sqlite3

def setup_logging():
    """Configures logging for data transformation."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'data_transformation.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)

def load_cleaned_data():
    """Loads the latest cleaned datasets for transformation."""
    local_file = "data_lake/processed/cleaned_data_local.csv"
    kaggle_file = "data_lake/processed/cleaned_data_kaggle.csv"

    df_local = pd.read_csv(local_file) if os.path.exists(local_file) else None
    df_kaggle = pd.read_csv(kaggle_file) if os.path.exists(kaggle_file) else None

    return df_local, df_kaggle

def feature_engineering(df, dataset_name):
    """Creates new aggregated and derived features safely by checking column availability."""
    if df is None:
        logging.warning(f"No data available for {dataset_name}.")
        return None

    if dataset_name == "local":
        # Local dataset transformations
        if {'Balance', 'NumOfProducts'}.issubset(df.columns):
            df['TotalBalance'] = df['Balance'] * df['NumOfProducts']
        else:
            logging.warning(f"Skipping TotalBalance for {dataset_name} as required columns are missing.")

        if {'Tenure', 'NumOfProducts'}.issubset(df.columns):
            df['TenurePerProduct'] = df['Tenure'] / (df['NumOfProducts'] + 1)
        else:
            logging.warning(f"Skipping TenurePerProduct for {dataset_name} as required columns are missing.")

        if {'IsActiveMember', 'HasCrCard'}.issubset(df.columns):
            df['ActivityScore'] = df['IsActiveMember'] * df['HasCrCard']
        else:
            logging.warning(f"Skipping ActivityScore for {dataset_name} as required columns are missing.")

    elif dataset_name == "kaggle":
        # Kaggle dataset transformations
        if {'MonthlyCharges', 'tenure'}.issubset(df.columns):
            df['AvgMonthlySpend'] = df['MonthlyCharges'] * df['tenure']
        else:
            logging.warning(f"Skipping AvgMonthlySpend for {dataset_name} as required columns are missing.")

        if {'TotalCharges', 'tenure'}.issubset(df.columns):
            df['ChargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
        else:
            logging.warning(f"Skipping ChargePerMonth for {dataset_name} as required columns are missing.")

    logging.info(f"Feature engineering applied to {dataset_name} dataset.")
    return df

def store_in_database(df, dataset_name):
    """Stores the transformed data in an SQLite database."""
    db_path = "data_lake/database/customer_churn.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    table_name = f"churn_{dataset_name}"
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

    logging.info(f"Stored {dataset_name} dataset in database table {table_name}.")
    print(f"Stored {dataset_name} dataset in database table {table_name}.")

def generate_sample_queries():
    """Creates sample SQL queries for transformed data retrieval."""
    queries = [
        "SELECT * FROM churn_local LIMIT 10;",
        "SELECT * FROM churn_kaggle LIMIT 10;",
        "SELECT CustomerId, TotalBalance FROM churn_local WHERE TotalBalance > 50000;",
        "SELECT AVG(ChargePerMonth) FROM churn_kaggle;"
    ]

    os.makedirs("data_lake/database", exist_ok=True)
    with open("data_lake/database/sample_queries.sql", "w") as f:
        for query in queries:
            f.write(query + "\n")

    logging.info("Sample SQL queries saved.")
    print("Sample SQL queries saved.")

def save_summary():
    """Saves a summary of the transformation logic."""
    summary = """
    Data Transformation Summary:
    - Local dataset:
      - Created 'TotalBalance' = Balance * NumOfProducts
      - Created 'TenurePerProduct' = Tenure / (NumOfProducts + 1)
      - Created 'ActivityScore' = IsActiveMember * HasCrCard
    - Kaggle dataset:
      - Created 'AvgMonthlySpend' = MonthlyCharges * tenure
      - Created 'ChargePerMonth' = TotalCharges / (tenure + 1)
    - Transformed data stored in SQLite database with tables: churn_local, churn_kaggle
    - Sample queries provided in 'data_lake/database/sample_queries.sql'
    """
    os.makedirs("data_lake/documentation", exist_ok=True)
    with open("data_lake/documentation/transformation_summary.txt", "w") as f:
        f.write(summary)
    logging.info("Transformation summary saved.")
    print("Transformation summary saved.")

if __name__ == "__main__":
    setup_logging()
    df_local, df_kaggle = load_cleaned_data()

    if df_local is not None:
        df_local = feature_engineering(df_local, "local")
        store_in_database(df_local, "local")

    if df_kaggle is not None:
        df_kaggle = feature_engineering(df_kaggle, "kaggle")
        store_in_database(df_kaggle, "kaggle")

    generate_sample_queries()
    save_summary()
