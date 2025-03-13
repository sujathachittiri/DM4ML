import os
import logging
import pandas as pd
import sqlite3
from feast import FeatureStore, FeatureView, Entity, Field
from feast.infra.offline_stores.file_source import FileSource
from feast.types import Float32, Int64, ValueType

def setup_logging():
    """Configures logging for feature store setup."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'feature_store.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)

def load_transformed_data(table_name):
    """Loads the transformed datasets from SQLite database."""
    db_path = "data_lake/database/customer_churn.db"
    if not os.path.exists(db_path):
        logging.error("Database file not found.")
        return None
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    
    return df

def configure_feature_store(df, table_name):
    """Configures and initializes the feature store with dataset-specific fields."""
    os.makedirs("feature_store", exist_ok=True)
    
    entity = Entity(name="customerID", value_type=ValueType.INT64)
    
    if table_name == "churn_local":
        features = [
            Field(name="TotalBalance", dtype=Float32),
            Field(name="TenurePerProduct", dtype=Float32),
            Field(name="ActivityScore", dtype=Float32)
        ]
    elif table_name == "churn_kaggle":
        features = [
            Field(name="AvgMonthlySpend", dtype=Float32),
            Field(name="ChargePerMonth", dtype=Float32)
        ]
    else:
        logging.error(f"Unknown dataset: {table_name}")
        return
    
    source = FileSource(
        path=f"data_lake/processed/cleaned_data_{table_name}.csv",
        event_timestamp_column="event_timestamp"  # Replace with your event timestamp column if available
    )

    feature_view = FeatureView(
        name=f"{table_name}_features",
        entities=[entity],
        ttl=None,
        schema=features,
        source=source
    )
    
    with open(f"feature_store/feature_metadata_{table_name}.txt", "w") as f:
        f.write("Feature Metadata:\n")
        if table_name == "churn_local":
            f.write("TotalBalance - Customer total balance (Balance * NumOfProducts)\n")
            f.write("TenurePerProduct - Tenure divided by number of products\n")
            f.write("ActivityScore - IsActiveMember * HasCrCard\n")
        elif table_name == "churn_kaggle":
            f.write("AvgMonthlySpend - Monthly Charges multiplied by tenure\n")
            f.write("ChargePerMonth - Total Charges divided by tenure\n")
    
    logging.info(f"Feature store configured for {table_name}.")
    print(f"Feature store configured for {table_name}.")

def save_sample_feature_retrieval(table_name):
    """Saves a sample feature retrieval query specific to each dataset."""
    if table_name == "churn_local":
        query = f"""
        SELECT customerID, TotalBalance, TenurePerProduct, ActivityScore
        FROM {table_name}
        WHERE TotalBalance > 50000;
        """
    elif table_name == "churn_kaggle":
        query = f"""
        SELECT customerID, AvgMonthlySpend, ChargePerMonth
        FROM {table_name}
        WHERE ChargePerMonth > 50;
        """
    else:
        logging.error(f"Unknown dataset: {table_name}")
        return
    
    os.makedirs("feature_store/queries", exist_ok=True)
    with open(f"feature_store/queries/sample_query_{table_name}.sql", "w") as f:
        f.write(query)
    
    logging.info(f"Sample feature retrieval query saved for {table_name}.")
    print(f"Sample feature retrieval query saved for {table_name}.")

if __name__ == "__main__":
    setup_logging()
    
    for dataset in ["churn_kaggle", "churn_local"]:
        df = load_transformed_data(dataset)
        if df is not None:
            configure_feature_store(df, dataset)
            save_sample_feature_retrieval(dataset)
