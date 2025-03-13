import pandas as pd
import numpy as np
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def setup_logging():
    """Configures logging for data preparation."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'data_preparation.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)

def get_latest_file(directory, file_pattern):
    """Finds the most recent file in a dynamically created directory."""
    subdirs = sorted(glob.glob(os.path.join(directory, "*/")), reverse=True)
    for subdir in subdirs:
        file_path = os.path.join(subdir, file_pattern)
        if os.path.exists(file_path):
            return file_path
    return None

def load_latest_data():
    """Loads the latest processed data for preparation from both local churn and Kaggle API datasets in chunks."""
    local_file = get_latest_file("data_lake/raw/csv", "customer_churn.csv")
    kaggle_file = get_latest_file("data_lake/raw/kaggle_api", "kaggle_churn.csv")

    chunk_size = 10000  # Read in chunks to avoid memory overload
    df_local = pd.concat(pd.read_csv(local_file, low_memory=True, chunksize=chunk_size)) if local_file else None
    df_kaggle = pd.concat(pd.read_csv(kaggle_file, low_memory=True, chunksize=chunk_size)) if kaggle_file else None

    if df_local is None:
        logging.error("No recent local customer churn data found.")
    if df_kaggle is None:
        logging.error("No recent Kaggle churn data found.")

    return df_local, df_kaggle

def process_dataset(df, dataset_name):
    """Cleans, encodes, and standardizes the dataset separately."""
    if df is None:
        logging.error(f"{dataset_name} dataset not available.")
        return None

    # Convert numeric columns stored as objects
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['customerID', 'TotalCharges']:
            df[col] = pd.to_numeric(df[col], errors='ignore')

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    logging.info(f"Handled missing values for {dataset_name} dataset.")

    # Identify categorical columns for One-Hot Encoding, excluding customerID and TotalCharges
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod','Geography',
                        'Gender']
    categorical_cols = [col for col in categorical_cols if col in df.columns] # Filter columns that actually exist in df
    # Encode categorical variables using One-Hot Encoding
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    logging.info(f"Categorical variables encoded using One-Hot Encoding for {dataset_name} dataset.")

    # Standardize numerical variables
    scaler = StandardScaler()
    numeric_cols = ['tenure','Tenure', 'MonthlyCharges', 'TotalCharges','CreditScore','Age','Balance','EstimatedSalary']
    numeric_cols = [col for col in numeric_cols if col in df.columns] # Filter columns that actually exist in df
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # Invalid values become NaN
        df[col].fillna(df[col].median(), inplace=True) # Fill NaNs with median
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    if 'Churn' in df.columns: 
        df["Churn"] = df["Churn"].map({'Yes': 1, 'No': 0})
    if 'Surname' in df.columns:
        df.drop('Surname', axis=1, inplace=True)
    logging.info(f"Numerical variables standardized for {dataset_name} dataset.")

    return df

def visualize_data(df, dataset_name):
    """Generates visualizations for data exploration while minimizing memory usage."""
    os.makedirs("data_lake/processed", exist_ok=True)
    df_sample = df.sample(n=min(2000, len(df)), random_state=42)  # Further reduce sample size to 2000
    
    # Plot numerical features separately
    numeric_cols = df_sample.select_dtypes(include=['number']).columns.tolist()
    plt.figure(figsize=(12, 6))
    df_sample[numeric_cols].hist(figsize=(12, 10), bins=20, layout=(4, 3))
    plt.tight_layout()
    plt.savefig(f"data_lake/processed/numeric_distribution_{dataset_name}.png")
    logging.info(f"Generated histogram for numerical features in {dataset_name} dataset.")
    
    # Boxplot for numerical features
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_sample[numeric_cols])
    plt.xticks(rotation=90)
    plt.title(f"Boxplot for Outlier Detection - {dataset_name}")
    plt.savefig(f"data_lake/processed/boxplot_{dataset_name}.png")
    logging.info(f"Generated boxplot for numerical features in {dataset_name} dataset.")
    
    # Plot categorical features separately
    categorical_cols = df_sample.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=df_sample[col])
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {col} - {dataset_name}")
        plt.savefig(f"data_lake/processed/categorical_{col}_{dataset_name}.png")
        logging.info(f"Generated count plot for {col} in {dataset_name} dataset.")

def save_cleaned_data(df, dataset_name):
    """Saves the cleaned dataset separately for each source."""
    output_file = f"data_lake/processed/cleaned_data_{dataset_name}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logging.info(f"Cleaned data saved to {output_file}")
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    setup_logging()
    df_local, df_kaggle = load_latest_data()

    # Process and save the local churn dataset
    if df_local is not None:
        df_local = process_dataset(df_local, "local")
        visualize_data(df_local, "local")
        save_cleaned_data(df_local, "local")

    # Process and save the Kaggle churn dataset
    if df_kaggle is not None:
        df_kaggle = process_dataset(df_kaggle, "kaggle")
        visualize_data(df_kaggle, "kaggle")
        save_cleaned_data(df_kaggle, "kaggle")
