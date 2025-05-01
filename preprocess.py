import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder

import os

# Move up TWO levels to the project root (identity-fraud-api/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_datasets():
    bank_sim = pd.read_csv(os.path.join(BASE_DIR, "data", "banksim.csv"))
    pay_sim = pd.read_csv(os.path.join(BASE_DIR, "data", "paysim.csv"))
    credit_fraud = pd.read_csv(os.path.join(BASE_DIR, "data", "creditcard.csv"))
    return bank_sim, pay_sim, credit_fraud


# def load_datasets():
#     """Load all datasets."""
#     bank_sim = pd.read_csv("data/banksim.csv")
#     pay_sim = pd.read_csv("data/paysim.csv")
#     credit_fraud = pd.read_csv("data/creditcard.csv")
#     return bank_sim, pay_sim, credit_fraud

def preprocess_data(df, target_column):
    """Clean and preprocess the dataset."""
    df = df.drop_duplicates()
    
    df = df.dropna()
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    if df[target_column].dtype == 'float64':
        df[target_column] = (df[target_column] > 0).astype(int)
    
    scaler = StandardScaler()
    features = df.drop(columns=[target_column])
    df[features.columns] = scaler.fit_transform(features)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled

def main():
    bank_sim, pay_sim, credit_fraud = load_datasets()
    
    datasets = {
        "banksim": (bank_sim, "fraud"),
        "paysim": (pay_sim, "isFraud"),
        "creditcard": (credit_fraud, "Class")
    }
    
    for name, (df, target) in datasets.items():
        X, y = preprocess_data(df, target)
        processed_df = pd.DataFrame(X, columns=df.drop(columns=[target]).columns)
        processed_df[target] = y
        output_dir = os.path.join(BASE_DIR, "processed_data")
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

        output_file = os.path.join(output_dir, f"processed_{name}.csv")
        processed_df.to_csv(output_file, index=False)

        print(f"Processed {name} dataset saved at: {output_file}")

if __name__ == "__main__":
    main()
