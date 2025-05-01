# import joblib
# import pandas as pd
# import os

# # Get absolute path to the models directory
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current directory
# MODEL_DIR = os.path.join(BASE_DIR, "models")  # Adjust path to models folder

# # Load trained models
# rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
# xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
# iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

# # Fraud Detection Functions
# def detect_identity_fraud(df):
#     """Predict fraud using the Random Forest model."""
#     predictions = rf_model.predict(df.drop(columns=["id"], errors="ignore"))
#     df["Fraud_Prediction"] = predictions
#     return df[df["Fraud_Prediction"] == 1]

# def detect_synthetic_fraud(df):
#     """Predict synthetic fraud using the XGBoost model."""
#     predictions = xgb_model.predict(df.drop(columns=["id"], errors="ignore"))
#     df["Fraud_Prediction"] = predictions
#     return df[df["Fraud_Prediction"] == 1]

# def detect_mule_accounts(df):
#     """Detect mule accounts using Isolation Forest."""
#     predictions = iso_forest.predict(df.drop(columns=["id"], errors="ignore"))
#     df["Fraud_Prediction"] = predictions
#     return df[df["Fraud_Prediction"] == -1]  # Anomalies are labeled as -1
#  
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.base import is_classifier

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load models with feature validation
def load_model(path):
    model = joblib.load(path)
    if not hasattr(model, 'feature_names_in_'):
        raise ValueError(f"Model at {path} missing feature_names_in_ attribute. Retrain with sklearn>=1.0")
    return model

rf_model = load_model(os.path.join(MODEL_DIR, "random_forest.pkl"))
xgb_model = load_model(os.path.join(MODEL_DIR, "xgboost.pkl"))
iso_forest = load_model(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

# Fraud Detection Core Functions
def align_features(df, model):
    """Ensure dataframe columns match model's training features"""
    required_features = model.feature_names_in_
    return df[[col for col in required_features if col in df.columns]]

def detect_identity_fraud(df):
    """Identity fraud detection with proper feature alignment"""
    df = align_features(df.drop(columns=["id", "fraud"], errors="ignore"), rf_model)
    
    if is_classifier(rf_model):
        df["Fraud_Score"] = rf_model.predict_proba(df)[:, 1]  # Probability of fraud class
    else:
        df["Fraud_Score"] = rf_model.predict(df)
    
    df["Fraud_Prediction"] = (df["Fraud_Score"] > 0.5).astype(int)
    return df[df["Fraud_Prediction"] == 1].copy()

def detect_synthetic_fraud(df):
    """Synthetic fraud detection with XGBoost calibration"""
    df = align_features(df.drop(columns=["id", "isFraud"], errors="ignore"), xgb_model)
    
    if is_classifier(xgb_model):
        df["Fraud_Score"] = xgb_model.predict_proba(df)[:, 1]
    else:
        df["Fraud_Score"] = xgb_model.predict(df)
    
    df["Fraud_Prediction"] = (df["Fraud_Score"] > 0.5).astype(int)
    return df[df["Fraud_Prediction"] == 1].copy()

def detect_mule_accounts(df):
    """Mule account detection with normalized anomaly scores"""
    df = align_features(df.drop(columns=["id", "Class"], errors="ignore"), iso_forest)
    
    # Calculate anomaly scores
    df["Anomaly_Score"] = iso_forest.decision_function(df)
    # Normalize to 0-1 range (higher = more anomalous)
    df["Anomaly_Score"] = (df["Anomaly_Score"] - df["Anomaly_Score"].min()) / (df["Anomaly_Score"].max() - df["Anomaly_Score"].min())
    
    df["Fraud_Prediction"] = (df["Anomaly_Score"] > 0.65).astype(int)  # Adjust threshold as needed
    return df[df["Fraud_Prediction"] == 1].copy()
