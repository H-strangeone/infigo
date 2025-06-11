# # import joblib
# # import pandas as pd
# # import os

# # # Get absolute path to the models directory
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current directory
# # MODEL_DIR = os.path.join(BASE_DIR, "models")  # Adjust path to models folder

# # # Load trained models
# # rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
# # xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
# # iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

# # # Fraud Detection Functions
# # def detect_identity_fraud(df):
# #     """Predict fraud using the Random Forest model."""
# #     predictions = rf_model.predict(df.drop(columns=["id"], errors="ignore"))
# #     df["Fraud_Prediction"] = predictions
# #     return df[df["Fraud_Prediction"] == 1]

# # def detect_synthetic_fraud(df):
# #     """Predict synthetic fraud using the XGBoost model."""
# #     predictions = xgb_model.predict(df.drop(columns=["id"], errors="ignore"))
# #     df["Fraud_Prediction"] = predictions
# #     return df[df["Fraud_Prediction"] == 1]

# # def detect_mule_accounts(df):
# #     """Detect mule accounts using Isolation Forest."""
# #     predictions = iso_forest.predict(df.drop(columns=["id"], errors="ignore"))
# #     df["Fraud_Prediction"] = predictions
# #     return df[df["Fraud_Prediction"] == -1]  # Anomalies are labeled as -1
# #  
# import joblib
# import pandas as pd
# import numpy as np
# import os
# from sklearn.base import is_classifier

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # Load models with feature validation
# def load_model(path):
#     model = joblib.load(path)
#     if not hasattr(model, 'feature_names_in_'):
#         raise ValueError(f"Model at {path} missing feature_names_in_ attribute. Retrain with sklearn>=1.0")
#     return model

# rf_model = load_model(os.path.join(MODEL_DIR, "random_forest.pkl"))
# xgb_model = load_model(os.path.join(MODEL_DIR, "xgboost.pkl"))
# iso_forest = load_model(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

# # Fraud Detection Core Functions
# def align_features(df, model):
#     """Ensure dataframe columns match model's training features"""
#     required_features = model.feature_names_in_
#     return df[[col for col in required_features if col in df.columns]]

# def detect_identity_fraud(df):
#     """Identity fraud detection with proper feature alignment"""
#     df = align_features(df.drop(columns=["id", "fraud"], errors="ignore"), rf_model)
    
#     if is_classifier(rf_model):
#         df["Fraud_Score"] = rf_model.predict_proba(df)[:, 1]  # Probability of fraud class
#     else:
#         df["Fraud_Score"] = rf_model.predict(df)
    
#     df["Fraud_Prediction"] = (df["Fraud_Score"] > 0.5).astype(int)
#     return df[df["Fraud_Prediction"] == 1].copy()

# def detect_synthetic_fraud(df):
#     """Synthetic fraud detection with XGBoost calibration"""
#     df = align_features(df.drop(columns=["id", "isFraud"], errors="ignore"), xgb_model)
    
#     if is_classifier(xgb_model):
#         df["Fraud_Score"] = xgb_model.predict_proba(df)[:, 1]
#     else:
#         df["Fraud_Score"] = xgb_model.predict(df)
    
#     df["Fraud_Prediction"] = (df["Fraud_Score"] > 0.5).astype(int)
#     return df[df["Fraud_Prediction"] == 1].copy()

# def detect_mule_accounts(df):
#     """Mule account detection with normalized anomaly scores"""
#     df = align_features(df.drop(columns=["id", "Class"], errors="ignore"), iso_forest)
    
#     # Calculate anomaly scores
#     df["Anomaly_Score"] = iso_forest.decision_function(df)
#     # Normalize to 0-1 range (higher = more anomalous)
#     df["Anomaly_Score"] = (df["Anomaly_Score"] - df["Anomaly_Score"].min()) / (df["Anomaly_Score"].max() - df["Anomaly_Score"].min())
    
#     df["Fraud_Prediction"] = (df["Anomaly_Score"] > 0.65).astype(int)  # Adjust threshold as needed
#     return df[df["Fraud_Prediction"] == 1].copy()
import joblib
import pandas as pd
import numpy as np
import os
import json
import shap
from sklearn.base import is_classifier
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ato_model = joblib.load(os.path.join(MODEL_DIR, "xgboost_ato.pkl"))
def load_model(path):
    model = joblib.load(path)
    if not hasattr(model, 'feature_names_in_'):
        raise ValueError(f"Model at {path} missing feature_names_in_")
    return model

rf_model = load_model(os.path.join(MODEL_DIR, "random_forest.pkl"))
xgb_model = load_model(os.path.join(MODEL_DIR, "xgboost_paysim.pkl")) 
iso_forest = load_model(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
def load_threshold(model_name):
    with open(os.path.join(MODEL_DIR, f"{model_name}_threshold.txt"), "r") as f:
        return float(f.read())
# def align_features(df, model):
#     """Ensure dataframe columns match model's training features"""
    
#     return df[model.feature_names_in_]
# def align_features(df, model):
#     # required = list(model.feature_names_in_)
#     # present = [col for col in required if col in df.columns]
#     # if len(present) < len(required):
#     #     missing = list(set(required) - set(df.columns))
#     #     print(f"⚠️ Missing features: {missing}")
#     # return df[present]
#     feature_names = list(model.feature_names_in_)
#     missing = [f for f in feature_names if f not in df.columns]
#     if missing:
#         raise ValueError(f"Missing features required by the model: {missing}")
#     return df[feature_names]

def align_features(df, model):
    """
    Ensure that `df` contains exactly the columns the model expects.
    Supports both sklearn-based models (feature_names_in_) and CatBoost (feature_names_).
    """
    # First, check for sklearn-style feature_names_in_
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    # Next, check for CatBoost-style feature_names_
    elif hasattr(model, "feature_names_"):
        feature_names = list(model.feature_names_)
    else:
        raise ValueError(f"Cannot find feature names on model {model}")

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features required by the model: {missing}")
    return df[feature_names]


def detect_identity_fraud(df):
    """Identity fraud detection with proper SHAP-compatible output"""
    df = align_features(df, rf_model)
    
    # Handle both classifier and regressor models
    if is_classifier(rf_model):
        df["Fraud_Score"] = rf_model.predict_proba(df)[:, 1]  # Probability of fraud class
    else:
        df["Fraud_Score"] = rf_model.predict(df)
    
    return df[df["Fraud_Score"] > 0.5].copy()

def detect_synthetic_fraud(df):
    """Synthetic fraud detection with XGBoost calibration"""
    df = align_features(df, xgb_model)
    
    # Handle binary and multi-class models
    if xgb_model.n_classes_ > 2:
        df["Fraud_Score"] = xgb_model.predict_proba(df)[:, 1]  # Use class 1 for fraud
    else:
        df["Fraud_Score"] = xgb_model.predict_proba(df)[:, 1]
    
    return df[df["Fraud_Score"] > 0.5].copy()

# def detect_mule_accounts(df):
#     """Mule account detection with normalized anomaly scores"""
#     df = align_features(df, iso_forest)
#     required_features = iso_forest.feature_names_in_
#     df = df[required_features].copy()
#     # Calculate and normalize anomaly scores
#     df["Anomaly_Score"] = iso_forest.decision_function(df)
#     df["Anomaly_Score"] = (df["Anomaly_Score"] - df["Anomaly_Score"].min()) / \
#                          (df["Anomaly_Score"].max() - df["Anomaly_Score"].min())
    
#     return df[df["Anomaly_Score"] > 0.65].copy()
def detect_account_takeover(df):
    df = align_features(df, ato_model)
    df["Fraud_Score"] = ato_model.predict_proba(df)[:, 1]
    threshold = load_threshold("xgboost_ato")
    df["Fraud_Prediction"] = (df["Fraud_Score"] > threshold).astype(int)
    return df[df["Fraud_Prediction"] == 1].copy()
def generate_fraud_profile(user_id, device_id, model, X_row, threshold=0.5):
    prob = model.predict_proba([X_row])[0][1]
    risk_level = "high" if prob > 0.75 else "medium" if prob > 0.4 else "low"
    return {
        "user_id": user_id,
        "device_id": device_id,
        "fraud_score": round(prob, 4),
        "risk_level": risk_level,
        "features": X_row.to_dict()
    }

# def detect_user_device_fraud(df):
    

#     model = CatBoostClassifier()
#     model.load_model(os.path.join(MODEL_DIR, "catboost_user_device_model.cbm"))

#     threshold = float(open(os.path.join(MODEL_DIR, "catboost_user_device_threshold.txt")).read())

#     df = align_features(df, model)
#     probs = model.predict_proba(df)[:, 1]
#     df["Fraud_Score"] = probs
#     df["Fraud_Prediction"] = (probs >= threshold).astype(int)

#     fraud_cases = df[df["Fraud_Prediction"] == 1].copy()

#     # Generate and save fraud profiles
#     profiles = []
#     for _, row in fraud_cases.iterrows():
#         profile = generate_fraud_profile(
#             user_id=row["user_id"],
#             device_id=row["device_hash"],
#             model=model,
#             X_row=row.drop(["user_id", "device_hash", "Fraud_Score", "Fraud_Prediction"])
#         )
#         profiles.append(profile)

#     # Save full fraud profile JSON
#     os.makedirs("logs", exist_ok=True)
#     with open("fraud_profiles.json", "w") as f:
#         json.dump(profiles, f, indent=4)

#     # Also append profiles to a persistent log
#     with open(os.path.join("logs", "user_device_log.txt"), "a") as log_file:
#         for profile in profiles:
#             log_file.write(json.dumps(profile) + "\n")


#     print(f"✅ {len(profiles)} fraud profiles saved to fraud_profiles.json")
#     import shap

#     explainer = shap.TreeExplainer(model)
#     X_sample = df.drop(["user_id", "device_hash"], axis=1).sample(1)
#     shap_values = explainer.shap_values(X_sample)

#     shap.plots.waterfall(
#         shap.Explanation(
#             values=shap_values[1][0],
#             base_values=explainer.expected_value[1],
#             data=X_sample.iloc[0]
#         )
#     )
#     return fraud_cases
from sklearn.ensemble import IsolationForest

# def detect_user_device_fraud(df):
#     model = CatBoostClassifier()
#     model.load_model(os.path.join(MODEL_DIR, "catboost_user_device_model.cbm"))
#     threshold = float(open(os.path.join(MODEL_DIR, "catboost_user_device_threshold.txt")).read())

#     # 🛠️ Add iso_score if not already in df
#     if "iso_score" not in df.columns:
#         iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
#         # Use numeric features only
#         num_df = df.select_dtypes(include=[np.number]).copy()
#         iso.fit(num_df)
#         df["iso_score"] = -iso.decision_function(num_df)
#     original_df = df.copy()

#     df = align_features(df, model)

#     probs = model.predict_proba(df)[:, 1]
#     df["Fraud_Score"] = probs
#     df["Fraud_Prediction"] = (probs >= threshold).astype(int)

#     fraud_cases = df[df["Fraud_Prediction"] == 1].copy()

#     # Generate and save fraud profiles
#     profiles = []
#     # for _, row in fraud_cases.iterrows():
#     #     profile = generate_fraud_profile(
#     #         user_id=row["user_id"],
#     #         device_id=row["device_hash"],
#     #         model=model,
#     #         X_row=row.drop(["user_id", "device_hash", "Fraud_Score", "Fraud_Prediction"])
#     #     )
#     #     profiles.append(profile)
#     for idx, row in fraud_cases.iterrows():
#         original_row = original_df.loc[idx] if idx in original_df.index else {}
#         profiles = generate_fraud_profile(
#             user_id=original_row.get("user_id", "unknown"),
#             device_id=original_row.get("device_hash", "unknown"),
#             model=model,
#             X_row=row.drop(["Fraud_Score", "Fraud_Prediction"], errors="ignore")
#         )
#     # Save JSON and log
#     with open("fraud_profiles.json", "w") as f:
#         json.dump(profiles, f, indent=4)

#     os.makedirs("logs", exist_ok=True)
#     with open("logs/user_device_log.txt", "a") as f:
#         for p in profiles:
#             f.write(json.dumps(p) + "\n")

#     explainer = shap.TreeExplainer(model)
#     X_sample = df.drop(["user_id", "device_hash"], axis=1, errors='ignore').sample(1)  
    
    
#     shap_values = explainer.shap_values(X_sample)

#     shap.plots.waterfall(
#         shap.Explanation(
#             values=shap_values[1][0],
#             base_values=explainer.expected_value[1],
#             data=X_sample.iloc[0]
#         )
#     )
#     print(f"✅ {len(profiles)} fraud profiles saved to fraud_profiles.json")
#     return fraud_cases
def detect_user_device_fraud(df):
    model = CatBoostClassifier()
    model.load_model(os.path.join(MODEL_DIR, "catboost_user_device_model.cbm"))
    threshold = float(open(os.path.join(MODEL_DIR, "catboost_user_device_threshold.txt")).read())

    # Save a copy before feature alignment
    original_df = df.copy()

    # If iso_score not already computed, add it
    if "iso_score" not in df.columns:
        iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        num_df = df.select_dtypes(include=[np.number]).copy()
        iso.fit(num_df)
        df["iso_score"] = -iso.decision_function(num_df)

    # Align features for the model
    df = align_features(df, model)

    # Get fraud probabilities
    probs = model.predict_proba(df)[:, 1]
    df["Fraud_Score"] = probs
    df["Fraud_Prediction"] = (probs >= threshold).astype(int)

    fraud_cases = df[df["Fraud_Prediction"] == 1].copy()

    # Generate profiles
    profiles = []
    for idx, row in fraud_cases.iterrows():
        original_row = original_df.loc[idx] if idx in original_df.index else {}
        profile = generate_fraud_profile(
            user_id=original_row.get("user_id", "unknown"),
            device_id=original_row.get("device_hash", "unknown"),
            model=model,
            X_row=row.drop(["Fraud_Score", "Fraud_Prediction"], errors="ignore")
        )
        profiles.append(profile)

    # Save outputs
    os.makedirs("logs", exist_ok=True)
    with open("fraud_profiles.json", "w") as f:
        json.dump(profiles, f, indent=4)

    with open("logs/user_device_log.txt", "a") as f:
        for p in profiles:
            f.write(json.dumps(p) + "\n")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_input = fraud_cases[model.feature_names_].sample(1)
    shap_values = explainer.shap_values(shap_input)

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0] if isinstance(shap_values, list) else shap_values[0],
            base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=shap_input.iloc[0]
        )
    )

    print(f"✅ {len(profiles)} fraud profiles saved to fraud_profiles.json")
    return fraud_cases
