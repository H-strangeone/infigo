import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")

MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load preprocessed data
banksim = pd.read_csv(os.path.join(DATA_DIR, "processed_banksim.csv"))
paysim = pd.read_csv(os.path.join(DATA_DIR, "processed_paysim.csv"))
creditcard = pd.read_csv(os.path.join(DATA_DIR, "processed_creditcard.csv"))

# Train function
def train_model(X, y, model, model_name):
    """Train model, evaluate, and save it"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    print(f"🔹 Model: {model_name}")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred)}\n")

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}.pkl"))

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
train_model(banksim.drop(columns=["fraud"]), banksim["fraud"], rf_model, "random_forest")

# Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
train_model(paysim.drop(columns=["isFraud"]), paysim["isFraud"], xgb_model, "xgboost")


iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(creditcard.drop(columns=["Class"]))

# Save Isolation Forest
joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
print(" Isolation Forest trained and saved!")