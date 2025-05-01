import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load test data
banksim = pd.read_csv(os.path.join(DATA_DIR, "processed_banksim.csv"))
paysim = pd.read_csv(os.path.join(DATA_DIR, "processed_paysim.csv"))
creditcard = pd.read_csv(os.path.join(DATA_DIR, "processed_creditcard.csv"))

# Load models
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

# Prepare test sets
X_banksim = banksim.drop(columns=["fraud"])
y_banksim = banksim["fraud"]

X_paysim = paysim.drop(columns=["isFraud"])
y_paysim = paysim["isFraud"]

X_creditcard = creditcard.drop(columns=["Class"])
y_creditcard = creditcard["Class"]

# Function to evaluate models
def evaluate_model(model, X, y, name):
    y_pred = model.predict(X)
    print(f"🔹 Evaluation for {name}:")
    print(classification_report(y, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y, y_pred)}\n")

# Evaluate models
evaluate_model(rf_model, X_banksim, y_banksim, "Random Forest")
evaluate_model(xgb_model, X_paysim, y_paysim, "XGBoost")
evaluate_model(iso_forest, X_creditcard, y_creditcard, "Isolation Forest")
