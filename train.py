# import os
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.ensemble import IsolationForest
# from sklearn.metrics import classification_report, roc_auc_score

# # Paths
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_DIR = os.path.join(BASE_DIR, "processed_data")

# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # Load preprocessed data
# banksim = pd.read_csv(os.path.join(DATA_DIR, "processed_banksim.csv"))
# paysim = pd.read_csv(os.path.join(DATA_DIR, "processed_paysim.csv"))
# creditcard = pd.read_csv(os.path.join(DATA_DIR, "processed_creditcard.csv"))

# # Train function
# def train_model(X, y, model, model_name):
#     """Train model, evaluate, and save it"""
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     # Evaluation
#     print(f"🔹 Model: {model_name}")
#     print(classification_report(y_test, y_pred))
#     print(f"AUC-ROC: {roc_auc_score(y_test, y_pred)}\n")

#     # Save model
#     joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}.pkl"))

# # Train Random Forest
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# train_model(banksim.drop(columns=["fraud"]), banksim["fraud"], rf_model, "random_forest")

# # Train XGBoost
# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
# train_model(paysim.drop(columns=["isFraud"]), paysim["isFraud"], xgb_model, "xgboost")


# iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
# iso_forest.fit(creditcard.drop(columns=["Class"]))

# # Save Isolation Forest
# joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
# print(" Isolation Forest trained and saved!")

import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier 
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, accuracy_score,f1_score,confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
banksim = pd.read_csv(os.path.join(DATA_DIR, "processed_banksim.csv"))
paysim = pd.read_csv(os.path.join(DATA_DIR, "processed_paysim.csv"))
creditcard = pd.read_csv(os.path.join(DATA_DIR, "processed_creditcard.csv"))
ato_df = pd.read_csv(os.path.join(DATA_DIR, "processed_account_takeover.csv"))
BEHAVIORAL_FEATURES = [
    'key_variance', 'key_speed', 'mouse_entropy',
    'movement_speed', 'session_events', 'unique_actions',
    'click_to_scroll_ratio', 'event_density', 'night_access',
    'session_hour'
]

FINGERPRINT_FEATURES = [
    'user_agent', 'screen_width', 'screen_height', 'timezone',
    'language', 'cpu_cores', 'gpu_renderer', 'platform', 'device_hash',
    'device_user_count', 'user_device_count'
]


# rf_param_grid = {
#     "clf__n_estimators": [100, 200],
#     "clf__max_depth": [None, 10, 20],
#     "clf__min_samples_split": [2, 5],
# }

# xgb_param_grid = {
#     "clf__n_estimators": [100, 200],
#     "clf__max_depth": [3, 6],
#     "clf__learning_rate": [0.01, 0.05],
#     "clf__scale_pos_weight": [1, 3, 5],
# }

# lgb_param_grid = {
#     "clf__n_estimators": [100, 200],
#     "clf__max_depth": [-1, 10, 20],
#     "clf__learning_rate": [0.01, 0.05],
# }
# def tune_rf(X_train, y_train):
#     rf = RandomForestClassifier(random_state=42, class_weight='balanced')
#     pipe = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", rf)])
#     cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     grid = GridSearchCV(estimator=pipe, param_grid=rf_param_grid,
#                         scoring="f1", cv=cv, n_jobs=-1, verbose=1)
#     grid.fit(X_train, y_train)
#     print(" Best RF params:", grid.best_params_, "| CV-F1:", grid.best_score_)
#     return grid.best_estimator_

# def tune_xgb(X_train, y_train):
#     xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
#     pipe = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", xgb)])
#     cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     grid = GridSearchCV(estimator=pipe, param_grid=xgb_param_grid,
#                         scoring="f1", cv=cv, n_jobs=-1, verbose=1)
#     grid.fit(X_train, y_train)
#     print(" Best XGB params:", grid.best_params_, "| CV-F1:", grid.best_score_)
#     return grid.best_estimator_

# def tune_lgb(X_train, y_train):
#     lgbm = lgb.LGBMClassifier(random_state=42)
#     pipe = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", lgbm)])
#     cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     grid = GridSearchCV(estimator=pipe, param_grid=lgb_param_grid,
#                         scoring="f1", cv=cv, n_jobs=-1, verbose=1)
#     grid.fit(X_train, y_train)
#     print(" Best LGBM params:", grid.best_params_, "| CV-F1:", grid.best_score_)
#     return grid.best_estimator_

# def get_ensemble_model():
#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
#     lgbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05)

#     ensemble = VotingClassifier(
#         estimators=[
#             ('rf', rf),
#             ('xgb', xgb),
#             ('lgbm', lgbm)
#         ],
#         voting='soft'
#     )
#     return ensemble
# def get_ensemble_model():
#     rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
#     xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=3)
#     lgbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, class_weight='balanced')

#     return VotingClassifier(
#         estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
#         voting='soft'
#     )

# def analyze_keystroke_dynamics(keystroke_timings):
#     try:
#         intervals = np.diff(json.loads(keystroke_timings))
#         return pd.Series({
#             'key_variance': np.var(intervals),
#             'key_speed': np.mean(intervals),
#             'key_entropy': np.log(np.var(intervals) + 1e-6)
#         })
#     except:
#         return pd.Series({'key_variance': 0, 'key_speed': 0, 'key_entropy': 0})

# def analyze_mouse_dynamics(mouse_trajectory):
#     try:
#         coords = np.array(json.loads(mouse_trajectory))
#         dxdy = np.diff(coords, axis=0)
#         angles = np.arctan2(dxdy[:,1], dxdy[:,0])
#         return pd.Series({
#             'mouse_entropy': np.var(angles),
#             'movement_speed': np.mean(np.linalg.norm(dxdy, axis=1)),
#             'curvature': np.mean(np.abs(np.diff(angles)))
#         })
#     except:
#         return pd.Series({'mouse_entropy': 0, 'movement_speed': 0, 'curvature': 0})

# Train + save function
# def train_model(X, y, model, model_name):
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     print(f"🔹 Model: {model_name}")
#     print(classification_report(y_test, y_pred))
#     print(f"AUC-ROC: {roc_auc_score(y_test, y_pred):.4f}")

#     joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}.pkl"))

#     if hasattr(model, "predict_proba"):
#         y_probs = model.predict_proba(X_test)[:, 1]
#         precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
#         youden = precision + recall - 1
#         best_thresh = thresholds[np.argmax(youden)]

#         with open(os.path.join(MODEL_DIR, f"{model_name}_threshold.txt"), "w") as f:
#             f.write(str(best_thresh))

#         print(f" Saved threshold: {best_thresh:.4f}\n")
# def train_user_device_model(filepath):
#     print(" Loading preprocessed data from:", filepath)
#     df = pd.read_csv(filepath)

#     # Target column
#     y = df['suspicious']
    
#     # Drop non-feature columns
#     if 'user_id' in df.columns:
#         df.drop(columns=['user_id', 'suspicious'], inplace=True)

#     X = df[BEHAVIORAL_FEATURES + FINGERPRINT_FEATURES]

#     # Train-test split
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

#     # Apply SMOTE to balance training data
#     print(" Applying SMOTE for class balancing...")
#     smote = SMOTE(random_state=42)
#     X_train, y_train = smote.fit_resample(X_train, y_train)


#     print(" Training LightGBM model...")
#     model = lgb.LGBMClassifier(
#         n_estimators=100,
#         learning_rate=0.05,
#     )

#     model.fit(X_train, y_train)

#     # Evaluation
#     y_pred = model.predict(X_test)
#     print(" Accuracy:", accuracy_score(y_test, y_pred))
#     print(" Report:\n", classification_report(y_test, y_pred))

#     # Save model
#     joblib.dump(model, os.path.join(MODEL_DIR, "user_device_lightgbm.pkl"))
#     print(f" Model saved to {os.path.join(MODEL_DIR, 'user_device_lightgbm.pkl')}")

# # 1. Random Forest → banksim
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# train_model(banksim.drop(columns=["fraud"]), banksim["fraud"], rf_model, "random_forest")

# # 2. XGBoost → paysim
# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
# train_model(paysim.drop(columns=["isFraud"]), paysim["isFraud"], xgb_model, "xgboost_paysim")

# # 3. XGBoost → account takeover
# ato_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
# train_model(ato_df.drop(columns=["is_ato"]), ato_df["is_ato"], ato_model, "xgboost_ato")

# # 4. Isolation Forest → creditcard
# iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
# iso_model.fit(creditcard.drop(columns=["Class"]))
# joblib.dump(iso_model, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
# print(" Isolation Forest saved.\n")
# def train_user_device_model(filepath):
#     print(" Loading preprocessed data from:", filepath)
#     df = pd.read_csv(filepath)

#     y = df['suspicious']
#     if 'suspicious' in df.columns:
#         df.drop(columns=['suspicious'], inplace=True)

#     X = df[BEHAVIORAL_FEATURES + FINGERPRINT_FEATURES]

    

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     sm = SMOTE(random_state=42)
#     X_train, y_train = sm.fit_resample(X_train, y_train)
#     print(" Training ensemble model (RF + XGBoost + LightGBM)...")
#     model = get_ensemble_model()
#     model.fit(X_train, y_train)
#     y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for suspicious=1
#     precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
#     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
#     best_thresh = thresholds[f1_scores.argmax()]
#     print(f" Best Threshold for F1: {best_thresh:.3f}")
#     y_pred = (y_probs >= best_thresh).astype(int)

#     print(" Accuracy:", accuracy_score(y_test, y_pred))
#     print(" Report:\n", classification_report(y_test, y_pred))

#     joblib.dump(model, os.path.join(MODEL_DIR, "user_device_ensemble.pkl"))
#     print(" Ensemble model saved as user_device_ensemble.pkl")
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     lgbm_model = model.named_estimators_['lgbm']  # Extract LightGBM model from ensemble
#     feat_importance = pd.Series(lgbm_model.feature_importances_, index=X.columns)
#     feat_importance = feat_importance.sort_values(ascending=False)

#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=feat_importance.values[:15], y=feat_importance.index[:15])
#     plt.title("Top 15 Feature Importances")
#     plt.tight_layout()
#     plt.show()


# def train_user_device_model(filepath):
#     print(" Loading preprocessed data from:", filepath)
#     df = pd.read_csv(filepath)

#     # 1. Split off labels
#     y = df.pop("suspicious")
#     X = df[BEHAVIORAL_FEATURES + FINGERPRINT_FEATURES]
#     iso = IsolationForest(contamination=0.1, random_state=42)
#     iso.fit(X)
#     X["iso_score"] = -iso.decision_function(X)
#     BEHAVIORAL_FEATURES.append("iso_score")
#     # 2. Train/Test split
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     all_preds = np.zeros(len(X))
#     all_labels = np.zeros(len(X))

#     # 3. Initialize CatBoost with class weights (fraud is rarer)
#     print(" Training CatBoostClassifier for user-device profiling…")
#     model = CatBoostClassifier(
#         iterations=2000,
#         learning_rate=0.01,
#         depth=6,
#         class_weights=[1, 5],      # adjust if your fraud:normal ratio is different
#         eval_metric="AUC",
#         verbose=100,
#         random_state=42
#     )

#     # 4. Fit with early stopping on the held-out test set
#     model.fit(
#         X_train, y_train,
#         eval_set=(X_test, y_test),
#         early_stopping_rounds=50
#     )

#     # 5. Get predicted probabilities on the untouched test set
#     y_probs = model.predict_proba(X_test)[:, 1]

#     # 6. Find the best threshold by G-mean (balance recall and specificity)
#     precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
#     best_gmean = 0.0
#     best_thresh = 0.5

#     for t in thresholds:
#         y_pred_temp = (y_probs >= t).astype(int)
#         tn, fp, fn, tp = confusion_matrix(y_test, y_pred_temp, labels=[0, 1]).ravel()
#         recall_pos = tp / (tp + fn + 1e-6)
#         specificity = tn / (tn + fp + 1e-6)
#         gmean = (recall_pos * specificity) ** 0.5

#         if gmean > best_gmean:
#             best_gmean, best_thresh = gmean, t

#     print(f" Best G-mean threshold: {best_thresh:.3f}  (G-mean = {best_gmean:.3f})")
#     y_pred = (y_probs >= best_thresh).astype(int)

#     # 7. Evaluate final predictions
#     print(" Final Accuracy:", accuracy_score(y_test, y_pred))
#     print(" Final Classification Report:\n", classification_report(y_test, y_pred))

#     # 8. (Optional) Show top CatBoost feature importances
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     feat_imp = pd.Series(
#         model.get_feature_importance(), index=X.columns
#     ).sort_values(ascending=False)

#     print(" Top CatBoost Features:")
#     print(feat_imp.head(10))

#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=feat_imp.values[:15], y=feat_imp.index[:15])
#     plt.title("Top 15 CatBoost Feature Importances")
#     plt.tight_layout()
#     plt.show()

#     # 9. Save the trained CatBoost model and threshold
#     model.save_model(os.path.join(MODEL_DIR, "catboost_user_device_model.cbm"))
#     with open(os.path.join(MODEL_DIR, "catboost_user_device_threshold.txt"), "w") as f:
#         f.write(str(best_thresh))

#     print(" CatBoost model saved to 'catboost_user_device_model.cbm'")
#     print(" Threshold saved to 'catboost_user_device_threshold.txt'")
def generate_balanced_data(filepath):
    df = pd.read_csv(filepath)
    y = df["suspicious"]
    X = df.drop(columns=["suspicious"])

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_bal, columns=X.columns)
    df_balanced["suspicious"] = y_bal
    balanced_path = os.path.join(DATA_DIR, "balanced_user_device_data.csv")
    df_balanced.to_csv(balanced_path, index=False)

    print(" Balanced dataset saved to:", balanced_path)
    return balanced_path


def train_user_device_model(filepath):
    print(" Loading preprocessed data from:", filepath)
    df = pd.read_csv(filepath)

    # 1. Split off labels
    y = df.pop("suspicious")
    X = df[BEHAVIORAL_FEATURES + FINGERPRINT_FEATURES].copy()

    # 2. Add unsupervised anomaly score (Isolation Forest)
    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    X["iso_score"] = -iso.fit(X).decision_function(X)

    # 3. 5-Fold Stratified Cross-Validation to tune threshold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds = []
    preds, trues = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"⏳ Starting fold {fold+1}/5…")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            class_weights=[1, 5],
            eval_metric="AUC",
            verbose=0,
            random_state=42
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30)

        y_probs = model.predict_proba(X_test)[:, 1]
        precision, recall, thresh = precision_recall_curve(y_test, y_probs)

        best_gmean = 0.0
        best_thresh = 0.5
        for t in thresh:
            y_temp = (y_probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_temp).ravel()
            recall_pos = tp / (tp + fn + 1e-6)
            specificity = tn / (tn + fp + 1e-6)
            gmean = (recall_pos * specificity) ** 0.5
            if gmean > best_gmean:
                best_thresh = t
                best_gmean = gmean

        print(f"   Fold {fold+1} best threshold = {best_thresh:.3f} (G-mean={best_gmean:.3f})")
        thresholds.append(best_thresh)
        preds.extend((y_probs >= best_thresh).astype(int))
        trues.extend(y_test)

    print(" CV Accuracy:", accuracy_score(trues, preds))
    print(" CV Classification Report:\n", classification_report(trues, preds))

    # 4. Train final model on full data with best threshold
    final_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        class_weights=[1, 5],
        eval_metric="AUC",
        verbose=100,
        random_state=42
    )
    final_model.fit(X, y)
    final_thresh = np.median(thresholds)

    # 5. Save model and threshold
    final_model.save_model(os.path.join(MODEL_DIR, "catboost_user_device_model.cbm"))
    with open(os.path.join(MODEL_DIR, "catboost_user_device_threshold.txt"), "w") as f:
        f.write(str(final_thresh))
    print(" Final model saved to 'catboost_user_device_model.cbm'")

    # 6. Feature Importance
    feat_imp = pd.Series(final_model.get_feature_importance(), index=X.columns).sort_values(ascending=False)
    print(" Top CatBoost Features:\n", feat_imp.head(10))

# if __name__ == "__main__":
#     train_user_device_model(os.path.join(DATA_DIR, "processed_user_device_profiling.csv"))
# if __name__ == "__main__": 
#     filepath = os.path.join(DATA_DIR, "processed_user_device_data_enhanced_xxl.csv")
#     train_user_device_model(filepath)
if __name__ == "__main__":
    original_path = os.path.join(DATA_DIR, "processed_user_device_data_enhanced_xxl.csv")
    balanced_path = generate_balanced_data(original_path)
    train_user_device_model(balanced_path)
 
