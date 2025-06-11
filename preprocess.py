# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# import os

# # Move up TWO levels to the project root (identity-fraud-api/)
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# def load_datasets():
#     bank_sim = pd.read_csv(os.path.join(BASE_DIR, "data", "banksim.csv"))
#     pay_sim = pd.read_csv(os.path.join(BASE_DIR, "data", "paysim.csv"))
#     credit_fraud = pd.read_csv(os.path.join(BASE_DIR, "data", "creditcard.csv"))
#     return bank_sim, pay_sim, credit_fraud


# # def load_datasets():
# #     """Load all datasets."""
# #     bank_sim = pd.read_csv("data/banksim.csv")
# #     pay_sim = pd.read_csv("data/paysim.csv")
# #     credit_fraud = pd.read_csv("data/creditcard.csv")
# #     return bank_sim, pay_sim, credit_fraud

# def preprocess_data(df, target_column):
#     """Clean and preprocess the dataset."""
#     df = df.drop_duplicates()
    
#     df = df.dropna()
    
#     for col in df.select_dtypes(include=['object']).columns:
#         df[col] = LabelEncoder().fit_transform(df[col])
    
#     if df[target_column].dtype == 'float64':
#         df[target_column] = (df[target_column] > 0).astype(int)
    
#     scaler = StandardScaler()
#     features = df.drop(columns=[target_column])
#     df[features.columns] = scaler.fit_transform(features)
    
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
    
#     smote = SMOTE(sampling_strategy=0.5, random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
    
#     return X_resampled, y_resampled

# def main():
#     bank_sim, pay_sim, credit_fraud = load_datasets()
    
#     datasets = {
#         "banksim": (bank_sim, "fraud"),
#         "paysim": (pay_sim, "isFraud"),
#         "creditcard": (credit_fraud, "Class")
#     }
    
#     for name, (df, target) in datasets.items():
#         X, y = preprocess_data(df, target)
#         processed_df = pd.DataFrame(X, columns=df.drop(columns=[target]).columns)
#         processed_df[target] = y
#         output_dir = os.path.join(BASE_DIR, "processed_data")
#         os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

#         output_file = os.path.join(output_dir, f"processed_{name}.csv")
#         processed_df.to_csv(output_file, index=False)

#         print(f"Processed {name} dataset saved at: {output_file}")

# if __name__ == "__main__":
#     main()
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import make_column_selector
import os

# Move up TWO levels to the project root (identity-fraud-api/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
def load_datasets():
    bank_sim = pd.read_csv(os.path.join(BASE_DIR, "data", "banksim.csv"))
    pay_sim = pd.read_csv(os.path.join(BASE_DIR, "data", "paysim.csv"))
    credit_fraud = pd.read_csv(os.path.join(BASE_DIR, "data", "creditcard.csv"))
    credit_fraud = pd.read_csv(os.path.join(BASE_DIR, "data", "account_takeover_synthetic.csv"))
    ato_df = pd.read_csv(os.path.join(DATA_DIR, "account_takeover_synthetic.csv"))
    return bank_sim, pay_sim, credit_fraud,ato_df


ato_df = pd.read_csv(os.path.join(DATA_DIR, "account_takeover_synthetic.csv"))
# def load_datasets():
#     """Load all datasets."""
#     bank_sim = pd.read_csv("data/banksim.csv")
#     pay_sim = pd.read_csv("data/paysim.csv")
#     credit_fraud = pd.read_csv("data/creditcard.csv")
#     return bank_sim, pay_sim, credit_fraud

# def preprocess_data(df, target_column):
#     """Clean and preprocess the dataset."""
#     df = df.drop_duplicates()
    
#     df = df.dropna()
    
#     for col in df.select_dtypes(include=['object']).columns:
#         df[col] = LabelEncoder().fit_transform(df[col])
    
#     if df[target_column].dtype == 'float64':
#         df[target_column] = (df[target_column] > 0).astype(int)
    
#     scaler = StandardScaler()
#     features = df.drop(columns=[target_column])
#     df[features.columns] = scaler.fit_transform(features)
    
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
    
#     # smote = SMOTE(sampling_strategy=0.5, random_state=42)
#     # X_resampled, y_resampled = smote.fit_resample(X, y)
    
#     # return X_resampled, y_resampled
#     X_scaled = scaler.fit_transform(X)

#     sm = SMOTE(random_state=42)
#     X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

#     processed = pd.DataFrame(X_resampled, columns=X.columns)
#     processed[target_column] = y_resampled
#     return processed

# def main():
#     bank_sim, pay_sim, credit_fraud = load_datasets()
    
#     datasets = {
#         "banksim": (bank_sim, "fraud"),
#         "paysim": (pay_sim, "isFraud"),
#         "creditcard": (credit_fraud, "Class")
#     }
    
#     for name, (df, target) in datasets.items():
#         X, y = preprocess_data(df, target)
#         processed_df = pd.DataFrame(X, columns=df.drop(columns=[target]).columns)
#         processed_df[target] = y
#         output_dir = os.path.join(BASE_DIR, "processed_data")
#         os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

#         output_file = os.path.join(output_dir, f"processed_{name}.csv")
#         processed_df.to_csv(output_file, index=False)

#         print(f"Processed {name} dataset saved at: {output_file}")

# if __name__ == "__main__":
#     main()

# import pandas as pd
# import numpy as np
# import json
# from sklearn.preprocessing import LabelEncoder, StandardScaler
def preprocess_user_device_data(filepath):
    df = pd.read_csv(filepath)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Encode categorical columns
    label_cols = ['ip_address','user_agent', 'timezone', 'language', 'gpu_renderer', 'platform']
    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Device fingerprint
    # df['device_hash'] = df[['ip_address', 'user_agent', 'screen_width', 'screen_height']] \
    #     .astype(str).agg('-'.join, axis=1).apply(lambda x: hash(x))
    if all(c in df.columns for c in ["ip_address", "user_agent","screen_width", "screen_height"]):
        df["device_hash"] = (
            df[["ip_address", "user_agent", "screen_width", "screen_height"]] 
            .astype(str)
            .agg("-".join, axis=1)
            .apply(lambda x: hash(x))
        )
    else:
        df["device_hash"] = 0
    # Analyze keystroke dynamics
    def analyze_keystroke(timings):
        try:
            intervals = np.diff(json.loads(timings))
            return pd.Series({
                'key_variance': np.var(intervals),
                'key_speed': np.mean(intervals),
                'key_entropy': np.log(np.var(intervals) + 1e-6)
            })
        except:
            return pd.Series({'key_variance': 0, 'key_speed': 0, 'key_entropy': 0})

    if "keystroke_timings" in df.columns:
        df[["key_variance", "key_speed", "key_entropy"]] = df["keystroke_timings"].apply(analyze_keystroke)
    else:
        df[["key_variance", "key_speed", "key_entropy"]] = (0, 0, 0)

    # Analyze mouse dynamics
    def analyze_mouse(mouse_json):
        try:
            coords = np.array(json.loads(mouse_json))
            dxdy = np.diff(coords, axis=0)
            angles = np.arctan2(dxdy[:, 1], dxdy[:, 0])
            return pd.Series({
                'mouse_entropy': np.var(angles),
                'movement_speed': np.mean(np.linalg.norm(dxdy, axis=1)),
                'curvature': np.mean(np.abs(np.diff(angles)))
            })
        except:
            return pd.Series({'mouse_entropy': 0, 'movement_speed': 0, 'curvature': 0})

    if "mouse_trajectory" in df.columns:
        df[["mouse_entropy", "movement_speed", "curvature"]] = df["mouse_trajectory"].apply(analyze_mouse)
    else:
        df[["mouse_entropy", "movement_speed", "curvature"]] = (0, 0, 0)

    if 'timestamp' in df.columns and "user_id" in df.columns:
        df['session_start'] = df['timestamp'].dt.floor('5min')
        df['session_id'] = df.groupby(['user_id', 'session_start']).ngroup()

        session_features = df.groupby('session_id').agg({
            'timestamp': 'count',                    # number of events
            'device_hash': 'nunique',                # unique devices/actions
        }).rename(columns={'timestamp': 'session_events', 'device_hash': 'unique_actions'})

        df = df.merge(session_features, on='session_id', how='left')
    else:
        df["session_events"]   = 0
        df["unique_actions"]   = 0
        df["session_id"]       = 0
    # Extra behavioral features for profiling
    if "click_count" in df.columns and "scroll_count" in df.columns:
        df["click_to_scroll_ratio"] = df["click_count"] / (df["scroll_count"] + 1)
    else:
        # Create a placeholder (all zeros), or skip entirely
        df["click_to_scroll_ratio"] = 0.0

    # If you have a column that defines how long the session lasted (e.g. seconds),
    # you can compute event_density. Otherwise skip or fill with zeros:
    if "session_duration" in df.columns:
        df["event_density"] = df["session_events"] / (df["session_duration"] + 1)
    else:
        df["event_density"] = 0.0

    # If you want to create “session_hour” and “night_access”, only do so if “timestamp” was present:
    if "timestamp" in df.columns:
        df["session_hour"] = df["timestamp"].dt.hour
        df["night_access"] = df["session_hour"].apply(lambda h: 1 if h < 6 else 0)
    else:
        df["session_hour"] = 0
        df["night_access"] = 0
    if "device_hash" in df.columns and "user_id" in df.columns:
        df["device_user_count"] = df.groupby("device_hash")["user_id"].transform("nunique")
        df["user_device_count"] = df.groupby("user_id")["device_hash"].transform("nunique")
    else:
        df["device_user_count"] = 0
        df["user_device_count"] = 0
    # --- Step 7: Drop unused/raw columns ---
    df.drop(columns=[
        'keystroke_timings', 'mouse_trajectory', 'ip_address',
        'timestamp', 'session_id', 'session_start'
    ], inplace=True, errors='ignore')
    # Drop any datetime columns before scaling
    features = df.drop(columns=['user_id', 'suspicious'], errors='ignore')
    datetime_cols = features.select_dtypes(include=['datetime64', 'object']).columns
    features = features.drop(columns=datetime_cols)

    # Now scale safely
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    processed_df = pd.DataFrame(X_scaled, columns=features.columns)

    # Make sure we add back user_id & device_hash as “metadata” (unscaled)
    if "user_id" in df.columns:
        processed_df["user_id"]      = df["user_id"].values
    if "device_hash" in df.columns:
        processed_df["device_hash"]  = df["device_hash"].values

    # Reattach the label
    if "suspicious" in df.columns:
        processed_df["suspicious"] = df["suspicious"].values

    # 13) Finally save to CSV
    processed_df.to_csv(
        os.path.join(OUTPUT_DIR, "processed_user_device_data_enhanced_xxl.csv"),
        index=False
    )
    print("✅ Saved as 'processed_user_device_data_enhancedllll.csv'")
    # y = df['suspicious'] if 'suspicious' in df.columns else None
    # return X_scaled, y, features.columns



# if __name__ == "__main__":
if __name__ == "__main__":
    processed_udp = preprocess_user_device_data(os.path.join(DATA_DIR, "synthetic_user_device_data_enhanced_xlarge.csv"))
    X_scaled, y, columns = processed_udp
    processed_df = pd.DataFrame(X_scaled, columns=columns)
    if y is not None:
        processed_df["suspicious"] = y.values
    processed_df.to_csv(os.path.join(OUTPUT_DIR, "processed_user_device_data_enhanced.csv2"), index=False)
    print("✅ Saved as 'processed_user_device_data_xl.csv'")
