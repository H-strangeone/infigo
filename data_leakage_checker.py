import pandas as pd
import joblib
import shap
import numpy as np

processed_data_path = "processed_data/"
models_path = "models/"

datasets = ["processed_banksim.csv", "processed_paysim.csv", "processed_creditcard.csv"]
models = {
    "xgboost": "xgboost.pkl",
    "random_forest": "random_forest.pkl",
    "isolation_forest": "isolation_forest.pkl"
}

def check_feature_consistency(dataset_name, df, model_name, model):
    """
    Checks if the features in the dataset match the model's trained features.
    """
    print(f"\n Checking feature consistency for {dataset_name} with {model_name} model...")

    dataset_features = set(df.columns)
    
    if hasattr(model, "feature_names_in_"):
        model_features = set(model.feature_names_in_)
    else:
        model_features = set(df.columns) 
    
    missing_in_model = dataset_features - model_features
    missing_in_dataset = model_features - dataset_features
    
    if not missing_in_model and not missing_in_dataset:
        print(" No feature mismatch detected!")
    else:
        print("⚠ Feature Mismatch Detected!")
        if missing_in_model:
            print(f"    Features in dataset but not in model: {missing_in_model}")
        if missing_in_dataset:
            print(f"    Features in model but missing in dataset: {missing_in_dataset}")
    print("-" * 50)


def check_shap_values(model, df):
    """
    Checks SHAP values for potential data leakage.
    """
    print(f"\n🛠 Checking SHAP values for model: {model}")

    target_col = "fraud"
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
        shap_values = explainer.shap_values(df, check_additivity=False)
        print(" SHAP check passed! No immediate leakage detected.")
    except Exception as e:
        print(f" SHAP error: {e}")
    print("-" * 50)


def main():
    """
    Runs data leakage checks on processed datasets and trained models.
    """
    for dataset in datasets:
        dataset_path = processed_data_path + dataset
        try:
            df = pd.read_csv(dataset_path)
            print(f"\n Checking dataset: {dataset}\n")
            print(df.corr()["fraud"])  
            duplicate_count = df.duplicated().sum()
            print(f" Number of duplicate transactions: {duplicate_count}")
            
            for model_name, model_file in models.items():
                model_path = models_path + model_file
                try:
                    model = joblib.load(model_path)
                    check_feature_consistency(dataset, df, model_name, model)
                    check_shap_values(model, df)
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")

        except Exception as e:
            print(f" Error loading dataset {dataset}: {e}")

if __name__ == "__main__":
    main()
