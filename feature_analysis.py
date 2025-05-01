import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

# Prompt user to select dataset
dataset_choice = input("Select dataset (banksim, creditcard, paysim): ").strip().lower()
file_paths = {
    "banksim": "processed_data/processed_banksim.csv",
    "creditcard": "processed_data/processed_creditcard.csv",
    "paysim": "processed_data/processed_paysim.csv"
}

if dataset_choice in file_paths:
    data = pd.read_csv(file_paths[dataset_choice])
    print(f"✅ Loaded {dataset_choice} dataset successfully!\n")
else:
    raise ValueError(" Invalid dataset selection. Choose from: banksim, creditcard, or paysim.")

# Detect target column
possible_targets = ["fraud_label", "is_fraud", "Class", "fraud"]
target = next((col for col in possible_targets if col in data.columns), None)

if target is None:
    print(f"No fraud label column found! Available columns: {data.columns.tolist()}")
    raise KeyError("Target fraud column not found. Please check dataset.")

print(f" Using '{target}' as the target column.\n")

# Convert fraud column to numeric if needed
data[target] = pd.to_numeric(data[target], errors='coerce')

# Remove non-numeric columns before correlation
numeric_data = data.select_dtypes(include=['number'])

# Drop columns with no variance (all unique values)
numeric_data = numeric_data.loc[:, numeric_data.nunique() > 1]

# Compute correlation with target
correlation_matrix = numeric_data.corr()
correlation_with_target = correlation_matrix[target].sort_values(ascending=False)
print(" Feature Correlation with Target:\n", correlation_with_target.dropna())  # Drop NaN values

# Plot heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")  # Removed 🔥 to avoid font issues
plt.show()

# Train a small XGBoost model for SHAP analysis
X = numeric_data.drop(columns=[target])
y = numeric_data[target]

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X, y)

# SHAP analysis
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Plot SHAP summary
shap.summary_plot(shap_values, X)
