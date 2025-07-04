# import streamlit as st
# import pandas as pd
# from test import detect_identity_fraud, detect_synthetic_fraud, detect_mule_accounts

# # Streamlit UI
# st.title("Fraud Detection System")

# # File uploader
# uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.write("### Uploaded Dataset:")
#     st.write(df.head())

#     # Buttons for different fraud detections
#     if st.button("Detect Identity Fraud"):
#         fraud_df = detect_identity_fraud(df)
#         st.write("### Identity Fraud Cases:")
#         st.write(fraud_df)

#     if st.button("Detect Synthetic Fraud"):
#         fraud_df = detect_synthetic_fraud(df)
#         st.write("### Synthetic Fraud Cases:")
#         st.write(fraud_df)

#     if st.button("Detect Mule Accounts"):
#         fraud_df = detect_mule_accounts(df)
#         st.write("### Mule Account Cases:")
#         st.write(fraud_df)
# import streamlit as st
# import pandas as pd
# import joblib
# import shap
# import streamlit_shap as st_shap
# import time

# # Streamlit Page Config
# st.set_page_config(
#     page_title="Fraud Detection App ",
#     page_icon="",
#     layout="wide"
# )

# # Animations
# def animate_success():
#     with st.spinner('Analyzing your data... '):
#         time.sleep(2)
#     st.success('Done! See results below! ')

# # Load your pre-trained model
# model = joblib.load("models/xgboost.pkl")  # adjust path if needed
# explainer = shap.Explainer(model)

# # Title
# st.title(" Fraud Detection System with Explainable AI")
# st.markdown("Upload your transactions to detect frauds and understand why they happen!")

# # Layout: 2 Columns
# left_col, right_col = st.columns([1, 2])

# with left_col:
#     uploaded_file = st.file_uploader(" Upload your CSV file", type=["csv"])

# with right_col:
#     st.info(" Please upload transaction data to detect frauds.\n\nSample format: [features used during training].")

# # If file uploaded
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.session_state["df_uploaded"] = df  # Save in session for later use

#     animate_success()

#     # Predictions
#     preds = model.predict(df)
#     df["Prediction"] = preds

#     # Show detected frauds
#     frauds = df[df["Prediction"] == 1]
#     st.subheader(" Defaulters (Detected Frauds):")
#     st.dataframe(frauds)

#     st.write("---")

#     # SHAP Analysis
#     st.subheader(" Top Reasons for Fraud Predictions (SHAP Summary)")
#     shap_values = explainer(df.drop(columns=["Prediction"]))  # exclude prediction column
#     st_shap.st_shap(shap.summary_plot(shap_values, df.drop(columns=["Prediction"]), plot_type="bar"), height=500)

#     # Optional: Number of frauds detected
#     st.success(f" Total Frauds Detected: {len(frauds)} out of {len(df)} records!")

# else:
#     st.warning("Please upload a file to begin fraud detection ")
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import shap
# import joblib
# import os
# import matplotlib.pyplot as plt
# from test import detect_identity_fraud, detect_synthetic_fraud, detect_mule_accounts

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# DATA_DIR = os.path.join(BASE_DIR, "data")

# # Load fraud models
# rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
# xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
# iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

# # Streamlit App
# st.set_page_config(page_title="Fraud Detection App ", layout="wide")
# st.title(" Fraud Detection Dashboard")

# st.sidebar.header("Upload your Dataset ")
# uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.success(" File Uploaded Successfully!")

#     # Show full data
#     st.subheader("Dataset Preview ")
#     st.dataframe(df, use_container_width=True)

#     # Fraud Detection Section
#     st.subheader("Fraud Detection Section ")

#     col1, col2, col3 = st.columns(3)

#     # Buttons for different frauds
#     if col1.button(" Identity Theft Check"):
#         identity_fraud = detect_identity_fraud(df)
#         st.warning(f" Identity Theft Cases Detected: {len(identity_fraud)}")
#         st.dataframe(identity_fraud)
#         csv = identity_fraud.to_csv(index=False).encode('utf-8')
#         st.download_button("⬇ Download Identity Thefts", data=csv, file_name='identity_fraud.csv', mime='text/csv')

#         # SHAP Importance Plot for Random Forest
#         st.subheader(" Feature Importance for Identity Fraud Detection")
#         explainer = shap.TreeExplainer(rf_model)
#         shap_values = explainer.shap_values(df.drop(columns=['Fraud_Prediction'], errors='ignore'))
#         shap.summary_plot(shap_values[1], df.drop(columns=['Fraud_Prediction'], errors='ignore'), plot_type="bar", show=False)
#         st.pyplot(plt.gcf())
#         plt.clf()

#     if col2.button(" Synthetic Identity Check"):
#         synthetic_fraud = detect_synthetic_fraud(df)
#         st.warning(f"Synthetic Identities Detected: {len(synthetic_fraud)}")
#         st.dataframe(synthetic_fraud)
#         csv = synthetic_fraud.to_csv(index=False).encode('utf-8')
#         st.download_button("⬇ Download Synthetic Identities", data=csv, file_name='synthetic_fraud.csv', mime='text/csv')

#         # SHAP Importance Plot for XGBoost
#         st.subheader(" Feature Importance for Synthetic Identity Detection")
#         explainer = shap.TreeExplainer(xgb_model)
#         shap_values = explainer.shap_values(df.drop(columns=['Fraud_Prediction'], errors='ignore'))
#         shap.summary_plot(shap_values, df.drop(columns=['Fraud_Prediction'], errors='ignore'), plot_type="bar", show=False)
#         st.pyplot(plt.gcf())
#         plt.clf()

#     if col3.button(" Mule Account Check"):
#         mule_fraud = detect_mule_accounts(df)
#         st.warning(f" Mule Accounts Detected: {len(mule_fraud)}")
#         st.dataframe(mule_fraud)
#         csv = mule_fraud.to_csv(index=False).encode('utf-8')
#         st.download_button("⬇ Download Mule Accounts", data=csv, file_name='mule_fraud.csv', mime='text/csv')

#         # SHAP Importance Plot for Isolation Forest
#         st.subheader(" Feature Importance for Mule Account Detection")
#         explainer = shap.TreeExplainer(iso_forest)
#         shap_values = explainer.shap_values(df.drop(columns=['Fraud_Prediction'], errors='ignore'))
#         shap.summary_plot(shap_values, df.drop(columns=['Fraud_Prediction'], errors='ignore'), plot_type="bar", show=False)
#         st.pyplot(plt.gcf())
#         plt.clf()

#     st.markdown("---")

#     # Animated Fraud Pie Chart
#     st.subheader("Fraud vs Normal Pie Chart ")
#     if 'Fraud_Prediction' in df.columns:
#         pie_data = df['Fraud_Prediction'].value_counts().reset_index()
#         pie_data.columns = ['Fraud_Label', 'Count']
#         pie_data['Fraud_Label'] = pie_data['Fraud_Label'].map({0: 'Normal', 1: 'Fraud', -1: 'Anomaly'})

#         fig = px.pie(pie_data, values='Count', names='Fraud_Label', title='Fraud vs Normal Accounts ',
#                      color_discrete_sequence=px.colors.sequential.RdBu)
#         st.plotly_chart(fig, use_container_width=True)

# else:
#     st.info(" Please upload a CSV file from the sidebar to begin!")
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import shap
# import joblib
# import os
# import matplotlib.pyplot as plt
# from test import detect_identity_fraud, detect_synthetic_fraud, detect_mule_accounts

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # Load models
# rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
# xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
# iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

# # Streamlit config
# st.set_page_config(page_title="Fraud Detection App ", layout="wide")
# st.title(" Fraud Detection Dashboard")

# # Preprocessing function
# def preprocess_data(df):
#     df = df.dropna()
#     df = df.apply(pd.to_numeric, errors='coerce')
#     df = df.dropna()
#     if len(df) > 5000:
#         df = df.sample(5000, random_state=42)
#     return df

# # SHAP plot function for Random Forest and XGBoost
# def plot_shap_summary(model, X):
#     explainer = shap.TreeExplainer(model)
#     sample_X = X.sample(min(300, len(X)), random_state=42)
#     shap_values = explainer.shap_values(sample_X)
    
#     if isinstance(shap_values, list):
#         # XGBoost returns list, pick class 1
#         shap.summary_plot(shap_values[1], sample_X, plot_type="bar", show=False)
#     else:
#         # RandomForest returns single array
#         shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    
#     st.pyplot(plt.gcf())
#     plt.clf()

# # Upload section
# st.sidebar.header("Upload your Dataset ")
# uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     df = preprocess_data(df)
#     st.success(" File Uploaded and Preprocessed!")

#     st.subheader(" Dataset Preview")
#     st.dataframe(df, use_container_width=True)

#     # Initial Pie Chart
#     st.subheader(" Initial Data Distribution")
#     if 'Class' in df.columns:
#         pie_data = df['Class'].value_counts().reset_index()
#         pie_data.columns = ['Label', 'Count']
#         pie_data['Label'] = pie_data['Label'].map({0: 'Normal', 1: 'Fraud', -1: 'Anomaly'})

#         fig = px.pie(pie_data, names='Label', values='Count', title="Normal vs Fraud Data ",
#                      color_discrete_sequence=px.colors.sequential.RdBu)
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("ℹ No 'Class' column found. Pie chart not available.")

#     st.markdown("---")

#     # Fraud Detection Section
#     st.subheader(" Fraud Detection Section")

#     col1, col2, col3 = st.columns(3)

#     if col1.button(" Detect Identity Theft"):
#         with st.spinner('Running Identity Theft Detection...'):
#             fraud_df = detect_identity_fraud(df)
#             st.warning(f"Identity Thefts Detected: {len(fraud_df)}")
#             st.dataframe(fraud_df)
#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("⬇ Download Identity Thefts", data=csv, file_name='identity_thefts.csv', mime='text/csv')
#             st.subheader(" Feature Importance for Identity Theft")
#             plot_shap_summary(rf_model, df.drop(columns=["Class"], errors="ignore"))

#     if col2.button(" Detect Synthetic Identities"):
#         with st.spinner('Running Synthetic Fraud Detection...'):
#             fraud_df = detect_synthetic_fraud(df)
#             st.warning(f" Synthetic Identities Detected: {len(fraud_df)}")
#             st.dataframe(fraud_df)
#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("⬇ Download Synthetic Identities", data=csv, file_name='synthetic_identities.csv', mime='text/csv')
#             st.subheader("Feature Importance for Synthetic Fraud")
#             plot_shap_summary(xgb_model, df.drop(columns=["Class"], errors="ignore"))

#     if col3.button(" Detect Mule Accounts"):
#         with st.spinner('Running Mule Account Detection...'):
#             fraud_df = detect_mule_accounts(df)
#             st.warning(f" Mule Accounts Detected: {len(fraud_df)}")
#             st.dataframe(fraud_df)
#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("⬇ Download Mule Accounts", data=csv, file_name='mule_accounts.csv', mime='text/csv')
#             st.subheader("Anomaly Detection (Mule Accounts) - No Feature Importance")
#             st.info("ℹ Isolation Forest does not provide SHAP feature importance.")

# else:
#     st.info(" Please upload a CSV file from the sidebar to begin!")

# app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import shap
# import joblib
# import os
# import matplotlib.pyplot as plt
# from test import detect_identity_fraud, detect_synthetic_fraud, detect_mule_accounts

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # Load models
# rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
# xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
# iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

# # Streamlit config
# st.set_page_config(page_title="Fraud Detection App ", layout="wide")
# st.title(" Fraud Detection Dashboard")

# # Preprocessing function
# def preprocess_data(df):
#     df = df.dropna()
#     df = df.apply(pd.to_numeric, errors='coerce')
#     df = df.dropna()
#     if len(df) > 5000:
#         df = df.sample(5000, random_state=42)
#     return df

# # SHAP plot function
# def plot_shap_summary(model, X, plot_type="bar"):
#     explainer = shap.TreeExplainer(model)
#     sample_X = X.sample(min(300, len(X)), random_state=42)
#     shap_values = explainer.shap_values(sample_X)

#     if isinstance(shap_values, list):
#         shap_values = shap_values[1]  # pick fraud class

#     shap.summary_plot(shap_values, sample_X, plot_type=plot_type, show=False)
#     st.pyplot(plt.gcf())
#     plt.clf()

# # Upload section
# st.sidebar.header("Upload your Dataset ")
# uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# # Fraud type tagger
# def tag_fraud(score):
#     if score >= 0.85:
#         return " Very Suspicious"
#     elif score >= 0.7:
#         return " Mild Anomaly"
#     else:
#         return " Normal"

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     df = preprocess_data(df)
#     st.success(" File Uploaded and Preprocessed!")

#     st.subheader(" Dataset Preview")
#     st.dataframe(df, use_container_width=True)

#     # Initial Pie Chart
#     st.subheader(" Initial Data Distribution")
#     if 'Class' in df.columns:
#         pie_data = df['Class'].value_counts().reset_index()
#         pie_data.columns = ['Label', 'Count']
#         pie_data['Label'] = pie_data['Label'].map({0: 'Normal', 1: 'Fraud', -1: 'Anomaly'})

#         fig = px.pie(pie_data, names='Label', values='Count', title="Normal vs Fraud Data ",
#                      color_discrete_sequence=px.colors.sequential.RdBu)
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("ℹNo 'Class' column found. Pie chart not available.")

#     st.markdown("---")

#     # Fraud Detection Section
#     st.subheader(" Fraud Detection Section")

#     col1, col2, col3 = st.columns(3)

#     if col1.button(" Detect Identity Theft"):
#         with st.spinner('Running Identity Theft Detection...'):
#             fraud_df = detect_identity_fraud(df)
#             fraud_df["Fraud Type"] = fraud_df["Fraud_Score"].apply(tag_fraud)
#             st.warning(f" Identity Thefts Detected: {len(fraud_df)}")
#             st.dataframe(fraud_df)

#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("⬇ Download Identity Thefts", data=csv, file_name='identity_thefts.csv', mime='text/csv')

#             st.subheader(" Feature Importance for Identity Theft")
#             plot_type = st.radio("Choose Plot Type:", ["Bar", "Beeswarm"], horizontal=True)
#             plot_shap_summary(rf_model, df.drop(columns=["Class"], errors="ignore"), plot_type=plot_type.lower())

#     if col2.button(" Detect Synthetic Identities"):
#         with st.spinner('Running Synthetic Fraud Detection...'):
#             fraud_df = detect_synthetic_fraud(df)
#             fraud_df["Fraud Type"] = fraud_df["Fraud_Score"].apply(tag_fraud)
#             st.warning(f" Synthetic Identities Detected: {len(fraud_df)}")
#             st.dataframe(fraud_df)

#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("⬇ Download Synthetic Identities", data=csv, file_name='synthetic_identities.csv', mime='text/csv')

#             st.subheader(" Feature Importance for Synthetic Fraud")
#             plot_type = st.radio("Choose Plot Type:", ["Bar", "Beeswarm"], horizontal=True, key="synthetic_plot")
#             plot_shap_summary(xgb_model, df.drop(columns=["Class"], errors="ignore"), plot_type=plot_type.lower())

#     if col3.button(" Detect Mule Accounts"):
#         with st.spinner('Running Mule Account Detection...'):
#             fraud_df = detect_mule_accounts(df)
#             fraud_df["Fraud Type"] = fraud_df["Anomaly_Score"].apply(lambda x: " Very Suspicious" if x > 0.2 else "⚠ Mild Anomaly")
#             st.warning(f" Mule Accounts Detected: {len(fraud_df)}")
#             st.dataframe(fraud_df)

#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("⬇ Download Mule Accounts", data=csv, file_name='mule_accounts.csv', mime='text/csv')

#             st.subheader(" Anomaly Detection (Mule Accounts) - No Feature Importance")
#             st.info("ℹ Isolation Forest does not provide SHAP feature importance.")

# else:
#     st.info(" Please upload a CSV file from the sidebar to begin!")
# app.py
# import numpy as np
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import shap
# import joblib
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# from test import detect_identity_fraud, detect_synthetic_fraud, detect_mule_accounts

# def safe_predict(model, df, drop_cols=None):
#     if drop_cols is None:
#         drop_cols = ["Fraud_Prediction", "Detection_Score", "Anomaly_Score", "Fraud_Score", "Fraud_Level"]
    
#     # Drop known post-processing columns
#     df = df.drop(columns=drop_cols, errors="ignore")
    
#     # Align features with model's training columns
#     model_features = model.feature_names_in_
#     df = df[[col for col in model_features if col in df.columns]]

#     return model.predict_proba(df)[:, 1]


# def plot_isolation_forest_anomalies(df, score_col='Anomaly_Score'):
#     st.subheader("Anomaly Score Distribution (Isolation Forest)")

#     fig, ax = plt.subplots()
#     sns.histplot(df[score_col], bins=30, kde=True, color='tomato', ax=ax)
#     ax.set_title("Anomaly Score Histogram")
#     st.pyplot(fig)

#     num_cols = df.select_dtypes(include=np.number).columns.tolist()
#     selected_cols = [col for col in num_cols if col != score_col][:2]

#     if len(selected_cols) == 2:
#         fig2 = px.scatter(
#             df, x=selected_cols[0], y=selected_cols[1],
#             color=score_col, title="2D Anomaly Visualization",
#             color_continuous_scale='Viridis'
#         )
#         st.plotly_chart(fig2, use_container_width=True)

#     top_anomalies = df.sort_values(score_col, ascending=False).head(10)
#     st.write("Top 10 Most Anomalous Records:")
#     st.dataframe(top_anomalies)

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # Load models
# rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
# xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
# iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

# # Streamlit config
# st.set_page_config(page_title="Fraud Detection App", layout="wide", initial_sidebar_state="expanded")
# st.title("Fraud Detection Dashboard")

# # Preprocessing
# def preprocess_data(df):
#     df = df.dropna()
#     df = df.apply(pd.to_numeric, errors='coerce')
#     df = df.dropna()
#     if len(df) > 5000:
#         df = df.sample(5000, random_state=42)
#     return df

# def find_label_column(df):
#     possible_labels = ['Class', 'isFraud', 'fraud', 'label', 'target']
#     for col in df.columns:
#         if col.lower() in [p.lower() for p in possible_labels]:
#             return col
#     return None

# def plot_shap_summary(model, X, plot_type):
#     explainer = shap.Explainer(model)
#     sample_X = X.sample(min(300, len(X)), random_state=42)
#     shap_values = explainer(sample_X)

#     st.subheader("Feature Importance")
#     if plot_type == "Bar Plot":
#         shap.plots.bar(shap_values, max_display=10, show=False)
#     else:
#         shap.plots.beeswarm(shap_values, max_display=20, show=False)

#     st.pyplot(plt.gcf())
#     plt.clf()

#     importance = np.abs(shap_values.values).mean(axis=0)
#     shap_df = pd.DataFrame({
#         'Feature': sample_X.columns,
#         'Importance': importance
#     }).sort_values('Importance', ascending=False).head(10)

#     st.dataframe(shap_df)
#     csv = shap_df.to_csv(index=False).encode('utf-8')
#     st.download_button("Download Feature Importance", data=csv, file_name="feature_importance.csv", mime='text/csv')

# def assign_fraud_labels(df, score_column='Detection_Score'):
#     df['Fraud_Level'] = pd.cut(df[score_column], bins=[-0.01, 0.5, 0.8, 1.0],
#                                labels=['Likely Normal', 'Mild Anomaly', 'Very Suspicious'])
#     return df

# # Upload
# st.sidebar.header("Upload your Dataset")
# uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     df = preprocess_data(df)
#     st.success("File Uploaded and Preprocessed!")

#     st.subheader("Dataset Preview")
#     st.dataframe(df, use_container_width=True)

#     label_col = find_label_column(df)
#     if label_col:
#         st.subheader("Initial Data Distribution")
#         pie_data = df[label_col].value_counts().reset_index()
#         pie_data.columns = ['Label', 'Count']
#         label_map = {0: 'Normal', 1: 'Fraud', -1: 'Anomaly'}
#         pie_data['Label'] = pie_data['Label'].map(label_map).fillna(pie_data['Label'].astype(str))

#         fig = px.pie(pie_data, names='Label', values='Count', title="Normal vs Fraud Data",
#                      color_discrete_sequence=px.colors.sequential.RdBu)
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("No fraud label column found. Pie chart not available.")

#     st.markdown("---")

#     st.subheader("Fraud Detection Section")
#     col1, col2, col3 = st.columns(3)

#     if col1.button("Detect Identity Theft"):
#         with st.spinner('Running Identity Theft Detection...'):
#             fraud_df = detect_identity_fraud(df)
#             fraud_df['Detection_Score'] = safe_predict(rf_model, fraud_df)
#             fraud_df = assign_fraud_labels(fraud_df)

#             st.warning(f"Identity Thefts Detected: {len(fraud_df)}")
#             st.dataframe(fraud_df)
#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("Download Identity Thefts", data=csv, file_name='identity_thefts.csv', mime='text/csv')
#             plot_type = st.radio("Select Feature Importance Plot Type:", ["Bar Plot", "Beeswarm Plot"], horizontal=True)
#             plot_shap_summary(rf_model, df.drop(columns=[label_col], errors="ignore"), plot_type)

#     if col2.button("Detect Synthetic Identities"):
#         with st.spinner('Running Synthetic Fraud Detection...'):
#             fraud_df = detect_synthetic_fraud(df)
#             fraud_df['Detection_Score'] = safe_predict(xgb_model, fraud_df)
#             fraud_df = assign_fraud_labels(fraud_df)

#             st.warning(f"Synthetic Identities Detected: {len(fraud_df)}")
#             st.dataframe(fraud_df)
#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("Download Synthetic Identities", data=csv, file_name='synthetic_identities.csv', mime='text/csv')
#             plot_type = st.radio("Select Feature Importance Plot Type:", ["Bar Plot", "Beeswarm Plot"], horizontal=True)
#             plot_shap_summary(xgb_model, df.drop(columns=[label_col], errors="ignore"), plot_type)

#     if col3.button("Detect Mule Accounts"):
#         with st.spinner('Running Mule Account Detection...'):
#             fraud_df = detect_mule_accounts(df)
#             fraud_df['Anomaly_Score'] = iso_forest.decision_function(
#                 fraud_df.drop(columns=["Fraud_Prediction", "Detection_Score"], errors="ignore")
#             )
#             fraud_df['Anomaly_Score'] = (fraud_df['Anomaly_Score'] - fraud_df['Anomaly_Score'].min()) / (
#                 fraud_df['Anomaly_Score'].max() - fraud_df['Anomaly_Score'].min()
#             )
#             fraud_df = assign_fraud_labels(fraud_df, score_column='Anomaly_Score')

#             st.warning(f"Mule Accounts Detected: {len(fraud_df)}")
#             st.dataframe(fraud_df)
#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("Download Mule Accounts", data=csv, file_name='mule_accounts.csv', mime='text/csv')
#             plot_isolation_forest_anomalies(fraud_df, score_col='Anomaly_Score')

# else:
#     st.info("Please upload a CSV file from the sidebar to begin!")
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import shap
# import joblib
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from test import detect_identity_fraud, detect_synthetic_fraud, detect_mule_accounts

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # Load models
# rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
# xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
# iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

# # Streamlit config
# st.set_page_config(page_title="Fraud Detection App ", layout="wide")
# st.title(" Fraud Detection Dashboard")

# # Preprocessing

# def preprocess_data(df):
#     df = df.dropna()
#     df = df.apply(pd.to_numeric, errors='coerce')
#     df = df.dropna()
#     if len(df) > 5000:
#         df = df.sample(5000, random_state=42)
#     return df

# # Label detection

# def find_label_column(df):
#     candidates = ['Class', 'isFraud', 'fraud', 'label', 'target']
#     for col in df.columns:
#         if col.lower() in [c.lower() for c in candidates]:
#             return col
#     return None

# # Fraud labels

# def assign_fraud_labels(df, score_column='Detection_Score'):
#     df['Fraud_Level'] = pd.cut(
#         df[score_column],
#         bins=[-0.01, 0.5, 0.8, 1.0],
#         labels=[' Likely Normal', ' Mild Anomaly', ' Very Suspicious']
#     )
#     return df

# # SHAP summary and top 10 bar plot

# def plot_shap_summary(model, X, plot_type):
#     model_features = model.feature_names_in_
#     X = X[[col for col in model_features if col in X.columns]]
#     explainer = shap.Explainer(model)
#     sample_X = X.sample(min(300, len(X)), random_state=42)
#     shap_values = explainer(sample_X)

#     st.subheader(" SHAP Summary Plot")
#     plt.figure()

#     try:
#         # Handle multi-class outputs
#         if hasattr(shap_values, 'values') and shap_values.values.ndim == 3:
#             # Pick fraud class (1)
#             shap_values.values = shap_values.values[:, :, 1]
#             shap_values.data = shap_values.data  # Optional: match structure
#             shap_values.base_values = shap_values.base_values[:, 1] if shap_values.base_values.ndim > 1 else shap_values.base_values

#         if plot_type == "Bar Plot":
#             shap.plots.bar(shap_values, max_display=10, show=False)
#         else:
#             shap.plots.beeswarm(shap_values, max_display=20, show=False)

#         st.pyplot(plt.gcf())
#         plt.clf()

#         # Show top 10 importance
#         importance = np.abs(shap_values.values).mean(axis=0)
#         top10 = pd.DataFrame({
#             'Feature': sample_X.columns,
#             'Importance': importance
#         }).sort_values('Importance', ascending=False).head(10)

#         st.subheader(" Top 10 Fraud-Contributing Features")
#         st.dataframe(top10)
#         fig, ax = plt.subplots(figsize=(8, 5))
#         sns.barplot(data=top10, x='Importance', y='Feature', palette='coolwarm', ax=ax)
#         ax.set_title("Top 10 SHAP Feature Importances")
#         st.pyplot(fig)
#         plt.clf()

#         csv = top10.to_csv(index=False).encode('utf-8')
#         st.download_button("Download Top 10 Feature Importances", data=csv, file_name="top_10_fraud_factors.csv")

#     except Exception as e:
#         st.error(f"Failed to render SHAP plot: {e}")

# # Safe predict_proba

# def safe_predict_proba(model, df):
#     features = model.feature_names_in_
#     df = df[[col for col in features if col in df.columns]]
#     return model.predict_proba(df)[:, 1]

# # Upload section
# st.header("Upload your Dataset ")
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     df = preprocess_data(df)
#     st.success(" File Uploaded and Preprocessed!")

#     st.subheader(" Dataset Preview")
#     st.dataframe(df, use_container_width=True)

#     label_col = find_label_column(df)
#     if label_col:
#         pie_data = df[label_col].value_counts().reset_index()
#         pie_data.columns = ['Label', 'Count']
#         pie_data['Label'] = pie_data['Label'].map({0: 'Normal', 1: 'Fraud', -1: 'Anomaly'}).fillna('Other')
#         fig = px.pie(pie_data, names='Label', values='Count', title="Normal vs Fraud Data ",
#                      color_discrete_sequence=px.colors.sequential.RdBu)
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("No label column found. Pie chart skipped.")

#     st.markdown("---")
#     st.subheader(" Fraud Detection Section")
#     col1, col2, col3 = st.sidebar.columns(3)

#     if col1.button(" Detect Identity Theft"):
#         with st.spinner("Running Identity Theft Detection..."):
#             fraud_df = detect_identity_fraud(df)
#             fraud_df['Detection_Score'] = safe_predict_proba(rf_model, fraud_df)
#             fraud_df = assign_fraud_labels(fraud_df)
#             st.session_state.fraud_df = fraud_df
#             st.session_state.fraud_model = rf_model
#             st.session_state.input_df = df.drop(columns=[label_col], errors="ignore")
#             st.success(f"Detected {len(fraud_df)} identity theft cases.")
#             st.dataframe(fraud_df)
#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("Download Identity Thefts", data=csv, file_name='identity_thefts.csv')

#     if col2.button(" Detect Synthetic Identities"):
#         with st.spinner("Running Synthetic Identity Detection..."):
#             fraud_df = detect_synthetic_fraud(df)
#             fraud_df['Detection_Score'] = safe_predict_proba(xgb_model, fraud_df)
#             fraud_df = assign_fraud_labels(fraud_df)
#             st.session_state.fraud_df = fraud_df
#             st.session_state.fraud_model = xgb_model
#             st.session_state.input_df = df.drop(columns=[label_col], errors="ignore")
#             st.success(f"Detected {len(fraud_df)} synthetic identities.")
#             st.dataframe(fraud_df)
#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("Download Synthetic Identities", data=csv, file_name='synthetic_identities.csv')

#     if col3.button(" Detect Mule Accounts"):
#         with st.spinner("Running Mule Account Detection..."):
#             fraud_df = detect_mule_accounts(df)
#             fraud_df['Detection_Score'] = (fraud_df['Detection_Score'] - fraud_df['Detection_Score'].min()) / \
#                                           (fraud_df['Detection_Score'].max() - fraud_df['Detection_Score'].min())
#             fraud_df = assign_fraud_labels(fraud_df)
#             st.session_state.fraud_df = fraud_df
#             st.session_state.fraud_model = None
#             st.success(f"Detected {len(fraud_df)} mule accounts.")
#             st.dataframe(fraud_df)
#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("Download Mule Accounts", data=csv, file_name='mule_accounts.csv')

#     if "fraud_df" in st.session_state and not st.session_state.fraud_df.empty and "fraud_model" in st.session_state and st.session_state.fraud_model:
#         st.markdown("---")
#         st.subheader(" SHAP for Individual Fraud Case")

#         selected_index = st.selectbox("Select a row to explain", st.session_state.fraud_df.index.tolist(), key="row_selector")

#         if selected_index is not None:
#             row = st.session_state.fraud_df.loc[selected_index]
#             model = st.session_state.fraud_model
#             input_data = row[model.feature_names_in_].values.reshape(1, -1)

#             # Create SHAP explainer and explain single instance
#             explainer = shap.Explainer(model)
#             shap_values = explainer(input_data)

#             st.write("### Selected Row")
#             st.dataframe(row.to_frame().T, use_container_width=True)

#             st.write("### SHAP Force Plot for Selected Row")
#             shap.initjs()
#             st_shap = st.components.v1.html(shap.plots.force(shap_values[0], matplotlib=False), height=300)

#             st.write("### SHAP Waterfall Plot")
#             fig = shap.plots.waterfall(shap_values[0], max_display=10, show=False)
#             st.pyplot(fig.figure)

# else:
#     st.info("Please upload a CSV file to begin.")
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import shap
# import joblib
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from test import detect_identity_fraud, detect_synthetic_fraud, detect_mule_accounts

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# DATA_PATH = os.path.join(BASE_DIR, "data")

# # Load models
# rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
# xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
# iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

# # Streamlit config
# st.set_page_config(page_title="Fraud Detection App ", layout="wide")
# st.title(" Fraud Detection Dashboard")

# # Preprocessing
# def preprocess_data(df):
#     df = df.dropna()
#     df = df.apply(pd.to_numeric, errors='coerce')
#     df = df.dropna()
#     if len(df) > 5000:
#         df = df.sample(5000, random_state=42)
#     return df

# # Label detection
# def find_label_column(df):
#     candidates = ['Class', 'isFraud', 'fraud', 'label', 'target']
#     for col in df.columns:
#         if col.lower() in [c.lower() for c in candidates]:
#             return col
#     return None

# # Fraud labels
# def assign_fraud_labels(df, score_column='Detection_Score'):
#     df['Fraud_Level'] = pd.cut(
#         df[score_column],
#         bins=[-0.01, 0.5, 0.8, 1.0],
#         labels=[' Likely Normal', ' Mild Anomaly', ' Very Suspicious']
#     )
#     return df

# # SHAP summary and top 10 bar plot
# def plot_shap_summary(model, X, plot_type):
#     model_features = model.feature_names_in_
#     X = X[[col for col in model_features if col in X.columns]]
#     explainer = shap.Explainer(model)
#     sample_X = X.sample(min(300, len(X)), random_state=42)
#     shap_values = explainer(sample_X)

#     st.subheader(" SHAP Summary Plot")
#     plt.figure()

#     try:
#         if hasattr(shap_values, 'values') and shap_values.values.ndim == 3:
#             shap_values.values = shap_values.values[:, :, 1]
#             shap_values.base_values = shap_values.base_values[:, 1] if shap_values.base_values.ndim > 1 else shap_values.base_values

#         if plot_type == "Bar Plot":
#             shap.plots.bar(shap_values, max_display=10, show=False)
#         else:
#             shap.plots.beeswarm(shap_values, max_display=20, show=False)

#         st.pyplot(plt.gcf())
#         plt.clf()

#         importance = np.abs(shap_values.values).mean(axis=0)
#         top10 = pd.DataFrame({
#             'Feature': sample_X.columns,
#             'Importance': importance
#         }).sort_values('Importance', ascending=False).head(10)

#         st.subheader(" Top 10 Fraud-Contributing Features")
#         st.dataframe(top10)
#         fig, ax = plt.subplots(figsize=(8, 5))
#         sns.barplot(data=top10, x='Importance', y='Feature', palette='coolwarm', ax=ax)
#         ax.set_title("Top 10 SHAP Feature Importances")
#         st.pyplot(fig)
#         plt.clf()

#         csv = top10.to_csv(index=False).encode('utf-8')
#         st.download_button("Download Top 10 Feature Importances", data=csv, file_name="top_10_fraud_factors.csv")

#     except Exception as e:
#         st.error(f"Failed to render SHAP plot: {e}")

# # Safe predict_proba
# def safe_predict_proba(model, df):
#     features = model.feature_names_in_
#     df = df[[col for col in features if col in df.columns]]
#     return model.predict_proba(df)[:, 1]

# # File selection
# st.sidebar.header("Data Source")
# data_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
# selected_file = st.sidebar.selectbox("Select dataset to load", data_files)

# if selected_file:
#     file_path = os.path.join(DATA_PATH, selected_file)
#     try:
#         df = pd.read_csv(file_path)
#         df = preprocess_data(df)
#         st.success(f" Loaded dataset: {selected_file}")
#     except Exception as e:
#         st.error(f"Failed to load dataset: {e}")
#         st.stop()

#     st.subheader(" Dataset Preview")
#     st.dataframe(df, use_container_width=True)

#     label_col = find_label_column(df)
#     if label_col:
#         pie_data = df[label_col].value_counts().reset_index()
#         pie_data.columns = ['Label', 'Count']
#         pie_data['Label'] = pie_data['Label'].map({0: 'Normal', 1: 'Fraud', -1: 'Anomaly'}).fillna('Other')
#         fig = px.pie(pie_data, names='Label', values='Count', title="Normal vs Fraud Data ",
#                      color_discrete_sequence=px.colors.sequential.RdBu)
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("No label column found. Pie chart skipped.")

#     st.markdown("---")
#     st.subheader(" Fraud Detection Section")
#     col1, col2, col3 = st.columns(3)

#     if col1.button(" Detect Identity Theft"):
#         with st.spinner("Running Identity Theft Detection..."):
#             fraud_df = detect_identity_fraud(df)
#             fraud_df['Detection_Score'] = safe_predict_proba(rf_model, fraud_df)
#             fraud_df = assign_fraud_labels(fraud_df)
#             st.session_state.fraud_df = fraud_df
#             st.session_state.fraud_model = rf_model
#             st.session_state.input_df = df.drop(columns=[label_col], errors="ignore")
#             st.success(f"Detected {len(fraud_df)} identity theft cases.")
#             st.dataframe(fraud_df)
#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("Download Identity Thefts", data=csv, file_name='identity_thefts.csv')

#     if col2.button(" Detect Synthetic Identities"):
#         with st.spinner("Running Synthetic Identity Detection..."):
#             fraud_df = detect_synthetic_fraud(df)
#             fraud_df['Detection_Score'] = safe_predict_proba(xgb_model, fraud_df)
#             fraud_df = assign_fraud_labels(fraud_df)
#             st.session_state.fraud_df = fraud_df
#             st.session_state.fraud_model = xgb_model
#             st.session_state.input_df = df.drop(columns=[label_col], errors="ignore")
#             st.success(f"Detected {len(fraud_df)} synthetic identities.")
#             st.dataframe(fraud_df)
#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("Download Synthetic Identities", data=csv, file_name='synthetic_identities.csv')

#     if col3.button(" Detect Mule Accounts"):
#         with st.spinner("Running Mule Account Detection..."):
#             fraud_df = detect_mule_accounts(df)
#             fraud_df['Detection_Score'] = (fraud_df['Detection_Score'] - fraud_df['Detection_Score'].min()) / \
#                                           (fraud_df['Detection_Score'].max() - fraud_df['Detection_Score'].min())
#             fraud_df = assign_fraud_labels(fraud_df)
#             st.session_state.fraud_df = fraud_df
#             st.session_state.fraud_model = None
#             st.success(f"Detected {len(fraud_df)} mule accounts.")
#             st.dataframe(fraud_df)
#             csv = fraud_df.to_csv(index=False).encode('utf-8')
#             st.download_button("Download Mule Accounts", data=csv, file_name='mule_accounts.csv')

#     if "fraud_df" in st.session_state and "fraud_model" in st.session_state and st.session_state.fraud_model:
#         st.markdown("---")
#         st.subheader(" SHAP Feature Explanation")
#         plot_type = st.radio("Choose SHAP Plot Type", ["Bar Plot", "Beeswarm Plot"], horizontal=True, key="shap_radio")
#         plot_shap_summary(st.session_state.fraud_model, st.session_state.input_df, plot_type)
# else:
#     st.info("Please select a CSV file from the sidebar.")
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import shap
# import joblib
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from test import detect_identity_fraud, detect_synthetic_fraud, detect_account_takeover, detect_user_device_fraud
# from catboost import CatBoostClassifier
# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# DATA_PATH = os.path.join(BASE_DIR, "processed_data")

# # Load models
# rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
# xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
# iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
# def get_model_features(model):
#     """Returns feature names from sklearn or CatBoost models."""
#     if hasattr(model, "feature_names_in_"):
#         return list(model.feature_names_in_)
#     elif hasattr(model, "feature_names_"):
#         return list(model.feature_names_)
#     else:
#         raise AttributeError("Model does not have accessible feature names.")

# st.set_page_config(page_title="Fraud Detection App", layout="wide")
# st.title("Fraud Detection Dashboard")

# def preprocess_data(df):
#     df = df.dropna()
#     df = df.apply(pd.to_numeric, errors='coerce')
#     df = df.dropna()
#     if len(df) > 5000:
#         df = df.sample(5000, random_state=42)
#     return df


# def find_label_column(df):
#     for col in df.columns:
#         if col.lower() in ['class', 'isfraud', 'fraud', 'label', 'target']:
#             return col
#     return None

# def assign_fraud_labels(df, score_column='Detection_Score'):
#     df['Fraud_Level'] = pd.cut(
#         df[score_column],
#         bins=[-0.01, 0.5, 0.8, 1.0],
#         labels=['Likely Normal', 'Mild Anomaly', 'Very Suspicious']
#     )
#     return df

# def safe_predict_proba(model, df):
#     features = get_model_features(model)
#     df = df[[col for col in features if col in df.columns]]
#     return model.predict_proba(df)[:, 1]


# def plot_shap_summary(model, X, plot_type):
#     try:
#         features = [col for col in get_model_features(model) if col in X.columns]
#         X = X[features]
#         explainer = shap.TreeExplainer(model)
#         sample_X = X.sample(min(300, len(X)), random_state=42)
#         shap_values = explainer.shap_values(sample_X)
#         # Handle multi-class models
#         if isinstance(shap_values, list):
#             # Pick class 1 (fraud) if available
#             shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
#         st.subheader(" SHAP Summary Plot")
#         plt.figure()
#         if plot_type == "Bar Plot":
#             shap.summary_plot(shap_values, sample_X, plot_type="bar", max_display=10, show=False)
#         else:
#             shap.summary_plot(shap_values, sample_X, plot_type="violin", max_display=10, show=False)
#         st.pyplot(plt.gcf())
#         plt.clf()
#         # Top 10 features as a bar plot
#         vals = np.abs(shap_values).mean(0)
#         top_features = pd.DataFrame({
#             'Feature': sample_X.columns,
#             'Importance': vals
#         }).sort_values('Importance', ascending=False).head(10)
#         st.subheader("Top 10 SHAP Feature Importances")
#         st.dataframe(top_features)
#         fig, ax = plt.subplots(figsize=(8, 5))
#         sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis', ax=ax)
#         ax.set_title("Top 10 SHAP Feature Importances")
#         st.pyplot(fig)
#         plt.clf()
#         csv = top_features.to_csv(index=False).encode('utf-8')
#         st.download_button("Download Top 10 Features", data=csv, file_name="top_10_fraud_factors.csv")
#     except Exception as e:
#         st.error(f"SHAP summary failed: {e}")

# # Sidebar file selection
# st.sidebar.header(" Data Source")
# files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
# selected_file = st.sidebar.selectbox("Select a dataset", files)

# if selected_file:
#     df = pd.read_csv(os.path.join(DATA_PATH, selected_file))
#     df = preprocess_data(df)
#     st.success(f" Loaded dataset: {selected_file}")
#     st.dataframe(df.head(100), use_container_width=True)
#     label_col = find_label_column(df)
#     if label_col:
#         pie_data = df[label_col].value_counts().reset_index()
#         pie_data.columns = ['Label', 'Count']
#         pie_data['Label'] = pie_data['Label'].map({0: 'Normal', 1: 'Fraud', -1: 'Anomaly'}).fillna('Other')
#         fig = px.pie(pie_data, names='Label', values='Count', title="Fraud vs Normal")
#         st.plotly_chart(fig, use_container_width=True)
#     st.subheader(" Run Fraud Detection")
#     col1, col2, col3 = st.sidebar.columns(3)
#     if col1.button("Detect Identity Theft"):
#         with st.spinner("Detecting Identity Theft..."):
#             fraud_df = detect_identity_fraud(df)
#             fraud_df['Detection_Score'] = safe_predict_proba(rf_model, fraud_df)
#             fraud_df = assign_fraud_labels(fraud_df)
#             st.session_state.fraud_df = fraud_df
#             st.session_state.fraud_model = rf_model
#             st.session_state.input_df = df[get_model_features(rf_model)]
#             st.success(f" {len(fraud_df)} identity thefts found")
#             st.dataframe(fraud_df.head(100), use_container_width=True)
#     if col2.button("Detect Synthetic Identities"):
#         with st.spinner("Detecting Synthetic Identities..."):
#             fraud_df = detect_synthetic_fraud(df)
#             fraud_df['Detection_Score'] = safe_predict_proba(xgb_model, fraud_df)
#             fraud_df = assign_fraud_labels(fraud_df)
#             st.session_state.fraud_df = fraud_df
#             st.session_state.fraud_model = xgb_model
#             st.session_state.input_df = df[get_model_features(xgb_model)]
#             st.success(f" {len(fraud_df)} synthetic identities found")
#             st.dataframe(fraud_df.head(100), use_container_width=True)
    
    ##############################################################
    # if col3.button("Detect Mule Accounts"):
    #     with st.spinner("Detecting Mule Accounts..."):
    #         fraud_df = detect_mule_accounts(df)
    #         features = iso_forest.feature_names_in_
    #         # Only use the features the model was trained on
    #         fraud_df['Anomaly_Score'] = iso_forest.decision_function(fraud_df[features])
    #         fraud_df['Anomaly_Score'] = (fraud_df['Anomaly_Score'] - fraud_df['Anomaly_Score'].min()) / \
    #                                 (fraud_df['Anomaly_Score'].max() - fraud_df['Anomaly_Score'].min())
    #         fraud_df = assign_fraud_labels(fraud_df, 'Anomaly_Score')
    #         st.session_state.fraud_df = fraud_df
    #         st.session_state.fraud_model = None
    #         st.success(f" {len(fraud_df)} mule accounts found")
    #         st.dataframe(fraud_df.head(100), use_container_width=True)
    #         # Add right after the mule detection code where you show fraud_df
    #         if len(fraud_df) > 0:
    #             st.markdown("---")
    #             st.subheader(" Why Accounts Are Flagged as Mules")
                
    #             explanation_map = {
    #                 'Number of linked accounts': 'Linked to many other accounts, indicating possible mule network',
    #                 'Transaction frequency': 'High frequency of transactions unusual for normal users',
    #                 'Average transaction amount': 'Average transaction amount is inconsistent with typical behavior',
    #                 'Device count': 'Multiple devices used to access account, suspicious activity',
    #                 'Geographic location variance': 'Accessed from many different locations, unusual for one user',
    #                 'Account age': 'Account is relatively new, which is suspicious',
    #                 'Unusual login times': 'Logins at odd hours, not typical for normal users',
    #                 'Circular transaction patterns': 'Money moves in circular patterns, a sign of laundering',
    #                 'Rapid money movement': 'Money moves rapidly through the account, suspicious behavior'
    #             }
                
    #             selected_index = st.selectbox("Select account to explain:", fraud_df.index)
    #             st.write("### Why this account is flagged as a mule:")
                
    #             features = [col for col in fraud_df.columns if col in explanation_map]
    #             importance_data = []
                
    #             for f in features:
    #                 try:
    #                     val = fraud_df.loc[selected_index, f]
    #                     imp = abs(float(val)) / 10  # Convert to float and normalize
    #                     importance_data.append({
    #                         "Feature": f,
    #                         "Importance": imp,
    #                         "Explanation": explanation_map[f]
    #                     })
    #                 except (KeyError, ValueError):
    #                     continue
            
    #             if importance_data:
    #                 importance_df = pd.DataFrame(importance_data)
    #                 importance_df = importance_df.sort_values('Importance', ascending=False).head(9)
                    
    #                 # Plotting section (MUST BE INSIDE THIS IF BLOCK)
    #                 fig, ax = plt.subplots(figsize=(10, 6))
    #                 sns.barplot(x='Importance', y='Feature', data=importance_df, palette='coolwarm', ax=ax)
    #                 plt.title('Top Reasons Why This Account is Flagged as a Mule')
                    
    #                 # Add text annotations
    #                 for index, row in importance_df.iterrows():
    #                     plt.text(row['Importance'] + 0.01, index, row['Explanation'], va='center', fontsize=9)
                    
    #                 plt.tight_layout()
    #                 st.pyplot(fig)
    #                 plt.clf()
                    
    #                 # Show dataframe
    #                 st.dataframe(importance_df)
    #             else:
    #                 st.warning("No explanatory features available for this account")

    #         # Plot the explanations
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         sns.barplot(x='Importance', y='Feature', data=importance_df, palette='coolwarm')
    #         plt.title('Top Reasons Why This Account is Flagged as a Mule')
            
    #         # Add explanations as text
    #         for index, row in importance_df.iterrows():
    #             plt.text(row['Importance'] + 0.01, importance_df.index.get_indexer([index])[0], 
    #                     row['Explanation'], va='center', fontsize=9)
            
    #         plt.tight_layout()
    #         st.pyplot(fig)
    #         plt.clf()

################################################################################################## 
    # if col3.button("Detect Account Takeover"):
    #     with st.spinner("Detecting account takeovers..."):
    #         ato_model = joblib.load("models/xgboost_ato.pkl")
    #         df_fraud = detect_account_takeover(df)
    #         st.session_state.fraud_df = df_fraud
    #         st.session_state.fraud_model = ato_model
    #         st.session_state.input_df = df[ato_model.feature_names_in_]
    #         st.success(f"Detected {len(df_fraud)} potential ATO frauds")
    #         st.dataframe(df_fraud.head(100))
    # col4 = st.sidebar.button("Profile User-Device Behavior")
    # if col4:
    #     with st.spinner("Profiling user-device behavior..."):
    #         udf_fraud_df = detect_user_device_fraud(df.copy())
    #         udf_fraud_df = assign_fraud_labels(udf_fraud_df, score_column="Fraud_Score")
    #         st.session_state.fraud_df = udf_fraud_df
    #         model = CatBoostClassifier()
    #         model.load_model(os.path.join(MODEL_DIR, "catboost_user_device_model.cbm"))
    #         st.session_state.fraud_model = model
    #         cat_model = st.session_state.fraud_model
    #         if hasattr(cat_model, "feature_names_"):
    #             st.session_state.input_df = udf_fraud_df[cat_model.feature_names_]
    #         else:
    #             st.warning("CatBoost model does not contain feature names. Falling back to full dataframe.")
    #             st.session_state.input_df = udf_fraud_df
    #         st.success(f" {len(udf_fraud_df)} suspicious user-device cases detected")
    #         st.dataframe(udf_fraud_df.head(100), use_container_width=True)
    #                     # Export flagged cases
    #         csv_export = udf_fraud_df.to_csv(index=False).encode("utf-8")
    #         st.download_button(" Download Flagged Cases CSV", csv_export, file_name="flagged_user_device_cases.csv")

    #         json_export = udf_fraud_df.to_json(orient="records", indent=4)
    #         st.download_button(" Download Flagged Profiles JSON", json_export, file_name="user_device_fraud_profiles.json")
    #         with st.expander(" Preview Fraud Profile JSON"):
    #             st.json(json_export)

# SHAP Explanation Section
# if "fraud_df" in st.session_state and "fraud_model" in st.session_state and st.session_state.fraud_model:
#     st.markdown("---")
#     st.subheader(" SHAP for Individual Fraud Case")
#     selected_index = st.selectbox("Select a row to explain", st.session_state.fraud_df.index.tolist(), key="row_selector")
#     if selected_index is not None:
#         row = st.session_state.fraud_df.loc[selected_index]
#         model = st.session_state.fraud_model
#         input_data = pd.DataFrame([row[model.feature_names_in_].values], columns=model.feature_names_in_)
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(input_data)
#         # Handle multi-class
#         if isinstance(shap_values, list):
#             shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
#         st.write("### Selected Row")
#         st.dataframe(row.to_frame().T, use_container_width=True)
#         st.write("### SHAP Force Plot for Selected Row")
#         try:
#             shap.initjs()
#             # Use matplotlib backend for single row
#             shap.force_plot(
#                 explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value,
#                 shap_values[0],
#                 input_data.iloc[0],
#                 matplotlib=True,
#                 show=False
#             )
#             st.pyplot(plt.gcf())
#             plt.clf()
#         except Exception as e:
#             st.warning(f"Force plot failed: {e}")
#         st.write("### SHAP Waterfall Plot")
#         try:
#             fig = shap.plots.waterfall(shap.Explanation(
#                 values=shap_values[0],
#                 base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value,
#                 data=input_data.iloc[0].values,
#                 feature_names=model.feature_names_in_
#             ), max_display=10, show=False)
#             st.pyplot(fig)
#             plt.clf()
#         except Exception as e:
#             st.warning(f"Waterfall plot failed: {e}")
#     st.subheader(" Global SHAP Importance")
#     plot_type = st.radio("Choose SHAP Plot Type", ["Bar Plot", "Beeswarm Plot"], horizontal=True)
#     plot_shap_summary(st.session_state.fraud_model, st.session_state.input_df, plot_type)
# else:
#     st.info("Please add your dataset into the 'processed_data' folder and select it.")
# SHAP Explanation Section
# if "fraud_df" in st.session_state and "fraud_model" in st.session_state and st.session_state.fraud_model:
#     st.markdown("---")
#     st.subheader(" SHAP for Individual Fraud Case")
#     selected_index = st.selectbox("Select a row to explain", st.session_state.fraud_df.index.tolist(), key="row_selector")
#     if selected_index is not None:
#         row = st.session_state.fraud_df.loc[selected_index]
#         model = st.session_state.fraud_model
#         # Use model.feature_names_ for CatBoost (if available), fallback to full row
#         if hasattr(model, "feature_names_"):
#             feature_names = model.feature_names_
#             input_data = pd.DataFrame([row[feature_names].values], columns=feature_names)
#         else:
#             feature_names = row.index.tolist()
#             input_data = pd.DataFrame([row.values], columns=feature_names)
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(input_data)
#         # Handle multi-class
#         if isinstance(shap_values, list):
#             shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
#         st.write("### Selected Row")
#         st.dataframe(row.to_frame().T, use_container_width=True)
        
#         # Show top contributing features for this row
#         st.write("### Top Contributing Features for This Row")
#         contrib = pd.DataFrame({
#             'Feature': get_model_features(model),
#             'SHAP Value': shap_values[0]
#         }).sort_values('SHAP Value', key=np.abs, ascending=False)
#         st.dataframe(contrib.head(10))
#         fig, ax = plt.subplots(figsize=(8, 5))
#         sns.barplot(data=contrib.head(10), x='SHAP Value', y='Feature', palette='viridis', ax=ax)
#         ax.set_title("Top Contributing Features for This Row")
#         st.pyplot(fig)
#         plt.clf()

#         st.write("### SHAP Force Plot for Selected Row")
#         try:
#             shap.initjs()
#             shap.force_plot(
#                 explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value,
#                 shap_values[0],
#                 input_data.iloc[0],
#                 matplotlib=True,
#                 show=False
#             )
#             st.pyplot(plt.gcf())
#             plt.clf()
#         except Exception as e:
#             st.warning(f"Force plot failed: {e}")

#         st.write("### SHAP Waterfall Plot")
#         try:
#             plt.figure()
#             shap.plots._waterfall.waterfall_legacy(
#                 explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value,
#                 shap_values[0],
#                 feature_names=get_model_features(model),
#                 max_display=10
#             )
#             st.pyplot(plt.gcf())
#             plt.clf()
#         except Exception as e:
#             st.warning(f"Waterfall plot failed: {e}")
#     st.subheader(" Global SHAP Importance")
#     plot_type = st.radio("Choose SHAP Plot Type", ["Bar Plot", "Beeswarm Plot"], horizontal=True)
#     plot_shap_summary(st.session_state.fraud_model, st.session_state.input_df, plot_type)
# else:
#     st.info("Please add your dataset into the 'processed_data' folder and select it.")

from db import supabase 
import streamlit as st


from datetime import datetime, timezone

from db import reset_password,get_user_by_employee_id
from auth import signup, login, verify_otp, send_reset_otp
import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from test import detect_identity_fraud, detect_synthetic_fraud, detect_account_takeover, detect_user_device_fraud
from catboost import CatBoostClassifier


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "reset_mode" not in st.session_state:
    st.session_state.reset_mode = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""


st.set_page_config(page_title=" Insurance Fraud Detection", layout="wide")

# -- AUTHENTICATION UI ------------------------
if not st.session_state.authenticated:
    st.title(" Login / Signup")

    tab = st.sidebar.radio("Select", ["Login", "Signup", "Verify OTP", "Forgot Password"])

    if tab == "Signup":
        emp_id = st.text_input("Employee ID")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Signup"):
            if signup(emp_id, password, email):
                st.success(" Signup successful! Check your email for OTP.")
            else:
                st.error("Signup failed")

    elif tab == "Verify OTP":
        emp_id = st.text_input("Employee ID")
        otp = st.text_input("Enter OTP")
        if st.button("Verify"):
            if verify_otp(emp_id, otp):
                st.session_state.authenticated = True
                st.session_state.user_id = emp_id
                st.success(" OTP verified! Welcome.")
                st.rerun()
            else:
                st.error(" Invalid or expired OTP.")

    elif tab == "Login":
        emp_id = st.text_input("Employee ID")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = get_user_by_employee_id(emp_id)
            if user and user.get("login_attempts", 0) >= 3:
                st.error(" Account locked due to 3 failed attempts.")
            elif login(emp_id, password):
                st.session_state.authenticated = True
                st.session_state.user_id = emp_id
                st.success(" Login successful.")
                st.rerun()
            else:
                st.error(" Invalid credentials.")


    elif tab == "Forgot Password":
        emp_id = st.text_input("Employee ID")
        if st.button("Send Reset OTP"):
            if send_reset_otp(emp_id):
                st.success("📧 Reset OTP sent to your registered email.")
                st.session_state.reset_mode = True
                st.session_state.emp_id_for_reset = emp_id
            else:
                st.error(" Could not send OTP. Please check Employee ID.")

        if st.session_state.reset_mode:
            otp = st.text_input("Enter OTP")
            new_password = st.text_input("New Password", type="password")
            if st.button("Reset Password"):
                from auth import reset_password
                if reset_password(st.session_state.emp_id_for_reset, otp, new_password):
                    st.success(" Password reset successful!")
                    st.session_state.reset_mode = False
                else:
                    st.error(" Invalid OTP or failed to reset.")
    st.stop()
  



# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "processed_data")

# Load models
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
def get_model_features(model):
    """Returns feature names from sklearn or CatBoost models."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    elif hasattr(model, "feature_names_"):
        return list(model.feature_names_)
    else:
        raise AttributeError("Model does not have accessible feature names.")


st.title("Fraud Detection Dashboard")
with st.sidebar:
    st.markdown("### 👤 Account")
    st.markdown(f"Welcome, **{st.session_state.user_id}**")
    user_info = get_user_by_employee_id(st.session_state.user_id)
    if user_info:
        st.markdown(f" Logged in as: `{user_info['email']}`")


    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.user_id = ""
        st.rerun()

def preprocess_data(df):
    df = df.dropna()
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)
    return df


def find_label_column(df):
    for col in df.columns:
        if col.lower() in ['class', 'isfraud', 'fraud', 'label', 'target']:
            return col
    return None

def assign_fraud_labels(df, score_column='Detection_Score'):
    df['Fraud_Level'] = pd.cut(
        df[score_column],
        bins=[-0.01, 0.5, 0.8, 1.0],
        labels=['Likely Normal', 'Mild Anomaly', 'Very Suspicious']
    )
    return df

def safe_predict_proba(model, df):
    features = get_model_features(model)
    df = df[[col for col in features if col in df.columns]]
    return model.predict_proba(df)[:, 1]


def plot_shap_summary(model, X, plot_type):
    try:
        features = [col for col in get_model_features(model) if col in X.columns]
        X = X[features]
        explainer = shap.TreeExplainer(model)
        sample_X = X.sample(min(300, len(X)), random_state=42)
        shap_values = explainer.shap_values(sample_X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        st.subheader(" SHAP Summary Plot")
        plt.figure()
        if plot_type == "Bar Plot":
            shap.summary_plot(shap_values, sample_X, plot_type="bar", max_display=10, show=False)
        else:
            shap.summary_plot(shap_values, sample_X, plot_type="violin", max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
        vals = np.abs(shap_values).mean(0)
        top_features = pd.DataFrame({
            'Feature': sample_X.columns,
            'Importance': vals
        }).sort_values('Importance', ascending=False).head(10)
        st.subheader("Top 10 SHAP Feature Importances")
        st.dataframe(top_features)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis', ax=ax)
        ax.set_title("Top 10 SHAP Feature Importances")
        st.pyplot(fig)
        plt.clf()
        csv = top_features.to_csv(index=False).encode('utf-8')
        st.download_button("Download Top 10 Features", data=csv, file_name="top_10_fraud_factors.csv")
    except Exception as e:
        st.error(f"SHAP summary failed: {e}")

st.sidebar.header(" Data Source")
files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
selected_file = st.sidebar.selectbox("Select a dataset", files)

if selected_file:
    df = pd.read_csv(os.path.join(DATA_PATH, selected_file))
    df = preprocess_data(df)
    st.success(f" Loaded dataset: {selected_file}")
    st.dataframe(df.head(100), use_container_width=True)
    label_col = find_label_column(df)
    if label_col:
        pie_data = df[label_col].value_counts().reset_index()
        pie_data.columns = ['Label', 'Count']
        pie_data['Label'] = pie_data['Label'].map({0: 'Normal', 1: 'Fraud', -1: 'Anomaly'}).fillna('Other')
        fig = px.pie(pie_data, names='Label', values='Count', title="Fraud vs Normal")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader(" Run Fraud Detection")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    if col1.button("Detect Identity Theft",use_container_width=True):
        with st.spinner("Detecting Identity Theft..."):
            fraud_df = detect_identity_fraud(df)
            fraud_df['Detection_Score'] = safe_predict_proba(rf_model, fraud_df)
            fraud_df = assign_fraud_labels(fraud_df)
            st.session_state.fraud_df = fraud_df
            st.session_state.fraud_model = rf_model
            st.session_state.input_df = df[get_model_features(rf_model)]
            st.success(f" {len(fraud_df)} identity thefts found")
            st.dataframe(fraud_df.head(100), use_container_width=True)
    if col2.button("Detect Synthetic Identities",use_container_width=True):
        with st.spinner("Detecting Synthetic Identities..."):
            fraud_df = detect_synthetic_fraud(df)
            fraud_df['Detection_Score'] = safe_predict_proba(xgb_model, fraud_df)
            fraud_df = assign_fraud_labels(fraud_df)
            st.session_state.fraud_df = fraud_df
            st.session_state.fraud_model = xgb_model
            st.session_state.input_df = df[get_model_features(xgb_model)]
            st.success(f" {len(fraud_df)} synthetic identities found")
            st.dataframe(fraud_df.head(100), use_container_width=True)
    
    ##############################################################
    # if col3.button("Detect Mule Accounts"):
    #     with st.spinner("Detecting Mule Accounts..."):
    #         fraud_df = detect_mule_accounts(df)
    #         features = iso_forest.feature_names_in_
    #         # Only use the features the model was trained on
    #         fraud_df['Anomaly_Score'] = iso_forest.decision_function(fraud_df[features])
    #         fraud_df['Anomaly_Score'] = (fraud_df['Anomaly_Score'] - fraud_df['Anomaly_Score'].min()) / \
    #                                 (fraud_df['Anomaly_Score'].max() - fraud_df['Anomaly_Score'].min())
    #         fraud_df = assign_fraud_labels(fraud_df, 'Anomaly_Score')
    #         st.session_state.fraud_df = fraud_df
    #         st.session_state.fraud_model = None
    #         st.success(f" {len(fraud_df)} mule accounts found")
    #         st.dataframe(fraud_df.head(100), use_container_width=True)
    #         # Add right after the mule detection code where you show fraud_df
    #         if len(fraud_df) > 0:
    #             st.markdown("---")
    #             st.subheader(" Why Accounts Are Flagged as Mules")
                
    #             explanation_map = {
    #                 'Number of linked accounts': 'Linked to many other accounts, indicating possible mule network',
    #                 'Transaction frequency': 'High frequency of transactions unusual for normal users',
    #                 'Average transaction amount': 'Average transaction amount is inconsistent with typical behavior',
    #                 'Device count': 'Multiple devices used to access account, suspicious activity',
    #                 'Geographic location variance': 'Accessed from many different locations, unusual for one user',
    #                 'Account age': 'Account is relatively new, which is suspicious',
    #                 'Unusual login times': 'Logins at odd hours, not typical for normal users',
    #                 'Circular transaction patterns': 'Money moves in circular patterns, a sign of laundering',
    #                 'Rapid money movement': 'Money moves rapidly through the account, suspicious behavior'
    #             }
                
    #             selected_index = st.selectbox("Select account to explain:", fraud_df.index)
    #             st.write("### Why this account is flagged as a mule:")
                
    #             features = [col for col in fraud_df.columns if col in explanation_map]
    #             importance_data = []
                
    #             for f in features:
    #                 try:
    #                     val = fraud_df.loc[selected_index, f]
    #                     imp = abs(float(val)) / 10  # Convert to float and normalize
    #                     importance_data.append({
    #                         "Feature": f,
    #                         "Importance": imp,
    #                         "Explanation": explanation_map[f]
    #                     })
    #                 except (KeyError, ValueError):
    #                     continue
            
    #             if importance_data:
    #                 importance_df = pd.DataFrame(importance_data)
    #                 importance_df = importance_df.sort_values('Importance', ascending=False).head(9)
                    
    #                 # Plotting section (MUST BE INSIDE THIS IF BLOCK)
    #                 fig, ax = plt.subplots(figsize=(10, 6))
    #                 sns.barplot(x='Importance', y='Feature', data=importance_df, palette='coolwarm', ax=ax)
    #                 plt.title('Top Reasons Why This Account is Flagged as a Mule')
                    
    #                 # Add text annotations
    #                 for index, row in importance_df.iterrows():
    #                     plt.text(row['Importance'] + 0.01, index, row['Explanation'], va='center', fontsize=9)
                    
    #                 plt.tight_layout()
    #                 st.pyplot(fig)
    #                 plt.clf()
                    
    #                 # Show dataframe
    #                 st.dataframe(importance_df)
    #             else:
    #                 st.warning("No explanatory features available for this account")

    #         # Plot the explanations
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         sns.barplot(x='Importance', y='Feature', data=importance_df, palette='coolwarm')
    #         plt.title('Top Reasons Why This Account is Flagged as a Mule')
            
    #         # Add explanations as text
    #         for index, row in importance_df.iterrows():
    #             plt.text(row['Importance'] + 0.01, importance_df.index.get_indexer([index])[0], 
    #                     row['Explanation'], va='center', fontsize=9)
            
    #         plt.tight_layout()
    #         st.pyplot(fig)
    #         plt.clf()

################################################################################################## 
    if col3.button("Detect Account Takeover",use_container_width=True):
        with st.spinner("Detecting account takeovers..."):
            ato_model = joblib.load("models/xgboost_ato.pkl")
            df_fraud = detect_account_takeover(df)
            st.session_state.fraud_df = df_fraud
            st.session_state.fraud_model = ato_model
            st.session_state.input_df = df[ato_model.feature_names_in_]
            st.success(f"Detected {len(df_fraud)} potential ATO frauds")
            st.dataframe(df_fraud.head(100))

    if col4.button("Profile User-Device Behavior",use_container_width=True):
        with st.spinner("Profiling user-device behavior..."):
            udf_fraud_df = detect_user_device_fraud(df.copy())
            udf_fraud_df = assign_fraud_labels(udf_fraud_df, score_column="Fraud_Score")
            st.session_state.fraud_df = udf_fraud_df
            model = CatBoostClassifier()
            model.load_model(os.path.join(MODEL_DIR, "catboost_user_device_model.cbm"))
            st.session_state.fraud_model = model
            cat_model = st.session_state.fraud_model
            if hasattr(cat_model, "feature_names_"):
                st.session_state.input_df = udf_fraud_df[cat_model.feature_names_]
            else:
                st.warning("CatBoost model does not contain feature names. Falling back to full dataframe.")
                st.session_state.input_df = udf_fraud_df
            st.success(f" {len(udf_fraud_df)} suspicious user-device cases detected")
            st.dataframe(udf_fraud_df.head(100), use_container_width=True)
                        
            csv_export = udf_fraud_df.to_csv(index=False).encode("utf-8")
            st.download_button("📤 Download Flagged Cases CSV", csv_export, file_name="flagged_user_device_cases.csv")

            json_export = udf_fraud_df.to_json(orient="records", indent=4)
            st.download_button("📥 Download Flagged Profiles JSON", json_export, file_name="user_device_fraud_profiles.json")
            with st.expander("📋 Preview Fraud Profile JSON"):
                st.json(json_export)

# SHAP Explanation Section
# if "fraud_df" in st.session_state and "fraud_model" in st.session_state and st.session_state.fraud_model:
#     st.markdown("---")
#     st.subheader(" SHAP for Individual Fraud Case")
#     selected_index = st.selectbox("Select a row to explain", st.session_state.fraud_df.index.tolist(), key="row_selector")
#     if selected_index is not None:
#         row = st.session_state.fraud_df.loc[selected_index]
#         model = st.session_state.fraud_model
#         input_data = pd.DataFrame([row[model.feature_names_in_].values], columns=model.feature_names_in_)
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(input_data)
#         # Handle multi-class
#         if isinstance(shap_values, list):
#             shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
#         st.write("### Selected Row")
#         st.dataframe(row.to_frame().T, use_container_width=True)
#         st.write("### SHAP Force Plot for Selected Row")
#         try:
#             shap.initjs()
#             # Use matplotlib backend for single row
#             shap.force_plot(
#                 explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value,
#                 shap_values[0],
#                 input_data.iloc[0],
#                 matplotlib=True,
#                 show=False
#             )
#             st.pyplot(plt.gcf())
#             plt.clf()
#         except Exception as e:
#             st.warning(f"Force plot failed: {e}")
#         st.write("### SHAP Waterfall Plot")
#         try:
#             fig = shap.plots.waterfall(shap.Explanation(
#                 values=shap_values[0],
#                 base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value,
#                 data=input_data.iloc[0].values,
#                 feature_names=model.feature_names_in_
#             ), max_display=10, show=False)
#             st.pyplot(fig)
#             plt.clf()
#         except Exception as e:
#             st.warning(f"Waterfall plot failed: {e}")
#     st.subheader(" Global SHAP Importance")
#     plot_type = st.radio("Choose SHAP Plot Type", ["Bar Plot", "Beeswarm Plot"], horizontal=True)
#     plot_shap_summary(st.session_state.fraud_model, st.session_state.input_df, plot_type)
# else:
#     st.info("Please add your dataset into the 'processed_data' folder and select it.")
# SHAP Explanation Section
if "fraud_df" in st.session_state and "fraud_model" in st.session_state and st.session_state.fraud_model:
    st.markdown("---")
    st.subheader(" SHAP for Individual Fraud Case")
    selected_index = st.selectbox("Select a row to explain", st.session_state.fraud_df.index.tolist(), key="row_selector")
    if selected_index is not None:
        row = st.session_state.fraud_df.loc[selected_index]
        model = st.session_state.fraud_model
        if hasattr(model, "feature_names_"):
            feature_names = model.feature_names_
            input_data = pd.DataFrame([row[feature_names].values], columns=feature_names)
        else:
            feature_names = row.index.tolist()
            input_data = pd.DataFrame([row.values], columns=feature_names)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        st.write("### Selected Row")
        st.dataframe(row.to_frame().T, use_container_width=True)
        
        st.write("### Top Contributing Features for This Row")
        contrib = pd.DataFrame({
            'Feature': get_model_features(model),
            'SHAP Value': shap_values[0]
        }).sort_values('SHAP Value', key=np.abs, ascending=False)
        st.dataframe(contrib.head(10))
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=contrib.head(10), x='SHAP Value', y='Feature', palette='viridis', ax=ax)
        ax.set_title("Top Contributing Features for This Row")
        st.pyplot(fig)
        plt.clf()

        st.write("### SHAP Force Plot for Selected Row")
        try:
            shap.initjs()
            shap.force_plot(
                explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value,
                shap_values[0],
                input_data.iloc[0],
                matplotlib=True,
                show=False
            )
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.warning(f"Force plot failed: {e}")

        st.write("### SHAP Waterfall Plot")
        try:
            plt.figure()
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value,
                shap_values[0],
                feature_names=get_model_features(model),
                max_display=10
            )
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.warning(f"Waterfall plot failed: {e}")
    st.subheader(" Global SHAP Importance")
    plot_type = st.radio("Choose SHAP Plot Type", ["Bar Plot", "Beeswarm Plot"], horizontal=True)
    plot_shap_summary(st.session_state.fraud_model, st.session_state.input_df, plot_type)
# ==============================
#  Visual Behavior Profiling
# ==============================
if "fraud_df" in st.session_state:
    st.markdown("---")
    st.header(" Behavioral & Device Risk Profiling")

    selected_index = st.selectbox(" Select a Row for Detailed Analysis", st.session_state.fraud_df.index.tolist(), key="device_risk_index")
    row = st.session_state.fraud_df.loc[selected_index]

    st.subheader(" Device Passport Check")
    st.markdown(f"""
    - **Browser**: `{row.get('user_agent', 'N/A')}` 
    - **Screen**: `{row.get('screen_width', 'N/A')}x{row.get('screen_height', 'N/A')}` 
    - **Location (timezone)**: `{row.get('timezone', 'N/A')}`
    - **Language**: `{row.get('language', 'N/A')}`
    """)

    st.subheader(" Device Risk Indicators")
    st.markdown(f"""
    - `Browser Risk`: `{row.get('user_agent_risk', 'Low')}`
    - `Screen Resolution Risk`: `{row.get('screen_width')}x{row.get('screen_height')} - Common`
    - `Location Risk`: `{row.get('timezone', 'N/A')} - {('High Risk' if 'VPN' in str(row.get('timezone', '')) else 'Normal')}`
    """)

    st.subheader(" Behavioral Biometrics")
    st.markdown(f"""
    - **Typing Speed**: `{round(row.get('key_speed', 0), 2)} ms`
    - **Keystroke Variance**: `{round(row.get('key_variance', 0), 2)}`
    - **Mouse Movement Entropy**: `{round(row.get('mouse_entropy', 0), 2)}`
    - **Session Events**: `{row.get('session_events', 'N/A')}`
    - **Unique Actions**: `{row.get('unique_actions', 'N/A')}`
    """)

    st.subheader("Risk")
    fraud_score = row.get("Fraud_Score", row.get("Detection_Score", 0))
    color = "🟢 Green (Safe)" if fraud_score <= 0.5 else "🟡 Yellow (Check)" if fraud_score <= 0.8 else "🔴 Red (Block)"
    st.markdown(f"**Risk Score:** `{round(fraud_score, 2)}` → **{color}**")

    st.markdown("---")

else:
    st.info("Please add your dataset into the 'processed_data' folder and select it.")
