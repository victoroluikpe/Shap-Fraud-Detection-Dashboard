# app.py
"""
Fraud Detection Dashboard 
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components


# ========= Streamlit page config =========
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

st.title(" Fraud Detection Dashboard")
st.markdown("Predict and explain fraud using Random Forest and LSTM models.")


# ========= Load Artifacts =========
@st.cache_resource
def load_artifacts():
    try:
        rf_model = joblib.load("random_forest_tuned.pkl")
    except:
        rf_model = None

    try:
        lstm_model = tf.keras.models.load_model("best_lstm.h5")
    except:
        lstm_model = None

    try:
        scaler = joblib.load("scaler.pkl")
    except:
        scaler = None

    try:
        with open("feature_columns.json") as f:
            feature_columns = json.load(f)
    except:
        feature_columns = None

    return rf_model, lstm_model, scaler, feature_columns


rf_model, lstm_model, scaler, feature_columns = load_artifacts()


# ========= File Uploader =========
uploaded_file = st.file_uploader("📂 Upload CSV with transactions", type=["csv"])

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df_uploaded.head())

    if feature_columns is None or scaler is None:
        st.error("❌ Feature columns or scaler missing. Please ensure artifacts are uploaded.")
    else:
        # Guarantee all features exist
        for col in feature_columns:
            if col not in df_uploaded.columns:
                df_uploaded[col] = 0

        df_features = df_uploaded[feature_columns].fillna(0)
        X_scaled = scaler.transform(df_features.values.astype(np.float32))

        # ========= Random Forest Prediction =========
        if rf_model is not None:
            st.subheader("🌲 Random Forest Predictions")
            rf_preds = rf_model.predict(X_scaled)
            df_uploaded["RF_Prediction"] = rf_preds
            st.write(df_uploaded[["RF_Prediction"]].head())

            # SHAP for RF
            explainer_rf = shap.TreeExplainer(rf_model)
            shap_values_rf = explainer_rf.shap_values(df_features)

            st.markdown("**Feature Importance (Random Forest)**")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values_rf, df_features, plot_type="bar", show=False)
            st.pyplot(fig)

        # ========= LSTM Prediction =========
        if lstm_model is not None:
            st.subheader("🔮 LSTM Predictions")

            # Reshape for LSTM
            X_lstm = np.expand_dims(X_scaled, axis=1)
            lstm_preds = (lstm_model.predict(X_lstm) > 0.5).astype(int).flatten()
            df_uploaded["LSTM_Prediction"] = lstm_preds
            st.write(df_uploaded[["LSTM_Prediction"]].head())

            # SHAP for LSTM (KernelExplainer)
            st.markdown("**Feature Importance (LSTM)**")
            try:
                explainer_lstm = shap.KernelExplainer(lstm_model.predict, X_scaled[:50])
                shap_values_lstm = explainer_lstm.shap_values(X_scaled[:50], nsamples=50)

                fig, ax = plt.subplots()
                shap.summary_plot(shap_values_lstm, df_features.iloc[:50], show=False)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"SHAP for LSTM skipped: {e}")

else:
    st.info("👆 Upload a CSV file to start predictions.")
