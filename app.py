# app.py
"""
Fraud Detection Dashboard
- Random Forest and LSTM are loaded independently
- No upload controls for artifacts (only CSV upload for input data)
- Fixed LSTM deserialization across TF versions
- Fixed RF pyplot warnings
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

BASE_DIR = Path(__file__).parent
RF_MODEL_PATH   = BASE_DIR / "random_forest_tuned.pkl"
SCALER_PATH     = BASE_DIR / "scaler.pkl"
FEATURES_PATH   = BASE_DIR / "feature_columns.json"
LSTM_H5_PATH    = BASE_DIR / "best_lstm.h5"

# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
@st.cache_resource
def load_pickle(path: Path):
    return joblib.load(path) if path.exists() else None

@st.cache_resource
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def _keras_custom_objects(tf):
    return {
        "LeakyReLU": tf.keras.layers.LeakyReLU,
        "swish": tf.keras.activations.swish,
        "gelu": getattr(tf.keras.activations, "gelu", tf.nn.gelu),
    }

@st.cache_resource
def load_lstm_safe(path: Path):
    if not path.exists():
        return None, "best_lstm.h5 not found"
    try:
        import tensorflow as tf
    except Exception as e:
        return None, f"TensorFlow import failed: {e}"

    # First try normally
    try:
        return tf.keras.models.load_model(str(path), compile=False), None
    except Exception as e1:
        # Second try: patch config
        from tensorflow.keras.models import model_from_json
        import h5py, io
        try:
            with h5py.File(str(path), "r") as f:
                model_config = f.attrs.get("model_config")
            if model_config is not None:
                model_json = model_config.decode("utf-8")
                # strip unsupported keys
                model_json = model_json.replace('"batch_shape": [null, 1, 45],', "")
                model = model_from_json(model_json, custom_objects=_keras_custom_objects(tf))
                model.load_weights(str(path))
                return model, None
        except Exception as e2:
            return None, f"LSTM load failed.\nError1: {e1}\nError2: {e2}"

# ---------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------
def predict_rf(model, X_scaled):
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def predict_lstm(model, X_scaled):
    X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    probs = model.predict(X_seq, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    return preds, probs

# ---------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------
def shap_force_plot_html(explanation):
    obj = shap.plots.force(explanation, matplotlib=False)
    html = obj.html() if hasattr(obj, "html") else str(obj)
    return f"<head>{shap.getjs()}</head><body>{html}</body>"

def st_shap(plot_html, height=320):
    components.html(plot_html, height=height)

# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
def main():
    st.title("Fraud Detection Dashboard")

    model_choice = st.sidebar.radio("Select Model", ["Random Forest", "LSTM"])

    st.header("1) Upload CSV")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if not uploaded:
        st.info("Upload your transactions CSV to continue.")
        st.stop()

    try:
        df_uploaded = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    rf_model = load_pickle(RF_MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
    feature_columns = load_json(FEATURES_PATH)

    if isinstance(feature_columns, dict) and "feature_columns" in feature_columns:
        feature_columns = feature_columns["feature_columns"]

    if rf_model is None or scaler is None or feature_columns is None:
        st.error("Missing required artifacts (RF model, scaler, or feature_columns.json).")
        st.stop()

    df_features = df_uploaded[feature_columns].fillna(0)
    X_scaled = scaler.transform(df_features.values.astype(np.float32))

    row_index = st.number_input("Row index", min_value=0, max_value=len(df_features)-1, value=0)

    lstm_model, lstm_error = (None, None)
    if model_choice == "LSTM":
        lstm_model, lstm_error = load_lstm_safe(LSTM_H5_PATH)

    if st.button("🔮 Predict & Explain"):
        if model_choice == "Random Forest":
            preds, probs = predict_rf(rf_model, X_scaled)
            explainer = shap.TreeExplainer(rf_model)
            explanation = explainer(X_scaled[row_index:row_index+1])
        elif model_choice == "LSTM":
            if lstm_model is None:
                st.error(f"LSTM not available. {lstm_error}")
                st.stop()
            preds, probs = predict_lstm(lstm_model, X_scaled)
            explanation = shap.Explanation(
                values=np.zeros(len(feature_columns)),
                base_values=probs[row_index],
                data=X_scaled[row_index],
                feature_names=feature_columns,
            )

        st.subheader("Prediction")
        st.write(f"Row {row_index}: {'🚨 Fraud' if preds[row_index] else '✅ Legitimate'}")
        st.write(f"Probability of Fraud: {probs[row_index]:.2%}")

        st.subheader("Explanation (SHAP)")
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(explanation, show=False, max_display=10)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Waterfall plot failed: {e}")

        try:
            st_shap(shap_force_plot_html(explanation))
        except Exception as e:
            st.warning(f"Force plot failed: {e}")

if __name__ == "__main__":
    main()
