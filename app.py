# app.py
"""
Fraud Detection Dashboard
- Both Random Forest & LSTM exposed independently
- No interference: failure in one model does not block the other
- Robust artifact loading (repo root or optional upload on Streamlit Cloud)
- SHAP explanations with graceful fallbacks
"""

import warnings
warnings.filterwarnings("ignore")

import json
import io
import tempfile
from pathlib import Path
from typing import Tuple, Optional

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

# ---------------------------------------------------------------------
# Paths (default artifacts location is repo root)
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
RF_MODEL_PATH = BASE_DIR / "random_forest_tuned.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.json"
LSTM_H5_PATH = BASE_DIR / "best_lstm.h5"

# ---------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pickle(path: Path):
    if not path or not Path(path).exists():
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner=True)
def load_json(path: Path):
    if not path or not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------
# LSTM robust loader
# ---------------------------------------------------------------------
def _keras_custom_objects(tf):
    return {
        "LSTM": tf.keras.layers.LSTM,
        "GRU": tf.keras.layers.GRU,
        "Dense": tf.keras.layers.Dense,
        "Dropout": tf.keras.layers.Dropout,
        "BatchNormalization": tf.keras.layers.BatchNormalization,
        "LayerNormalization": tf.keras.layers.LayerNormalization,
        "LeakyReLU": tf.keras.layers.LeakyReLU,
        "PReLU": tf.keras.layers.PReLU,
        "ELU": tf.keras.layers.ELU,
        "tf": tf,
    }

@st.cache_resource(show_spinner=True)
def load_lstm_safe(path: Optional[Path]) -> Tuple[Optional[object], Optional[str], Optional[str]]:
    if path is None or not Path(path).exists():
        return None, "File not found", None
    try:
        import tensorflow as tf
    except Exception as e_imp:
        return None, f"TensorFlow import failed: {repr(e_imp)}", None
    try:
        m = tf.keras.models.load_model(str(path), compile=False)
        return m, None, tf.__version__
    except Exception as e1:
        try:
            m = tf.keras.models.load_model(str(path), compile=False, custom_objects=_keras_custom_objects(tf))
            return m, None, tf.__version__
        except Exception as e2:
            return None, f"LSTM load failed.\nError1: {e1}\nError2: {e2}", tf.__version__

# ---------------------------------------------------------------------
# Utils: CSV, preprocessing, predictions
# ---------------------------------------------------------------------
def read_csv(uploaded_file):
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)

def preprocess_uploaded(df_raw, features):
    df = df_raw.copy()
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def predict_rf(rf_model, scaler, X):
    Xs = scaler.transform(X)
    probs = rf_model.predict_proba(Xs)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def predict_lstm(lstm_model, scaler, X):
    Xs = scaler.transform(X)
    Xs = Xs.reshape((Xs.shape[0], 1, Xs.shape[1]))
    out = lstm_model.predict(Xs, verbose=0)
    out = np.asarray(out).reshape(out.shape[0], -1)
    probs = out[:, 1] if out.shape[1] > 1 else out[:, 0]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

# ---------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------
def explain_rf(rf_model, X, feature_names):
    explainer = shap.TreeExplainer(rf_model)
    vals = explainer.shap_values(X)
    return vals, explainer.expected_value

def explain_lstm(lstm_model, X, feature_names):
    import tensorflow as tf
    explainer = shap.DeepExplainer(lstm_model, X[:50].reshape((50, 1, X.shape[1])))
    vals = explainer.shap_values(X[:1].reshape((1, 1, X.shape[1])))
    return vals, explainer.expected_value

# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
def main():
    st.title("Fraud Detection Dashboard")

    st.sidebar.header("Controls")
    model_choice = st.sidebar.radio("Choose model", ["Random Forest", "LSTM"])
    allow_upload = st.sidebar.checkbox("Upload artifacts manually", value=False)

    # Optional uploads (Streamlit Cloud helper)
    if allow_upload:
        rf_file = st.sidebar.file_uploader("Random Forest (.pkl)")
        scaler_file = st.sidebar.file_uploader("Scaler (.pkl)")
        feat_file = st.sidebar.file_uploader("feature_columns.json")
        lstm_file = st.sidebar.file_uploader("LSTM (.h5)")

        if rf_file: 
            RF_MODEL_PATH.write_bytes(rf_file.read())
        if scaler_file: 
            SCALER_PATH.write_bytes(scaler_file.read())
        if feat_file: 
            FEATURES_PATH.write_bytes(feat_file.read())
        if lstm_file: 
            LSTM_H5_PATH.write_bytes(lstm_file.read())

    # Load artifacts
    rf_model = load_pickle(RF_MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
    features = load_json(FEATURES_PATH)
    if isinstance(features, dict) and "feature_columns" in features:
        features = features["feature_columns"]

    # Upload dataset
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    if not uploaded:
        st.info("Upload CSV to continue")
        st.stop()
    df_uploaded = read_csv(uploaded)
    df_features = preprocess_uploaded(df_uploaded, features)

    # Predictions
    if model_choice == "Random Forest":
        if rf_model is None or scaler is None or features is None:
            st.error("Random Forest artifacts missing.")
            st.stop()
        preds, probs = predict_rf(rf_model, scaler, df_features)
        st.success(f"Prediction on first row: {preds[0]} (prob={probs[0]:.2%})")
        if st.checkbox("Show SHAP explanation"):
            vals, base = explain_rf(rf_model, scaler.transform(df_features), features)
            shap.summary_plot(vals, df_features, show=False)
            st.pyplot(bbox_inches="tight")

    elif model_choice == "LSTM":
        lstm_model, lstm_err, tfv = load_lstm_safe(LSTM_H5_PATH)
        if lstm_model is None:
            st.error(f"LSTM not available. {lstm_err}")
            st.stop()
        preds, probs = predict_lstm(lstm_model, scaler, df_features)
        st.success(f"Prediction on first row: {preds[0]} (prob={probs[0]:.2%})")
        if st.checkbox("Show SHAP explanation"):
            try:
                vals, base = explain_lstm(lstm_model, df_features.values, features)
                shap.summary_plot(vals, df_features, show=False)
                st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.warning(f"SHAP for LSTM failed: {e}")

if __name__ == "__main__":
    main()
