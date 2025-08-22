# app.py
"""
Fraud Detection with RF, LSTM & SHAP
- TensorFlow is lazy-loaded only if/when LSTM is selected (faster Streamlit startup)
- Artifacts expected in repo root:
    random_forest_tuned.pkl
    scaler.pkl
    feature_columns.json
    best_lstm.h5   (optional; only if "LSTM" is selected)
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

# -----------------------------------------------------------------------------
# Page config (keep very top)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Fraud Detection with RF, LSTM & SHAP", layout="wide")

# -----------------------------------------------------------------------------
# Paths (artifacts in repo root)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
RF_MODEL_PATH   = BASE_DIR / "random_forest_tuned.pkl"
SCALER_PATH     = BASE_DIR / "scaler.pkl"
FEATURES_PATH   = BASE_DIR / "feature_columns.json"
LSTM_H5_PATH    = BASE_DIR / "best_lstm.h5"   # optional

# -----------------------------------------------------------------------------
# Cached loaders (called on demand)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pickle(path: Path):
    return joblib.load(path) if path.exists() else None

@st.cache_resource(show_spinner=True)
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- TensorFlow is intentionally NOT imported at module import time ---
def _keras_custom_objects(tf):
    """Create a generous custom_objects map after TF is imported."""
    co = {
        "swish": tf.keras.activations.swish,
        "gelu": getattr(tf.keras.activations, "gelu", tf.nn.gelu),
        "leaky_relu": tf.nn.leaky_relu,
        "LeakyReLU": tf.keras.layers.LeakyReLU,
        "relu": tf.keras.activations.relu,
        "elu": tf.keras.activations.elu,
        "selu": tf.keras.activations.selu,
        "tanh": tf.keras.activations.tanh,
        "sigmoid": tf.keras.activations.sigmoid,
        "softmax": tf.keras.activations.softmax,
        "LayerNormalization": tf.keras.layers.LayerNormalization,
        "BatchNormalization": tf.keras.layers.BatchNormalization,
        "PReLU": tf.keras.layers.PReLU,
        "ELU": tf.keras.layers.ELU,
        "ReLU": tf.keras.layers.ReLU,
        "Dense": tf.keras.layers.Dense,
        "Dropout": tf.keras.layers.Dropout,
        "LSTM": tf.keras.layers.LSTM,
        "GRU": tf.keras.layers.GRU,
        "RNN": tf.keras.layers.RNN,
        "tf": tf,
    }
    return co

@st.cache_resource(show_spinner=True)
def load_lstm_safe(path: Path):
    if not path.exists():
        return None, None
    try:
        import tensorflow as tf  # lazy import
    except Exception as e_imp:
        return None, f"TensorFlow import failed: {repr(e_imp)}"
    try:
        m = tf.keras.models.load_model(str(path), compile=False)
        return m, None
    except Exception as e1:
        try:
            m = tf.keras.models.load_model(
                str(path), compile=False, custom_objects=_keras_custom_objects(tf)
            )
            return m, None
        except Exception as e2:
            return None, f"LSTM load failed.\nFirst error: {repr(e1)}\nSecond error: {repr(e2)}"

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def st_shap(plot_html, height=320):
    components.html(plot_html, height=height)

def safe_str(x):
    try:
        return str(x)
    except Exception:
        return "Other"

def extract_categorical_bases_from_features(features):
    bases = [c.split("_")[0] for c in features if "_" in c]
    return sorted(list(set(bases)))

def preprocess_uploaded(df_raw, features, cardinality_threshold=100):
    log = {"dropped_high_cardinality": [], "encoded": [], "added_missing": [], "dropped_extra": []}
    df = df_raw.copy()
    categorical_bases = extract_categorical_bases_from_features(features)
    for c in list(df.columns):
        if df[c].dtype == "object" and c not in categorical_bases:
            nunique = df[c].nunique(dropna=True)
            if nunique > cardinality_threshold:
                df.drop(columns=[c], inplace=True)
                log["dropped_high_cardinality"].append((c, nunique))
    for cat in categorical_bases:
        if cat in df.columns:
            df[cat] = df[cat].apply(safe_str).fillna("Other")
            dummies = pd.get_dummies(df[cat], prefix=cat, drop_first=False)
            dummies = dummies[[c for c in dummies.columns if c in features]]
            df = pd.concat([df.drop(columns=[cat]), dummies], axis=1)
            log["encoded"].append(cat)
    for col in features:
        if col not in df.columns:
            df[col] = 0
            log["added_missing"].append(col)
    extras = [c for c in df.columns if c not in features]
    if extras:
        df = df.drop(columns=extras)
        log["dropped_extra"].extend(extras)
    df = df[features]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df, log

def predict_rf(rf_model, X_scaled):
    probs = rf_model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def predict_lstm(lstm_model, X_scaled_2d):
    X_seq = X_scaled_2d.reshape((X_scaled_2d.shape[0], 1, X_scaled_2d.shape[1]))
    probs = lstm_model.predict(X_seq, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    return preds, probs

# --- Robust CSV reader --------------------------------------------------------
def read_csv_robust(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        pass
    uploaded_file.seek(0)
    for sep in [",", ";", "\t", "|"]:
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, sep=sep, encoding=enc)
            except Exception:
                continue
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
def main():
    st.title("💳 Fraud Detection with RF, LSTM & SHAP")

    # Sidebar
    st.sidebar.header("Model & Options")
    model_choice = st.sidebar.selectbox("Model", ["Random Forest", "LSTM"], index=0)
    background_size = st.sidebar.slider(
        "Background sample for SHAP (for LSTM explainer)",
        min_value=10, max_value=300, value=64, step=8
    )
    nsamples_kernel = st.sidebar.slider(
        "SHAP nsamples (KernelExplainer fallback)",
        min_value=50, max_value=400, value=150, step=25
    )

    st.header("1) Upload CSV")
    uploaded = st.file_uploader("Upload a raw CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_uploaded = read_csv_robust(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()
    else:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

    # Load artifacts
    rf_model = load_pickle(RF_MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
    feature_columns = load_json(FEATURES_PATH)

    if rf_model is None or scaler is None or feature_columns is None:
        st.error("Missing required artifacts. Ensure RF model, scaler, and feature_columns.json are in the repo root.")
        st.stop()

    if isinstance(feature_columns, dict) and "feature_columns" in feature_columns:
        feature_columns = list(feature_columns["feature_columns"])
    elif isinstance(feature_columns, list):
        feature_columns = list(feature_columns)
    else:
        st.error("feature_columns.json could not be parsed into a list.")
        st.stop()

    df_features, prep_log = preprocess_uploaded(df_uploaded, feature_columns)
    st.success(f"✅ Preprocessed: {df_features.shape[0]} rows × {df_features.shape[1]} features.")
    with st.expander("Preprocessing log"):
        st.json(prep_log)

    try:
        X_scaled = scaler.transform(df_features.values.astype(np.float32))
    except Exception as e:
        st.error(f"Scaler failed: {e}")
        st.stop()

    st.header("2) Pick a row to Predict & Explain")
    row_index = st.number_input(
        "Row index (0-based)",
        min_value=0, max_value=max(0, len(df_features)-1), value=0, step=1
    )

    lstm_model = None
    lstm_error = None
    if model_choice == "LSTM":
        lstm_model, lstm_error = load_lstm_safe(LSTM_H5_PATH)
        if lstm_model is None:
            st.error("Could not load LSTM model from 'best_lstm.h5'. Using Random Forest is still available.")
            with st.expander("Show LSTM load error details"):
                st.code(lstm_error or "No details available.")

    if st.button("🔮 Predict & Explain Selected Row"):
        if model_choice == "Random Forest" or (model_choice == "LSTM" and lstm_model is None):
            preds, probs = predict_rf(rf_model, X_scaled)
            explanation = shap.TreeExplainer(rf_model).shap_values(X_scaled[row_index:row_index+1])[1]
        else:
            preds, probs = predict_lstm(lstm_model, X_scaled)
            explanation = shap.DeepExplainer(lstm_model, X_scaled[:64].reshape(-1, 1, X_scaled.shape[1])).shap_values(
                X_scaled[row_index:row_index+1].reshape(1, 1, -1)
            )[0].reshape(-1)

        pred_label = "🚨 Fraudulent" if preds[row_index] == 1 else "✅ Legitimate"
        st.subheader("Prediction")
        st.write(f"**Row {row_index} Prediction:** {pred_label}")
        st.write(f"**Probability of Fraud:** {probs[row_index]:.2%}")

        st.subheader("Explanation (SHAP)")
        st.bar_chart(pd.Series(explanation, index=feature_columns).sort_values(key=abs, ascending=False).head(10))

if __name__ == "__main__":
    main()
