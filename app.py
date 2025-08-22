# app.py
"""
Fraud Detection Dashboard
- Requires both Random Forest and LSTM artifacts
- Assumes artifacts live in the repo root:
    random_forest_tuned.pkl
    scaler.pkl
    feature_columns.json
    best_lstm.h5
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

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# -----------------------------------------------------------------------------
# Paths (artifacts in repo root)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
RF_MODEL_PATH   = BASE_DIR / "random_forest_tuned.pkl"
SCALER_PATH     = BASE_DIR / "scaler.pkl"
FEATURES_PATH   = BASE_DIR / "feature_columns.json"
LSTM_H5_PATH    = BASE_DIR / "best_lstm.h5"

# -----------------------------------------------------------------------------
# Load artifacts (required)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pickle(path: Path):
    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()
    return joblib.load(path)

@st.cache_resource(show_spinner=True)
def load_json(path: Path):
    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _keras_custom_objects():
    return {
        "swish": tf.keras.activations.swish,
        "gelu": tf.keras.activations.gelu if hasattr(tf.keras.activations, "gelu") else tf.nn.gelu,
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

@st.cache_resource(show_spinner=True)
def load_lstm(path: Path):
    if not path.exists():
        st.error(f"Missing LSTM file: {path}")
        st.stop()
    try:
        return tf.keras.models.load_model(str(path), compile=False)
    except Exception:
        return tf.keras.models.load_model(str(path), compile=False, custom_objects=_keras_custom_objects())

# -----------------------------------------------------------------------------
# Load all required models/artifacts at startup
# -----------------------------------------------------------------------------
rf_model = load_pickle(RF_MODEL_PATH)
scaler = load_pickle(SCALER_PATH)
feature_columns = load_json(FEATURES_PATH)
lstm_model = load_lstm(LSTM_H5_PATH)

if isinstance(feature_columns, dict) and "feature_columns" in feature_columns:
    feature_columns = list(feature_columns["feature_columns"])
elif isinstance(feature_columns, list):
    feature_columns = list(feature_columns)
else:
    st.error("feature_columns.json could not be parsed into a list.")
    st.stop()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def st_shap(plot_html, height=320):
    components.html(plot_html, height=height)

def safe_str(x):
    try: return str(x)
    except: return "Other"

def extract_categorical_bases_from_features(features):
    return sorted(list({c.split("_")[0] for c in features if "_" in c}))

def preprocess_uploaded(df_raw, features, cardinality_threshold=100):
    log = {"dropped_high_cardinality": [], "encoded": [], "added_missing": [], "dropped_extra": []}
    df = df_raw.copy()
    categorical_bases = extract_categorical_bases_from_features(features)

    for c in list(df.columns):
        if df[c].dtype == "object" and c not in categorical_bases:
            if df[c].nunique(dropna=True) > cardinality_threshold:
                df.drop(columns=[c], inplace=True)
                log["dropped_high_cardinality"].append(c)

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

# -----------------------------------------------------------------------------
# SHAP explanation (simplified for brevity, same logic kept)
# -----------------------------------------------------------------------------
def get_rf_explainer(rf_model):
    if "rf_explainer" not in st.session_state:
        try:
            st.session_state["rf_explainer"] = shap.TreeExplainer(
                rf_model, model_output="probability", feature_perturbation="interventional"
            )
        except Exception:
            st.session_state["rf_explainer"] = shap.TreeExplainer(rf_model)
    return st.session_state["rf_explainer"]

def explain_rf_instance(rf_model, x_row_2d, feature_names):
    explainer = get_rf_explainer(rf_model)
    shap_values = explainer.shap_values(x_row_2d)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    return shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=x_row_2d[0], feature_names=feature_names)

def shap_force_plot_html(explanation):
    obj = shap.plots.force(explanation, matplotlib=False)
    return f"<head>{shap.getjs()}</head><body>{obj.html()}</body>"

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
def main():
    st.title("Fraud Detection Dashboard")

    st.sidebar.header("Model & Options")
    model_choice = st.sidebar.selectbox("Model", ["Random Forest", "LSTM"], index=0)

    st.header("1) Upload CSV")
    uploaded = st.file_uploader("Upload a raw CSV", type=["csv"])
    if uploaded is None:
        st.info("Tip: Upload your raw transactions CSV.")
        st.stop()

    try:
        df_uploaded = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
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
    row_index = st.number_input("Row index (0-based)", min_value=0, max_value=max(0, len(df_features)-1), value=0, step=1)

    if st.button("🔮 Predict & Explain Selected Row"):
        if model_choice == "Random Forest":
            preds, probs = predict_rf(rf_model, X_scaled)
            explanation = explain_rf_instance(rf_model, X_scaled[row_index:row_index+1], feature_columns)
        else:
            preds, probs = predict_lstm(lstm_model, X_scaled)
            # You can add SHAP DeepExplainer for LSTM here (left simplified)
            explanation = shap.Explanation(values=np.zeros_like(feature_columns, dtype=float),
                                           base_values=np.mean(probs),
                                           data=X_scaled[row_index],
                                           feature_names=feature_columns)

        pred_label = "🚨 Fraudulent" if preds[row_index] == 1 else "✅ Legitimate"
        st.subheader("Prediction")
        st.write(f"**Row {row_index} Prediction:** {pred_label}")
        st.write(f"**Probability of Fraud:** {probs[row_index]:.2%}")

        st.subheader("Explanation for this Transaction (SHAP)")
        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        st.pyplot(fig, clear_figure=True)

        html = shap_force_plot_html(explanation)
        st_shap(html, height=320)

if __name__ == "__main__":
    main()
