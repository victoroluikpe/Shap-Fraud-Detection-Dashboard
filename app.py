# app.py
"""
Fraud Detection Dashboard

- Artifacts expected in BASE DIR (same folder as app.py):
    random_forest_tuned.pkl
    scaler.pkl
    feature_columns.json
    best_lstm.h5
    best_lstm_quantized.tflite
    random_forest_tuned.onnx
    random_forest_tuned_int8.onnx
    model.ipynb (not loaded, just stored)
- Random Forest loaded at start (fast)
- LSTM lazy-loaded only when selected
- Missing trained feature columns in uploaded CSV are added as zeros (prevents KeyError)
- Per-row predict & explain (button)
- SHAP: TreeExplainer for RF; DeepExplainer → KernelExplainer fallback for LSTM
- Download predictions button included
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# --------------------------
# Config / paths
# --------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

BASE_DIR = Path(__file__).parent  # project base folder

# Model artifact paths
RF_MODEL_PATH = BASE_DIR / "random_forest_tuned.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.json"
LSTM_H5_PATH = BASE_DIR / "best_lstm.h5" 
LSTM_TFLITE_PATH = BASE_DIR / "best_lstm_quantized.tflite"  # not used in app yet
RF_ONNX_PATH = BASE_DIR / "random_forest_tuned.onnx"        # not used in app yet
RF_ONNX_INT8_PATH = BASE_DIR / "random_forest_tuned_int8.onnx"  # not used in app yet

# --------------------------
# Cached loaders
# --------------------------
@st.cache_resource
def load_pickle(path: Path):
    return joblib.load(path) if path.exists() else None

@st.cache_resource
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --------------------------
# LSTM loader (robust)
# --------------------------
@st.cache_resource
def load_lstm_safe(h5_path: Path) -> Tuple[Optional[object], Optional[str]]:
    """
    Try to load LSTM:
    - try .h5 (since SavedModel dir not used here)
    """
    try:
        import tensorflow as tf
    except Exception as e:
        return None, f"TensorFlow import failed: {repr(e)}"

    if h5_path.exists():
        try:
            m = tf.keras.models.load_model(str(h5_path), compile=False)
            return m, None
        except Exception as e:
            return None, f".h5 load failed: {repr(e)}"
    else:
        return None, ".h5 file not found."

# --------------------------
# Utilities: CSV + features
# --------------------------
def read_csv_robust(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
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

def align_features_fill_zeros(df: pd.DataFrame, feature_columns):
    df = df.copy()
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# --------------------------
# Predictors
# --------------------------
def predict_rf(rf_model, X_scaled):
    probs = rf_model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def predict_lstm(lstm_model, X_scaled_2d):
    X_seq = X_scaled_2d.reshape((X_scaled_2d.shape[0], 1, X_scaled_2d.shape[1]))
    out = lstm_model.predict(X_seq, verbose=0)
    out = np.asarray(out).reshape(out.shape[0], -1)
    probs = out[:, 1] if out.shape[1] > 1 else out[:, 0]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

# --------------------------
# SHAP helpers (RF + LSTM)
# --------------------------
def explain_rf_instance(rf_model, x_row_2d, feature_names):
    explainer = shap.TreeExplainer(rf_model)
    out = explainer.shap_values(x_row_2d)
    if isinstance(out, shap.Explanation):
        return out
    if isinstance(out, list):
        idx = 1 if len(out) > 1 else 0
        values = out[idx][0]
    else:
        values = np.array(out).reshape(-1)

    # 🔧 Ensure length matches features
    if len(values) > len(feature_names):
        values = values[:len(feature_names)]
    elif len(values) < len(feature_names):
        values = np.pad(values, (0, len(feature_names) - len(values)))

    return shap.Explanation(values=values,
                            base_values=np.mean(rf_model.predict_proba(x_row_2d)[:,1]),
                            data=x_row_2d[0], feature_names=feature_names)

def explain_lstm_instance(lstm_model, X_scaled_all, idx, feature_names, bg_size=64, nsamples=150):
    n = X_scaled_all.shape[0]
    bg_size = int(max(10, min(bg_size, n)))
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(n, size=bg_size, replace=False) if n > bg_size else np.arange(n)
    background_2d = X_scaled_all[bg_idx]
    background_3d = background_2d.reshape((background_2d.shape[0], 1, background_2d.shape[1]))
    x0_2d = X_scaled_all[idx:idx+1]
    x0_3d = x0_2d.reshape((1, 1, X_scaled_all.shape[1]))

    try:
        explainer = shap.DeepExplainer(lstm_model, background_3d)
        sv = explainer.shap_values(x0_3d)
        values = sv[0][0] if isinstance(sv, list) else sv[0]
    except Exception:
        def f(x2d):
            xseq = x2d.reshape((x2d.shape[0], 1, x2d.shape[1]))
            out = lstm_model.predict(xseq, verbose=0)
            out = np.asarray(out).reshape(out.shape[0], -1)
            return out[:, 1] if out.shape[1] > 1 else out[:, 0]
        explainer = shap.KernelExplainer(f, background_2d)
        sv = explainer.shap_values(x0_2d, nsamples=min(nsamples, 2 * background_2d.shape[1] + 1))
        values = sv[0] if isinstance(sv, list) else sv

    values = np.array(values).reshape(-1)
    if len(values) > len(feature_names):
        values = values[:len(feature_names)]
    elif len(values) < len(feature_names):
        values = np.pad(values, (0, len(feature_names) - len(values)))

    return shap.Explanation(values=values,
                            base_values=np.mean(lstm_model.predict(x0_3d, verbose=0)),
                            data=x0_2d[0], feature_names=feature_names)

def shap_force_plot_html(explanation):
    try:
        obj = shap.plots.force(explanation, matplotlib=False)
        html = obj.html() if hasattr(obj, "html") else str(obj)
    except Exception:
        html = str(explanation)
    return f"<head>{shap.getjs()}</head><body>{html}</body>"

def st_shap(plot_html, height=360):
    components.html(plot_html, height=height)

# --------------------------
# App UI
# --------------------------
def main():
    st.title("Fraud Detection Dashboard")
    st.write("Upload a CSV, choose model, pick a row, then click Predict & Explain.")

    model_choice = st.sidebar.selectbox("Model", ["Random Forest", "LSTM"])
    background_size = st.sidebar.slider("LSTM SHAP background size", 10, 300, 64, 8)
    nsamples_kernel = st.sidebar.slider("LSTM KernelExplainer nsamples", 50, 400, 150, 25)

    # Upload CSV
    st.header("1) Upload CSV")
    uploaded = st.file_uploader("Upload raw CSV", type=["csv"])
    if uploaded is None:
        st.info("Please upload your transactions CSV file to continue.")
        st.stop()

    df_uploaded = read_csv_robust(uploaded)

    # Load RF artifacts
    rf_model = load_pickle(RF_MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
    feature_columns = load_json(FEATURES_PATH)

    if rf_model is None or scaler is None or feature_columns is None:
        st.error("❌ Missing required artifacts in base directory.")
        st.write("Expected files:", RF_MODEL_PATH, SCALER_PATH, FEATURES_PATH)
        st.stop()
    if isinstance(feature_columns, dict) and "feature_columns" in feature_columns:
        feature_columns = feature_columns["feature_columns"]

    df_aligned = align_features_fill_zeros(df_uploaded, feature_columns)
    X_scaled = scaler.transform(df_aligned.values.astype(np.float32))
    st.success(f"✅ Preprocessed: {df_aligned.shape[0]} rows × {df_aligned.shape[1]} features.")

    row_index = st.number_input("Row index", 0, len(df_aligned)-1, 0)

    lstm_model, lstm_error = (None, None)
    if model_choice == "LSTM":
        lstm_model, lstm_error = load_lstm_safe(LSTM_H5_PATH)
        if lstm_model is None:
            st.warning("⚠️ LSTM could not be loaded. Random Forest is still available.")
            with st.expander("LSTM load error details"):
                st.code(lstm_error or "No details available.")

    if st.button("🔮 Predict & Explain Selected Row"):
        if model_choice == "Random Forest" or (model_choice == "LSTM" and lstm_model is None):
            preds, probs = predict_rf(rf_model, X_scaled)
            explanation = explain_rf_instance(rf_model, X_scaled[row_index:row_index+1], feature_columns)
            used_model = "Random Forest"
        else:
            preds, probs = predict_lstm(lstm_model, X_scaled)
            explanation = explain_lstm_instance(lstm_model, X_scaled, row_index, feature_columns,
                                                bg_size=background_size, nsamples=nsamples_kernel)
            used_model = "LSTM"

        st.subheader("Prediction")
        pred_label = "🚨 Fraudulent" if preds[row_index] == 1 else "✅ Legitimate"
        st.write(f"**Model used:** {used_model}")
        st.write(f"Prediction: {pred_label}")
        st.write(f"Fraud probability: {probs[row_index]:.2%}")

        st.subheader("Explanation (SHAP)")
        try:
            fig, _ = plt.subplots()
            shap.plots.waterfall(explanation, max_display=12, show=False)
            st.pyplot(fig)
        except Exception:
            vals = pd.Series(np.array(explanation.values).reshape(-1), index=feature_columns)
            top = vals.abs().sort_values(ascending=False).head(12)
            fig, ax = plt.subplots()
            ax.barh(top.index, vals.loc[top.index])
            ax.invert_yaxis()
            st.pyplot(fig)

        try:
            html = shap_force_plot_html(explanation)
            st_shap(html, height=360)
        except Exception:
            st.warning("Force plot failed.")

        out_df = df_uploaded.copy()
        out_df["RF_Prediction"] = predict_rf(rf_model, X_scaled)[0]
        if lstm_model is not None:
            out_df["LSTM_Prediction"] = predict_lstm(lstm_model, X_scaled)[0]
        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions", csv, "predictions.csv", "text/csv")

if __name__ == "__main__":
    main()
