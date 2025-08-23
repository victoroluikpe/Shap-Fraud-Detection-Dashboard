# app.py
"""
Fraud Detection Dashboard

- No artifact upload UI (artifacts expected in repo root):
    random_forest_tuned.pkl
    scaler.pkl
    feature_columns.json
    best_lstm.h5   (optional; only if LSTM is used)
- Random Forest loaded at start (fast)
- LSTM lazy-loaded only when selected (prevents TF cold-start unless needed)
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
BASE_DIR = Path(__file__).parent
RF_MODEL_PATH = BASE_DIR / "random_forest_tuned.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.json"
LSTM_H5_PATH = BASE_DIR / "best_lstm.h5"  # optional

# --------------------------
# Lightweight cached loaders
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
# Lazy robust LSTM loader
# --------------------------
def _keras_custom_objects(tf):
    # generous mapping for some custom layers/activations that models often use
    co = {
        "swish": getattr(tf.keras.activations, "swish", None),
        "gelu": getattr(tf.keras.activations, "gelu", getattr(tf.nn, "gelu", None)),
        "LeakyReLU": getattr(tf.keras.layers, "LeakyReLU", None),
        "PReLU": getattr(tf.keras.layers, "PReLU", None),
        "ELU": getattr(tf.keras.layers, "ELU", None),
        "LayerNormalization": getattr(tf.keras.layers, "LayerNormalization", None),
        "BatchNormalization": getattr(tf.keras.layers, "BatchNormalization", None),
        "Dense": getattr(tf.keras.layers, "Dense", None),
        "Dropout": getattr(tf.keras.layers, "Dropout", None),
        "LSTM": getattr(tf.keras.layers, "LSTM", None),
        "GRU": getattr(tf.keras.layers, "GRU", None),
        "RNN": getattr(tf.keras.layers, "RNN", None),
        "tf": tf,
    }
    return {k: v for k, v in co.items() if v is not None}

@st.cache_resource
def load_lstm_safe(path: Path) -> Tuple[Optional[object], Optional[str]]:
    """
    Try to load a Keras .h5 model robustly:
    - lazy import tensorflow
    - try load_model(..., compile=False)
    - then try with custom_objects
    - returns (model_or_None, error_message_or_None)
    """
    if not path.exists():
        return None, "LSTM file not found at path."
    try:
        import tensorflow as tf  # lazy import
    except Exception as e_imp:
        return None, f"TensorFlow import failed: {repr(e_imp)}"

    try:
        m = tf.keras.models.load_model(str(path), compile=False)
        return m, None
    except Exception as e1:
        try:
            m = tf.keras.models.load_model(str(path), compile=False, custom_objects=_keras_custom_objects(tf))
            return m, None
        except Exception as e2:
            return None, f"LSTM load failed.\nFirst error: {repr(e1)}\nSecond (with custom_objects): {repr(e2)}"

# --------------------------
# Utilities: CSV read + safe feature alignment
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
    # final attempt (will raise)
    return pd.read_csv(uploaded_file)

def align_features_fill_zeros(df: pd.DataFrame, feature_columns):
    """
    Ensure df contains all feature_columns. If a column is missing, add it as zeros.
    This avoids KeyError when uploaded CSV contains raw rather than one-hot features.
    """
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
# SHAP helpers
# --------------------------
def _coerce_single_row_single_output(values, base_values, x_row_2d, class_index=1):
    """Coerce common SHAP return shapes into (vec, base) for a single row."""
    v = np.array(values)
    # values -> vec
    if v.ndim == 1:
        vec = v
    elif v.ndim == 2:
        if v.shape[0] == 1 and v.shape[1] == x_row_2d.shape[1]:
            vec = v[0]
        elif v.shape[1] == x_row_2d.shape[1] and v.shape[0] > 1:
            idx = class_index if v.shape[0] > class_index else 0
            vec = v[idx, :]
        elif v.shape[1] == 1 and v.shape[0] == x_row_2d.shape[1]:
            vec = v[:, 0]
        else:
            vec = v.ravel()[: x_row_2d.shape[1]]
    elif v.ndim == 3:
        vec = v.reshape(-1)[: x_row_2d.shape[1]]
    else:
        vec = v.reshape(-1)[: x_row_2d.shape[1]]

    # base value -> base
    bv = np.array(base_values)
    if bv.ndim == 0:
        base = float(bv)
    elif bv.ndim == 1:
        base = float(bv[class_index] if bv.size > class_index else bv[0])
    elif bv.ndim == 2:
        base = float(bv[0, class_index] if bv.shape[1] > class_index else bv[0, 0])
    else:
        base = float(bv.reshape(-1)[0])

    return vec.astype(float), base

def explain_rf_instance(rf_model, x_row_2d, feature_names):
    """
    Return a SHAP Explanation-like object for a single row compatible with shap.plots.waterfall
    (makes TreeExplainer outputs robust across SHAP versions).
    """
    explainer = shap.TreeExplainer(rf_model)
    out = explainer.shap_values(x_row_2d)

    # If shap.Explanation returned
    if isinstance(out, shap.Explanation):
        values = out.values
        base_values = out.base_values
        data = out.data if out.data is not None else x_row_2d
        vec, base = _coerce_single_row_single_output(values, base_values, x_row_2d, class_index=1)
        datum = (data[0] if isinstance(data, np.ndarray) and data.shape[0] == 1 else x_row_2d[0])
        return shap.Explanation(values=vec, base_values=base, data=datum, feature_names=feature_names)

    # If list per class
    if isinstance(out, list):
        class_idx = 1 if len(out) > 1 else 0
        vals = out[class_idx]
        ev = explainer.expected_value
        base_values = ev[class_idx] if isinstance(ev, (list, tuple, np.ndarray)) and len(ev) > class_idx else ev
        vec, base = _coerce_single_row_single_output(vals, base_values, x_row_2d, class_index=0)
        return shap.Explanation(values=vec, base_values=base, data=x_row_2d[0], feature_names=feature_names)

    # fallback numeric
    vals = np.array(out)
    ev = explainer.expected_value
    vec, base = _coerce_single_row_single_output(vals, ev, x_row_2d, class_index=1)
    return shap.Explanation(values=vec, base_values=base, data=x_row_2d[0], feature_names=feature_names)

def explain_lstm_instance(lstm_model, X_scaled_all, idx, feature_names, bg_size=64, nsamples=150):
    """
    Explain single LSTM instance. Tries DeepExplainer (fast) then KernelExplainer fallback (slower).
    """
    n = X_scaled_all.shape[0]
    bg_size = int(max(10, min(bg_size, n)))
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(n, size=bg_size, replace=False) if n > bg_size else np.arange(n)
    background_2d = X_scaled_all[bg_idx]
    background_3d = background_2d.reshape((background_2d.shape[0], 1, background_2d.shape[1]))
    x0_2d = X_scaled_all[idx:idx+1]
    x0_3d = x0_2d.reshape((1, 1, X_scaled_all.shape[1]))

    try:
        import tensorflow as tf  # ensure TF present for DeepExplainer
        _ = tf.__version__
        explainer = shap.DeepExplainer(lstm_model, background_3d)
        sv_list = explainer.shap_values(x0_3d)
        sv_arr = sv_list[0] if isinstance(sv_list, list) else sv_list
        sv_vec = np.array(sv_arr).reshape(-1)[-X_scaled_all.shape[1]:]
        ev = explainer.expected_value
        base_value = float(np.array(ev).reshape(-1)[0]) if ev is not None else float(np.mean(lstm_model.predict(background_3d).reshape(background_3d.shape[0], -1)[:, 0]))
        return shap.Explanation(values=sv_vec, base_values=base_value, data=x0_2d[0], feature_names=feature_names)
    except Exception:
        # KernelExplainer fallback
        def f(x2d):
            xseq = x2d.reshape((x2d.shape[0], 1, x2d.shape[1]))
            out = lstm_model.predict(xseq, verbose=0)
            out = np.asarray(out).reshape(out.shape[0], -1)
            return out[:, 1] if out.shape[1] > 1 else out[:, 0]
        explainer = shap.KernelExplainer(f, background_2d)
        sv = explainer.shap_values(x0_2d, nsamples=min(nsamples, 2 * background_2d.shape[1] + 1))
        values = sv[0] if isinstance(sv, list) else sv
        base_value = float(np.mean(f(background_2d)))
        return shap.Explanation(values=values[0], base_values=base_value, data=x0_2d[0], feature_names=feature_names)

def shap_force_plot_html(explanation):
    try:
        obj = shap.plots.force(explanation, matplotlib=False)
        html = obj.html() if hasattr(obj, "html") else str(obj)
    except Exception:
        html = str(explanation)
    try:
        js = shap.getjs()
    except Exception:
        js = ""
    return f"<head>{js}</head><body>{html}</body>"

def st_shap(plot_html, height=360):
    components.html(plot_html, height=height)

# --------------------------
# App UI
# --------------------------
def main():
    st.title("Fraud Detection Dashboard")
    st.write("Upload a CSV, choose model, pick a row, then click Predict & Explain.")

    # Sidebar options
    model_choice = st.sidebar.selectbox("Model", ["Random Forest", "LSTM"])
    background_size = st.sidebar.slider("LSTM SHAP background size", min_value=10, max_value=300, value=64, step=8)
    nsamples_kernel = st.sidebar.slider("LSTM KernelExplainer nsamples", min_value=50, max_value=400, value=150, step=25)

    # Upload CSV
    st.header("1) Upload CSV")
    uploaded = st.file_uploader("Upload raw CSV", type=["csv"])
    if uploaded is None:
        st.info("Please upload your transactions CSV file to continue.")
        st.stop()

    try:
        df_uploaded = read_csv_robust(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Load RF artifacts (scaler + feature_columns) - RF is required
    rf_model = load_pickle(RF_MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
    feature_columns = load_json(FEATURES_PATH)

    missing = []
    if rf_model is None:
        missing.append(str(RF_MODEL_PATH))
    if scaler is None:
        missing.append(str(SCALER_PATH))
    if feature_columns is None:
        missing.append(str(FEATURES_PATH))

    if missing:
        st.error("Missing required artifacts:\n" + "\n".join(missing))
        st.stop()

    # Normalize feature_columns to list
    if isinstance(feature_columns, dict) and "feature_columns" in feature_columns:
        feature_columns = list(feature_columns["feature_columns"])
    elif isinstance(feature_columns, list):
        feature_columns = list(feature_columns)
    else:
        st.error("feature_columns.json could not be parsed into a list.")
        st.stop()

    # Align features: add missing trained features as zeros (prevents KeyError)
    df_aligned = align_features_fill_zeros(df_uploaded, feature_columns)
    st.success(f"✅ Preprocessed: {df_aligned.shape[0]} rows × {df_aligned.shape[1]} features.")
    with st.expander("Preview aligned features (first rows)"):
        st.dataframe(df_aligned.head())

    # Scale
    try:
        X_scaled = scaler.transform(df_aligned.values.astype(np.float32))
    except Exception as e:
        st.error(f"Scaler transform failed: {e}")
        st.stop()

    # Row selection
    st.header("2) Pick a row to Predict & Explain")
    row_index = st.number_input("Row index (0-based)", min_value=0, max_value=max(0, len(df_aligned)-1), value=0, step=1)

    # Lazy-load LSTM only if user chose it (so TF import is delayed)
    lstm_model = None
    lstm_error = None
    if model_choice == "LSTM":
        lstm_model, lstm_error = load_lstm_safe(LSTM_H5_PATH)
        if lstm_model is None:
            st.warning("LSTM could not be loaded. Random Forest is still available.")
            with st.expander("LSTM load error details"):
                st.code(lstm_error or "No details available.")

    # Predict & Explain button
    if st.button("🔮 Predict & Explain Selected Row"):
        # Choose execution path
        if model_choice == "Random Forest" or (model_choice == "LSTM" and lstm_model is None):
            # Use Random Forest
            preds, probs = predict_rf(rf_model, X_scaled)
            explanation = explain_rf_instance(rf_model, X_scaled[row_index:row_index+1], feature_columns)
            used_model_name = "Random Forest"
        else:
            # Use LSTM
            preds, probs = predict_lstm(lstm_model, X_scaled)
            explanation = explain_lstm_instance(lstm_model, X_scaled, row_index, feature_columns,
                                                bg_size=background_size, nsamples=nsamples_kernel)
            used_model_name = "LSTM"

        # Prediction output
        pred_label = "🚨 Fraudulent" if preds[row_index] == 1 else "✅ Legitimate"
        st.subheader("Prediction")
        st.write(f"**Model used:** {used_model_name}")
        st.write(f"**Row {row_index} Prediction:** {pred_label}")
        st.write(f"**Probability of Fraud:** {probs[row_index]:.2%}")

        # SHAP Explanation: Waterfall (fallback to bar)
        st.subheader("Explanation for this Transaction (SHAP)")
        st.markdown("**Waterfall Plot (feature contributions → prediction)**")
        try:
            num_feats = int(np.size(explanation.values))
            if num_feats == 0:
                raise ValueError("Empty SHAP values.")
            max_disp = max(1, min(12, num_feats))
            fig, ax = plt.subplots(figsize=(9, 6))
            # waterfall expects a shap.Explanation-like object
            shap.plots.waterfall(explanation, max_display=max_disp, show=False)
            st.pyplot(fig)
        except Exception as e_wf:
            st.warning(f"Waterfall plot failed ({e_wf}). Showing bar chart fallback.")
            vals = pd.Series(np.array(explanation.values).reshape(-1), index=feature_columns)
            top = vals.abs().sort_values(ascending=False).head(12)
            order = top.index.tolist()
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(order, vals.loc[order].values)
            ax.invert_yaxis()
            ax.set_xlabel("SHAP value (impact on model output)")
            ax.set_title("Top feature contributions")
            st.pyplot(fig)

        # Interactive force plot
        st.markdown("**Force Plot (interactive, colorful)**")
        try:
            html = shap_force_plot_html(explanation)
            st_shap(html, height=360)
        except Exception as e_force:
            st.warning(f"Force plot failed ({e_force}).")

        # Top feature table
        with st.expander("Top Feature Contributions (abs SHAP)"):
            vals = pd.Series(np.array(explanation.values).reshape(-1), index=feature_columns)
            top = vals.abs().sort_values(ascending=False).head(15)
            contrib = pd.DataFrame({
                "feature": top.index,
                "shap_value": vals.loc[top.index].values,
                "abs_contribution": top.values
            })
            st.dataframe(contrib)

        # Add predictions as columns to the uploaded DataFrame and offer download
        out_df = df_uploaded.copy()
        out_df["RF_Prediction"] = (predict_rf(rf_model, X_scaled)[0] if rf_model is not None else np.nan)
        if lstm_model is not None:
            out_df["LSTM_Prediction"] = predict_lstm(lstm_model, X_scaled)[0]
        # CSV download
        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")

    # End of main
    st.markdown("---")
    st.info("Notes: Place model artifacts in the repo root. If LSTM fails to load on Streamlit Cloud, check your tensorflow version in requirements.txt to match the TF used when saving the model.")

if __name__ == "__main__":
    main()
