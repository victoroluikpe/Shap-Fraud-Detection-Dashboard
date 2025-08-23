# app.py
"""
Fraud Detection Dashboard
- No demo dataset (CSV upload required)
- Both models exposed as choices
- LSTM loading made robust with multiple strategies + good error reporting
- Optional model artifact uploaders (useful on Streamlit Cloud)
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
# Keras custom objects helper (used only when loading .h5)
# ---------------------------------------------------------------------
def _keras_custom_objects(tf):
    # generous mapping for activations/layers often present in custom models
    co = {
        "swish": getattr(tf.keras.activations, "swish", getattr(tf.nn, "swish", None)),
        "gelu": getattr(tf.keras.activations, "gelu", getattr(tf.nn, "gelu", None)),
        "leaky_relu": getattr(tf.nn, "leaky_relu", None),
        "LeakyReLU": getattr(tf.keras.layers, "LeakyReLU", None),
        "PReLU": getattr(tf.keras.layers, "PReLU", None),
        "ELU": getattr(tf.keras.layers, "ELU", None),
        "LayerNormalization": getattr(tf.keras.layers, "LayerNormalization", None),
        "BatchNormalization": getattr(tf.keras.layers, "BatchNormalization", None),
        "Dense": getattr(tf.keras.layers, "Dense", None),
        "Dropout": getattr(tf.keras.layers, "Dropout", None),
        "LSTM": getattr(tf.keras.layers, "LSTM", None),
        "GRU": getattr(tf.keras.layers, "GRU", None),
        # make tf available inside Lambda layers if used
        "tf": tf,
    }
    return {k: v for k, v in co.items() if v is not None}

# ---------------------------------------------------------------------
# Robust LSTM loader - returns (model_or_None, error_message_or_None, tf_version_or_None)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_lstm_safe_from_path(path: Optional[Path]) -> Tuple[Optional[object], Optional[str], Optional[str]]:
    if path is None or not Path(path).exists():
        return None, "LSTM file not found at path.", None
    try:
        import tensorflow as tf  # lazy import
    except Exception as e_imp:
        return None, f"TensorFlow import failed: {repr(e_imp)}", None

    tf_version = getattr(tf, "__version__", "unknown")
    # Try load_model with compile=False
    try:
        m = tf.keras.models.load_model(str(path), compile=False)
        return m, None, tf_version
    except Exception as e1:
        # try again with generous custom_objects
        try:
            m = tf.keras.models.load_model(str(path), compile=False, custom_objects=_keras_custom_objects(tf))
            return m, None, tf_version
        except Exception as e2:
            # Last-resort: include both errors for diagnostics
            err = f"LSTM load failed.\nFirst error: {repr(e1)}\nSecond (with custom_objects): {repr(e2)}"
            return None, err, tf_version

# Variant: load from an uploaded file-like object bytes (Streamlit uploader)
def load_lstm_safe_from_bytes(b: bytes) -> Tuple[Optional[object], Optional[str], Optional[str]]:
    # write to a temp file and call the path loader
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tmp.write(b)
        tmp.flush()
        tmp.close()
        return load_lstm_safe_from_path(Path(tmp.name))
    finally:
        # do not delete immediately: cache_resource may refer to file while streaming;
        # system will clean tmp files eventually. If you prefer immediate cleanup,
        # remove the file here when safe.
        pass

# ---------------------------------------------------------------------
# Utilities: CSV reading, preprocessing, predictions, plotting
# ---------------------------------------------------------------------
def read_csv_robust(uploaded_file):
    """Try several encodings/separators to robustly read CSV uploads."""
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
    # allow pandas to raise the last error
    return pd.read_csv(uploaded_file)

def safe_str(x):
    try:
        return str(x)
    except Exception:
        return "Other"

def extract_categorical_bases_from_features(features):
    bases = [c.split("_")[0] for c in features if "_" in c]
    return sorted(list(set(bases)))

def preprocess_uploaded(df_raw, features, cardinality_threshold=100):
    """
    Align uploaded dataframe to the trained feature set (one-hot).
    Returns (df_aligned, log)
    """
    log = {"dropped_high_cardinality": [], "encoded": [], "added_missing": [], "dropped_extra": []}
    df = df_raw.copy()
    categorical_bases = extract_categorical_bases_from_features(features)

    for c in list(df.columns):
        if df[c].dtype == "object" and c not in categorical_bases:
            nunique = int(df[c].nunique(dropna=True))
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
            df[col] = 0.0
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
    # reshape to (n_samples, 1, n_features)
    X_seq = X_scaled_2d.reshape((X_scaled_2d.shape[0], 1, X_scaled_2d.shape[1]))
    out = lstm_model.predict(X_seq, verbose=0)
    out = np.asarray(out).reshape(out.shape[0], -1)
    probs = out[:, 1] if out.shape[1] > 1 else out[:, 0]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

# ---------------------------------------------------------------------
# SHAP robustness helpers (adapted to many SHAP output shapes)
# ---------------------------------------------------------------------
def _coerce_single_row_single_output(values, base_values, x_row_2d, class_index=1):
    v = np.array(values)
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

def get_rf_explainer(rf_model):
    if "rf_explainer" not in st.session_state:
        try:
            st.session_state["rf_explainer"] = shap.TreeExplainer(rf_model, model_output="probability",
                                                                 feature_perturbation="interventional")
        except Exception:
            st.session_state["rf_explainer"] = shap.TreeExplainer(rf_model)
    return st.session_state["rf_explainer"]

def explain_rf_instance(rf_model, x_row_2d, feature_names):
    explainer = get_rf_explainer(rf_model)
    out = explainer.shap_values(x_row_2d)
    if isinstance(out, shap.Explanation):
        values = out.values
        base_values = out.base_values
        data = out.data if out.data is not None else x_row_2d
        vec, base = _coerce_single_row_single_output(values, base_values, x_row_2d, class_index=1)
        datum = (data[0] if isinstance(data, np.ndarray) and data.shape[0] == 1 else x_row_2d[0])
        return shap.Explanation(values=vec, base_values=base, data=datum, feature_names=feature_names)
    if isinstance(out, list):
        class_idx = 1 if len(out) > 1 else 0
        vals = out[class_idx]
        ev = explainer.expected_value
        base_values = ev[class_idx] if isinstance(ev, (list, tuple, np.ndarray)) and len(ev) > class_idx else ev
        vec, base = _coerce_single_row_single_output(vals, base_values, x_row_2d, class_index=0)
        return shap.Explanation(values=vec, base_values=base, data=x_row_2d[0], feature_names=feature_names)
    vals = np.array(out)
    ev = explainer.expected_value
    vec, base = _coerce_single_row_single_output(vals, ev, x_row_2d, class_index=1)
    return shap.Explanation(values=vec, base_values=base, data=x_row_2d[0], feature_names=feature_names)

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
        import tensorflow as tf
        _ = tf.__version__
        explainer = shap.DeepExplainer(lstm_model, background_3d)
        sv_list = explainer.shap_values(x0_3d)
        sv_arr = sv_list[0] if isinstance(sv_list, list) else sv_list
        sv_vec = np.array(sv_arr).reshape(-1)[-X_scaled_all.shape[1]:]
        ev = explainer.expected_value
        base_value = float(np.array(ev).reshape(-1)[0]) if ev is not None else float(np.mean(lstm_model.predict(background_3d).reshape(background_3d.shape[0], -1)[:, 0]))
        return shap.Explanation(values=sv_vec, base_values=base_value, data=x0_2d[0], feature_names=feature_names)
    except Exception:
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

# ---------------------------------------------------------------------
# Main app UI
# ---------------------------------------------------------------------
def main():
    st.title("Fraud Detection Dashboard")

    st.sidebar.header("Model & SHAP options")
    model_choice = st.sidebar.selectbox("Choose model", ["Random Forest", "LSTM"], index=0)
    background_size = st.sidebar.slider("Background sample for SHAP (LSTM DeepExplainer)", 10, 300, 64, 8)
    nsamples_kernel = st.sidebar.slider("SHAP nsamples (Kernel fallback)", 50, 400, 150, 25)

    st.sidebar.markdown("### Optional: Upload artifacts (useful on Streamlit Cloud)")
    allow_upload = st.sidebar.checkbox("Upload model artifacts manually", value=False)

    uploaded_rf = uploaded_scaler = uploaded_features = uploaded_lstm = None
    if allow_upload:
        st.sidebar.write("Upload in this order: RandomForest (.pkl), Scaler (.pkl), feature_columns (.json), LSTM (.h5)")
        uploaded_rf = st.sidebar.file_uploader("random_forest_tuned.pkl", type=["pkl", "joblib"])
        uploaded_scaler = st.sidebar.file_uploader("scaler.pkl", type=["pkl", "joblib"])
        uploaded_features = st.sidebar.file_uploader("feature_columns.json", type=["json"])
        uploaded_lstm = st.sidebar.file_uploader("best_lstm.h5", type=["h5"])

    st.header("1) Upload CSV (required)")
    uploaded = st.file_uploader("Upload your raw transactions CSV", type=["csv"])
    if uploaded is None:
        st.info("Please upload a CSV to continue.")
        st.stop()

    try:
        df_uploaded = read_csv_robust(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()

    # If user uploaded artifacts, save temporarily and point loaders at them.
    # Note: we do not permanently overwrite repo files; these are stored in temp files.
    artifact_paths = {}
    if allow_upload:
        if uploaded_rf is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            tmp.write(uploaded_rf.getvalue())
            tmp.flush()
            tmp.close()
            artifact_paths["rf"] = Path(tmp.name)
        if uploaded_scaler is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            tmp.write(uploaded_scaler.getvalue())
            tmp.flush()
            tmp.close()
            artifact_paths["scaler"] = Path(tmp.name)
        if uploaded_features is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            tmp.write(uploaded_features.getvalue())
            tmp.flush()
            tmp.close()
            artifact_paths["features"] = Path(tmp.name)
        if uploaded_lstm is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
            tmp.write(uploaded_lstm.getvalue())
            tmp.flush()
            tmp.close()
            artifact_paths["lstm"] = Path(tmp.name)

    # Load RF, scaler, features (prefer uploaded if provided)
    rf_path = artifact_paths.get("rf", RF_MODEL_PATH)
    scaler_path = artifact_paths.get("scaler", SCALER_PATH)
    features_path = artifact_paths.get("features", FEATURES_PATH)
    lstm_path = artifact_paths.get("lstm", LSTM_H5_PATH)

    with st.spinner("Loading Random Forest + scaler + feature columns..."):
        rf_model = load_pickle(rf_path)
        scaler = load_pickle(scaler_path)
        feature_columns = load_json(features_path)

    missing = []
    if rf_model is None:
        missing.append(str(rf_path))
    if scaler is None:
        missing.append(str(scaler_path))
    if feature_columns is None:
        missing.append(str(features_path))
    if missing:
        st.error("Missing required artifacts (RF, scaler, feature_columns). Either place them in repo root or upload via sidebar.")
        st.write("Missing paths:\n" + "\n".join(missing))
        st.stop()

    # Normalize feature_columns variable
    if isinstance(feature_columns, dict) and "feature_columns" in feature_columns:
        feature_columns = list(feature_columns["feature_columns"])
    elif isinstance(feature_columns, list):
        feature_columns = list(feature_columns)
    else:
        st.error("feature_columns.json could not be parsed into a list.")
        st.stop()

    # Preprocess and scale
    df_features, prep_log = preprocess_uploaded(df_uploaded, feature_columns)
    st.success(f"✅ Preprocessed: {df_features.shape[0]} rows × {df_features.shape[1]} features.")
    with st.expander("Preprocessing log"):
        st.json(prep_log)

    try:
        X_scaled = scaler.transform(df_features.values.astype(np.float32))
    except Exception as e:
        st.error(f"Scaler transform failed: {e}")
        st.stop()

    st.header("2) Pick a row to Predict & Explain")
    row_index = st.number_input("Row index (0-based)", min_value=0, max_value=max(0, len(df_features)-1), value=0, step=1)

    # Load LSTM lazily if chosen
    lstm_model = None
    lstm_error = None
    lstm_tf_version = None
    if model_choice == "LSTM":
        with st.spinner("Attempting to load LSTM (this may take a few seconds)..."):
            # If user uploaded .h5 file bytes, we already wrote to temp and assigned lstm_path accordingly.
            lstm_model, lstm_error, lstm_tf_version = load_lstm_safe_from_path(lstm_path)
        if lstm_model is None:
            st.error("Could not load LSTM model from provided file/path.")
            with st.expander("LSTM load diagnostics"):
                st.write(f"Attempted path: {lstm_path}")
                st.code(str(lstm_error))
                st.write(f"Detected TensorFlow version: {lstm_tf_version}")
            st.info("You can either upload a working `best_lstm.h5` in the sidebar (enable 'Upload model artifacts manually') or switch to Random Forest.")

    # Predict/Explain
    if st.button("🔮 Predict & Explain Selected Row"):
        if model_choice == "Random Forest" or (model_choice == "LSTM" and lstm_model is None):
            preds, probs = predict_rf(rf_model, X_scaled)
            explanation = explain_rf_instance(rf_model, X_scaled[row_index:row_index+1], feature_columns)
            used_model_name = "Random Forest"
        else:
            preds, probs = predict_lstm(lstm_model, X_scaled)
            explanation = explain_lstm_instance(lstm_model, X_scaled, row_index,
                                                feature_columns, bg_size=background_size, nsamples=nsamples_kernel)
            used_model_name = f"LSTM (TF {lstm_tf_version or 'unknown'})"

        pred_label = "🚨 Fraudulent" if preds[row_index] == 1 else "✅ Legitimate"
        st.subheader("Prediction")
        st.write(f"**Model used:** {used_model_name}")
        st.write(f"**Row {row_index} Prediction:** {pred_label}")
        st.write(f"**Probability of Fraud:** {probs[row_index]:.2%}")

        st.subheader("Explanation for this Transaction (SHAP)")
        st.markdown("**Waterfall Plot (feature contributions → prediction)**")
        try:
            num_feats = int(np.size(explanation.values))
            if num_feats == 0:
                raise ValueError("Empty SHAP values.")
            max_disp = max(1, min(12, num_feats))
            fig, ax = plt.subplots(figsize=(9, 6))
            shap.plots.waterfall(explanation, max_display=max_disp, show=False)
            st.pyplot(fig, clear_figure=True)
        except Exception as e_wf:
            st.warning(f"Waterfall plot failed ({e_wf}). Falling back to bar chart.")
            vals = pd.Series(np.array(explanation.values).reshape(-1), index=feature_columns)
            top = vals.abs().sort_values(ascending=False).head(12)
            order = top.index.tolist()
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(order, vals.loc[order].values)
            ax.invert_yaxis()
            ax.set_xlabel("SHAP value (impact on model output)")
            ax.set_title("Top feature contributions")
            st.pyplot(fig, clear_figure=True)

        st.markdown("**Force Plot (interactive)**")
        try:
            html = shap_force_plot_html(explanation)
            components.html(html, height=360)
        except Exception as e_force:
            st.warning(f"Force plot failed ({e_force}).")

        with st.expander("Top Feature Contributions (abs SHAP)"):
            vals = pd.Series(np.array(explanation.values).reshape(-1), index=feature_columns)
            top = vals.abs().sort_values(ascending=False).head(15)
            contrib = pd.DataFrame({
                "feature": top.index,
                "shap_value": vals.loc[top.index].values,
                "abs_contribution": top.values
            })
            st.dataframe(contrib)

        st.markdown("---")
        st.success("Done. If LSTM fails on Streamlit Cloud, try uploading `best_lstm.h5` via the sidebar upload widget or confirm the TF version in your `requirements.txt` matches your local TF.")

if __name__ == "__main__":
    main()
