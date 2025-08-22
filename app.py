# app.py
"""
Fraud Detection Dashboard
- Requires BOTH Random Forest and LSTM artifacts
- Loads artifacts AFTER Streamlit session is ready (prevents SessionInfo errors)
- Assumes artifacts live in the repo root:
    random_forest_tuned.pkl
    scaler.pkl
    feature_columns.json
    best_lstm.h5   (Keras H5)  OR a SavedModel folder named best_lstm/
"""

import warnings
warnings.filterwarnings("ignore")

import json
import inspect
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
# Page config (keep very top)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


# -----------------------------------------------------------------------------
# Paths (artifacts in repo root)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
RF_MODEL_PATH   = BASE_DIR / "random_forest_tuned.pkl"
SCALER_PATH     = BASE_DIR / "scaler.pkl"
FEATURES_PATH   = BASE_DIR / "feature_columns.json"
LSTM_H5_PATH    = BASE_DIR / "best_lstm.h5"      # H5 file OR
LSTM_DIR_PATH   = BASE_DIR / "best_lstm"         # SavedModel directory (either works)


# -----------------------------------------------------------------------------
# Cached loaders (executed ONLY when called inside main())
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
    """Generous map to survive H5 differences / Lambda / version mismatches."""
    return {
        # Activations
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
        # Layers
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
        # Allow tf.* inside Lambda layers
        "tf": tf,
    }

def _supports_safe_mode():
    """Keras 3 has a 'safe_mode' kwarg; older Keras doesn't."""
    try:
        return "safe_mode" in inspect.signature(tf.keras.models.load_model).parameters
    except Exception:
        return False

@st.cache_resource(show_spinner=True)
def load_lstm_strict(h5_path: Path, dir_path: Path):
    """
    Load an LSTM model robustly:
      1) If SavedModel folder exists, load it first (most stable across versions)
      2) Else load H5 with multiple fallbacks:
         - compile=False
         - + custom_objects
         - + safe_mode=False (Keras 3)
    Returns: (model, debug_text) on success OR (None, error_text) on failure.
    """
    debug_lines = [f"TensorFlow: {tf.__version__}", f"Keras safe_mode support: {_supports_safe_mode()}"]

    # Prefer SavedModel directory if present
    if dir_path.exists() and dir_path.is_dir():
        try:
            m = tf.keras.models.load_model(str(dir_path), compile=False)
            debug_lines.append(f"Loaded SavedModel from: {dir_path}")
            return m, "\n".join(debug_lines)
        except Exception as e_dir:
            debug_lines.append(f"SavedModel load failed: {repr(e_dir)}")

    # Otherwise, require the .h5 file
    if not h5_path.exists():
        return None, "\n".join(debug_lines + [f"Missing LSTM model file: {h5_path}"])

    # Quick check for Git LFS pointer file (common Streamlit Cloud issue)
    try:
        if h5_path.stat().st_size < 1024:  # tiny files are suspicious
            head = h5_path.read_text(errors="ignore")[:200]
            if "git-lfs" in head.lower() or "oid sha256" in head.lower():
                debug_lines.append("Detected Git LFS pointer instead of real H5. Ensure Git LFS is enabled and the file is fetched.")
                return None, "\n".join(debug_lines)
    except Exception:
        pass

    # Try 1: simplest H5 load
    try:
        m = tf.keras.models.load_model(str(h5_path), compile=False)
        debug_lines.append(f"Loaded H5 without custom_objects: {h5_path.name}")
        return m, "\n".join(debug_lines)
    except Exception as e1:
        debug_lines.append(f"H5 load (no custom_objects) failed: {repr(e1)}")

    # Try 2: with generous custom_objects
    try:
        m = tf.keras.models.load_model(str(h5_path), compile=False, custom_objects=_keras_custom_objects())
        debug_lines.append(f"Loaded H5 with custom_objects: {h5_path.name}")
        return m, "\n".join(debug_lines)
    except Exception as e2:
        debug_lines.append(f"H5 load (custom_objects) failed: {repr(e2)}")

    # Try 3: Keras 3 safe_mode=False (allows Lambda/custom deserialization)
    if _supports_safe_mode():
        try:
            m = tf.keras.models.load_model(
                str(h5_path),
                compile=False,
                custom_objects=_keras_custom_objects(),
                safe_mode=False,
            )
            debug_lines.append(f"Loaded H5 with custom_objects + safe_mode=False: {h5_path.name}")
            return m, "\n".join(debug_lines)
        except Exception as e3:
            debug_lines.append(f"H5 load (custom_objects + safe_mode=False) failed: {repr(e3)}")

    # All attempts failed
    return None, "\n".join(debug_lines)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def st_shap(plot_html, height=320):
    """Embed SHAP HTML safely in Streamlit."""
    components.html(plot_html, height=height)

def safe_str(x):
    try:
        return str(x)
    except Exception:
        return "Other"

def extract_categorical_bases_from_features(features):
    return sorted(list({c.split("_")[0] for c in features if "_" in c}))

def preprocess_uploaded(df_raw, features, cardinality_threshold=100):
    """Make uploaded raw dataframe align with training feature set."""
    log = {"dropped_high_cardinality": [], "encoded": [], "added_missing": [], "dropped_extra": []}
    df = df_raw.copy()
    categorical_bases = extract_categorical_bases_from_features(features)

    # Drop super high-cardinality string cols (not explicitly in one-hot bases)
    for c in list(df.columns):
        if df[c].dtype == "object" and c not in categorical_bases:
            nunique = df[c].nunique(dropna=True)
            if nunique > cardinality_threshold:
                df.drop(columns=[c], inplace=True)
                log["dropped_high_cardinality"].append((c, nunique))

    # One-hot encode known categorical bases to match feature columns
    for cat in categorical_bases:
        if cat in df.columns:
            df[cat] = df[cat].apply(safe_str).fillna("Other")
            dummies = pd.get_dummies(df[cat], prefix=cat, drop_first=False)
            dummies = dummies[[c for c in dummies.columns if c in features]]  # keep only trained dummies
            df = pd.concat([df.drop(columns=[cat]), dummies], axis=1)
            log["encoded"].append(cat)

    # Add any missing trained features as zeros
    for col in features:
        if col not in df.columns:
            df[col] = 0
            log["added_missing"].append(col)

    # Drop any extra cols not used by the model
    extras = [c for c in df.columns if c not in features]
    if extras:
        df = df.drop(columns=extras)
        log["dropped_extra"].extend(extras)

    # Order columns, coerce to numbers
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
# SHAP explainers (robust to SHAP version output shapes)
# -----------------------------------------------------------------------------
def _coerce_single_row_single_output(values, base_values, x_row_2d, class_index=1):
    v = np.array(values)
    if v.ndim == 1:
        vec = v
    elif v.ndim == 2:
        if v.shape[0] == 1 and v.shape[1] == x_row_2d.shape[1]:
            vec = v[0]
        elif v.shape[1] == 1 and v.shape[0] == x_row_2d.shape[1]:
            vec = v[:, 0]
        elif v.shape[1] == x_row_2d.shape[1]:
            idx = class_index if v.shape[0] > 1 else 0
            vec = v[idx, :]
        elif v.shape[0] == x_row_2d.shape[1]:
            idx = class_index if v.shape[1] > 1 else 0
            vec = v[:, idx]
        else:
            vec = np.squeeze(v)
            if vec.ndim != 1:
                vec = vec.ravel()
    elif v.ndim == 3:
        if v.shape[0] == 1 and v.shape[1] == x_row_2d.shape[1]:
            idx = class_index if v.shape[2] > 1 else 0
            vec = v[0, :, idx]
        elif v.shape[0] == 1 and v.shape[2] == x_row_2d.shape[1]:
            idx = class_index if v.shape[1] > 1 else 0
            vec = v[0, idx, :]
        else:
            vec = v.reshape(-1)[:x_row_2d.shape[1]]
    else:
        vec = v.reshape(-1)[:x_row_2d.shape[1]]

    bv = np.array(base_values)
    if bv.ndim == 0:
        base = float(bv)
    elif bv.ndim == 1:
        base = float(bv[class_index] if bv.size > 1 else bv[0])
    elif bv.ndim == 2:
        base = float(bv[0, class_index] if bv.shape[1] > 1 else bv[0, 0])
    else:
        base = float(bv.reshape(-1)[0])

    return vec.astype(float), base

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
        base_values = ev[class_idx] if isinstance(ev, list) else ev
        vec, base = _coerce_single_row_single_output(vals, base_values, x_row_2d, class_index=0)
        return shap.Explanation(values=vec, base_values=base, data=x_row_2d[0], feature_names=feature_names)

    vals = np.array(out)
    ev = explainer.expected_value
    vec, base = _coerce_single_row_single_output(vals, ev, x_row_2d, class_index=1)
    return shap.Explanation(values=vec, base_values=base, data=x_row_2d[0], feature_names=feature_names)

def explain_lstm_instance(lstm_model, X_scaled_all, idx, feature_names, bg_size=64, nsamples=150):
    """
    Try DeepExplainer first; if it fails (e.g., Keras/TF/SHAP mismatch), fallback to KernelExplainer.
    Always return a well-formed shap.Explanation for a single row.
    """
    n = X_scaled_all.shape[0]
    bg_size = int(max(10, min(bg_size, n)))
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(n, size=bg_size, replace=False) if n > bg_size else np.arange(n)
    background_2d = X_scaled_all[bg_idx]
    background_3d = background_2d.reshape((background_2d.shape[0], 1, background_2d.shape[1]))

    x0_2d = X_scaled_all[idx:idx+1]
    x0_3d = x0_2d.reshape((1, 1, X_scaled_all.shape[1]))

    # Try DeepExplainer (works when SHAP has TF graph hooks)
    try:
        explainer = shap.DeepExplainer(lstm_model, background_3d)
        sv_list = explainer.shap_values(x0_3d)
        sv_arr = sv_list[0] if isinstance(sv_list, list) else sv_list
        sv_vec = np.array(sv_arr).reshape(-1)[-X_scaled_all.shape[1]:]
        ev = explainer.expected_value
        base_value = float(np.array(ev).reshape(-1)[0])
        return shap.Explanation(values=sv_vec, base_values=base_value, data=x0_2d[0], feature_names=feature_names)
    except Exception:
        # Fallback: model-agnostic KernelExplainer
        def f(x2d):
            xseq = x2d.reshape((x2d.shape[0], 1, x2d.shape[1]))
            out = lstm_model.predict(xseq, verbose=0)
            out = np.asarray(out).reshape(out.shape[0], -1)
            return out[:, 1] if out.shape[1] > 1 else out[:, 0]

        explainer = shap.KernelExplainer(f, background_2d)
        sv = explainer.shap_values(x0_2d, nsamples=min(nsamples, 2 * background_2d.shape[1] + 1))
        values = sv[0] if isinstance(sv, list) else sv
        base_value = float(np.mean(f(background_2d)))
        vec, base = _coerce_single_row_single_output(values, base_value, x0_2d, class_index=0)
        return shap.Explanation(values=vec, base_values=base, data=x0_2d[0], feature_names=feature_names)

def shap_force_plot_html(explanation):
    try:
        obj = shap.plots.force(explanation, matplotlib=False)
        html = obj.html()
        js = shap.getjs()
        return f"<head>{js}</head><body>{html}</body>"
    except Exception:
        return "<p>Force plot not available in this environment.</p>"


# --- Robust CSV reader --------------------------------------------------------
def read_csv_robust(uploaded_file):
    """Tries a few encodings/separators so uploads don't fail silently."""
    # Try pandas default first
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        pass

    # Try utf-8-sig + common delimiters
    uploaded_file.seek(0)
    for sep in [",", ";", "\t", "|"]:
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, sep=sep, encoding=enc)
            except Exception:
                continue

    # If all else fails, raise the original error
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)  # will throw


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
def main():
    st.title("Fraud Detection Dashboard")

    # Sidebar
    st.sidebar.header("Model & Options")
    model_choice = st.sidebar.selectbox("Model", ["Random Forest", "LSTM"], index=0)
    background_size = st.sidebar.slider(
        "Background sample for SHAP (LSTM explainer)",
        min_value=10, max_value=300, value=64, step=8
    )
    nsamples_kernel = st.sidebar.slider(
        "SHAP nsamples (KernelExplainer fallback)",
        min_value=50, max_value=400, value=150, step=25
    )

    # 1) Upload CSV
    st.header("1) Upload CSV")
    uploaded = st.file_uploader("Upload a raw CSV", type=["csv"])
    if uploaded is None:
        st.info("Tip: Upload your raw transactions CSV.")
        st.stop()

    try:
        df_uploaded = read_csv_robust(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # 2) Load REQUIRED artifacts (AFTER Streamlit session is ready)
    rf_model = load_pickle(RF_MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
    feature_columns = load_json(FEATURES_PATH)

    # Parse feature columns
    if isinstance(feature_columns, dict) and "feature_columns" in feature_columns:
        feature_columns = list(feature_columns["feature_columns"])
    elif isinstance(feature_columns, list):
        feature_columns = list(feature_columns)
    else:
        st.error("feature_columns.json could not be parsed into a list.")
        st.stop()

    # Load LSTM strictly (required)
    lstm_model, lstm_debug = load_lstm_strict(LSTM_H5_PATH, LSTM_DIR_PATH)
    if lstm_model is None:
        st.error("❌ Failed to load the LSTM model. See details below.")
        with st.expander("Show LSTM load diagnostics"):
            st.code(lstm_debug or "No details available.")
        st.stop()

    # 3) Preprocess & scale
    df_features, prep_log = preprocess_uploaded(df_uploaded, feature_columns)
    st.success(f"✅ Preprocessed: {df_features.shape[0]} rows × {df_features.shape[1]} features.")
    with st.expander("Preprocessing log"):
        st.json(prep_log)

    try:
        X_scaled = scaler.transform(df_features.values.astype(np.float32))
    except Exception as e:
        st.error(f"Scaler failed: {e}")
        st.stop()

    # 4) Predict & Explain
    st.header("2) Pick a row to Predict & Explain")
    row_index = st.number_input(
        "Row index (0-based)",
        min_value=0, max_value=max(0, len(df_features)-1), value=0, step=1
    )

    if st.button("🔮 Predict & Explain Selected Row"):
        if model_choice == "Random Forest":
            preds, probs = predict_rf(rf_model, X_scaled)
            explanation = explain_rf_instance(rf_model, X_scaled[row_index:row_index+1], feature_columns)
        else:  # LSTM
            preds, probs = predict_lstm(lstm_model, X_scaled)
            explanation = explain_lstm_instance(
                lstm_model, X_scaled, row_index, feature_columns,
                bg_size=background_size, nsamples=nsamples_kernel
            )

        pred_label = "🚨 Fraudulent" if preds[row_index] == 1 else "✅ Legitimate"
        st.subheader("Prediction")
        st.write(f"**Row {row_index} Prediction:** {pred_label}")
        st.write(f"**Probability of Fraud:** {probs[row_index]:.2%}")

        st.subheader("Explanation for this Transaction (SHAP)")

        # Waterfall with safeguards & fallback
        st.markdown("**Waterfall Plot (feature contributions → prediction)**")
        try:
            num_feats = int(np.size(explanation.values))
            if num_feats == 0:
                raise ValueError("Empty SHAP values.")
            max_disp = max(1, min(12, num_feats))
            fig, _ = plt.subplots(figsize=(9, 6))
            shap.plots.waterfall(explanation, max_display=max_disp, show=False)
            st.pyplot(fig, clear_figure=True)
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
            st.pyplot(fig, clear_figure=True)

        # Interactive force plot (best effort)
        st.markdown("**Force Plot (interactive)**")
        html = shap_force_plot_html(explanation)
        st_shap(html, height=320)

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


if __name__ == "__main__":
    main()
