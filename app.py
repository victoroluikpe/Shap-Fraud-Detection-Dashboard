# app.py
# Fraud Detection with RF, LSTM & SHAP
# - RF (scikit-learn) and LSTM (TensorFlow/Keras) are isolated
# - TensorFlow is imported only if/when LSTM is selected
# - Required artifacts:
#     random_forest_tuned.pkl
#     scaler.pkl
#     feature_columns.json
#     best_lstm.h5   (optional; needed only for LSTM)

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
# Streamlit page config (must be the first Streamlit call)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Fraud Detection with RF, LSTM & SHAP", layout="wide")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
RF_MODEL_PATH   = BASE_DIR / "random_forest_tuned.pkl"
SCALER_PATH     = BASE_DIR / "scaler.pkl"
FEATURES_PATH   = BASE_DIR / "feature_columns.json"
LSTM_H5_PATH    = BASE_DIR / "best_lstm.h5"  # optional

# -----------------------------------------------------------------------------
# Cached loaders (models/config)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pickle(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {path.name}: {e}")
        return None

@st.cache_resource(show_spinner=True)
def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        return None

def _keras_custom_objects(tf):
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

def load_lstm_model(h5_path: Path):
    """Lazy import TensorFlow and load LSTM. Returns (model, error_str)."""
    if not h5_path.exists():
        return None, None
    try:
        import tensorflow as tf  # lazy import (isolated)
    except Exception as e_imp:
        return None, f"TensorFlow import failed: {repr(e_imp)}"

    # Try simplest load
    try:
        m = tf.keras.models.load_model(str(h5_path), compile=False)
        return m, None
    except Exception as e1:
        # Try with generous custom_objects
        try:
            m = tf.keras.models.load_model(
                str(h5_path), compile=False, custom_objects=_keras_custom_objects(tf)
            )
            return m, None
        except Exception as e2:
            return None, f"LSTM load failed.\nFirst error: {repr(e1)}\nSecond error (with custom_objects): {repr(e2)}"

# -----------------------------------------------------------------------------
# Robust CSV reader
# -----------------------------------------------------------------------------
def read_csv_robust(uploaded_file):
    """Try multiple encodings/separators to avoid silent failures."""
    # default
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        pass
    # variants
    uploaded_file.seek(0)
    for sep in [",", ";", "\t", "|"]:
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, sep=sep, encoding=enc)
            except Exception:
                continue
    # raise original
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)

# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
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
    Align uploaded raw dataframe to training features:
    - Drop ultra-high-cardinality unknown text columns
    - One-hot known categorical bases; keep only trained dummies
    - Add missing features as 0; drop extras
    """
    log = {"dropped_high_cardinality": [], "encoded": [], "added_missing": [], "dropped_extra": []}
    df = df_raw.copy()
    categorical_bases = extract_categorical_bases_from_features(features)

    # Drop very high-cardinality text columns not in known bases
    for c in list(df.columns):
        if df[c].dtype == "object" and c not in categorical_bases:
            nunique = df[c].nunique(dropna=True)
            if nunique > cardinality_threshold:
                df.drop(columns=[c], inplace=True)
                log["dropped_high_cardinality"].append((c, nunique))

    # One-hot known bases; keep only trained dummy columns
    for cat in categorical_bases:
        if cat in df.columns:
            df[cat] = df[cat].apply(safe_str).fillna("Other")
            dummies = pd.get_dummies(df[cat], prefix=cat, drop_first=False)
            dummies = dummies[[c for c in dummies.columns if c in features]]
            df = pd.concat([df.drop(columns=[cat]), dummies], axis=1)
            log["encoded"].append(cat)

    # Add missing trained features as zeros
    for col in features:
        if col not in df.columns:
            df[col] = 0
            log["added_missing"].append(col)

    # Drop unseen extras
    extras = [c for c in df.columns if c not in features]
    if extras:
        df = df.drop(columns=extras)
        log["dropped_extra"].extend(extras)

    # Order & numeric
    df = df[features]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df, log

# -----------------------------------------------------------------------------
# Prediction pipelines (independent)
# -----------------------------------------------------------------------------
def predict_rf(rf_model, X_scaled):
    probs = rf_model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def predict_lstm(lstm_model, X_scaled_2d):
    # Treat each row as a 1-timestep sequence of all features
    X_seq = X_scaled_2d.reshape((X_scaled_2d.shape[0], 1, X_scaled_2d.shape[1]))
    probs = lstm_model.predict(X_seq, verbose=0)
    probs = np.asarray(probs).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    return preds, probs

# -----------------------------------------------------------------------------
# SHAP helpers (handle version differences)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def get_rf_explainer(rf_model):
    try:
        return shap.TreeExplainer(
            rf_model, model_output="probability", feature_perturbation="interventional"
        )
    except Exception:
        return shap.TreeExplainer(rf_model)

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

    if isinstance(out, list):  # typical for classification
        class_idx = 1 if len(out) > 1 else 0
        vals = out[class_idx]
        ev = explainer.expected_value
        base_values = ev[class_idx] if isinstance(ev, list) else ev
        vec, base = _coerce_single_row_single_output(vals, base_values, x_row_2d, class_index=0)
        return shap.Explanation(values=vec, base_values=base, data=x_row_2d[0], feature_names=feature_names)

    # ndarray fallback
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

    # Prefer DeepExplainer if TF is available (works well for Keras models)
    try:
        import tensorflow as tf  # ensure TF present for DeepExplainer path
        _ = tf.__version__
        explainer = shap.DeepExplainer(lstm_model, background_3d)
        sv_list = explainer.shap_values(x0_3d)
        sv_arr = sv_list[0] if isinstance(sv_list, list) else sv_list
        sv_vec = np.array(sv_arr).reshape(-1)[-X_scaled_all.shape[1]:]
        ev = explainer.expected_value
        base_value = float(np.array(ev).reshape(-1)[0])
        return shap.Explanation(values=sv_vec, base_values=base_value, data=x0_2d[0], feature_names=feature_names)
    except Exception:
        # KernelExplainer fallback (model-agnostic and slower)
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
    obj = shap.plots.force(explanation, matplotlib=False)
    try:
        html = obj.html()
    except Exception:
        html = str(obj)
    try:
        js = shap.getjs()
    except Exception:
        js = ""
    return f"<head>{js}</head><body>{html}</body>"

def render_shap_plots(explanation, top_k=12):
    """Render waterfall (with fallback) + force + top features table."""
    st.subheader("Explanation for this Transaction (SHAP)")

    # Waterfall with safe fallback
    st.markdown("**Waterfall Plot (feature contributions → prediction)**")
    try:
        num_feats = int(np.size(explanation.values))
        if num_feats == 0:
            raise ValueError("Empty SHAP values.")
        max_disp = max(1, min(top_k, num_feats))
        fig, _ = plt.subplots(figsize=(9, 6))
        shap.plots.waterfall(explanation, max_display=max_disp, show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception as e_wf:
        st.warning(f"Waterfall plot failed ({e_wf}). Showing bar chart fallback.")
        vals = pd.Series(np.array(explanation.values).reshape(-1), index=explanation.feature_names)
        top = vals.abs().sort_values(ascending=False).head(top_k)
        order = top.index.tolist()
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(order, vals.loc[order].values)
        ax.invert_yaxis()
        ax.set_xlabel("SHAP value (impact on model output)")
        ax.set_title("Top feature contributions")
        st.pyplot(fig, clear_figure=True)

    # Interactive force plot
    st.markdown("**Force Plot (interactive)**")
    try:
        html = shap_force_plot_html(explanation)
        components.html(html, height=320)
    except Exception as e_force:
        st.warning(f"Force plot failed ({e_force}).")

    # Top features table
    with st.expander("Top Feature Contributions (abs SHAP)"):
        vals = pd.Series(np.array(explanation.values).reshape(-1), index=explanation.feature_names)
        top = vals.abs().sort_values(ascending=False).head(15)
        contrib = pd.DataFrame({
            "feature": top.index,
            "shap_value": vals.loc[top.index].values,
            "abs_contribution": top.values
        })
        st.dataframe(contrib, use_container_width=True)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
def main():
    st.title("Fraud Detection with RF, LSTM & SHAP")

    # Sidebar: independent control, no cross-interference
    st.sidebar.header("Model & Options")
    model_choice = st.sidebar.selectbox("Model", ["Random Forest", "LSTM"], index=0)
    background_size = st.sidebar.slider(
        "Background sample for SHAP (LSTM)",
        min_value=10, max_value=300, value=64, step=8
    )
    nsamples_kernel = st.sidebar.slider(
        "SHAP nsamples (KernelExplainer fallback)",
        min_value=50, max_value=400, value=150, step=25
    )

    # Upload
    st.header("1) Upload CSV")
    uploaded = st.file_uploader("Upload a raw CSV", type=["csv"])
    if uploaded is None:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()
    try:
        df_uploaded = read_csv_robust(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Load required artifacts (RF & common scaler/features)
    rf_model = load_pickle(RF_MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
    feature_columns = load_json(FEATURES_PATH)

    missing = []
    if rf_model is None:
        missing.append(RF_MODEL_PATH.name)
    if scaler is None:
        missing.append(SCALER_PATH.name)
    if feature_columns is None:
        missing.append(FEATURES_PATH.name)
    if missing:
        st.error("Missing required artifacts:\n- " + "\n- ".join(missing))
        st.stop()

    # Parse feature list
    if isinstance(feature_columns, dict) and "feature_columns" in feature_columns:
        feature_columns = list(feature_columns["feature_columns"])
    elif isinstance(feature_columns, list):
        feature_columns = list(feature_columns)
    else:
        st.error("feature_columns.json could not be parsed into a list.")
        st.stop()

    # Preprocess & scale
    df_features, prep_log = preprocess_uploaded(df_uploaded, feature_columns)
    st.success(f"✅ Preprocessed: {df_features.shape[0]} rows × {df_features.shape[1]} features.")
    with st.expander("Preprocessing log"):
        st.json(prep_log)

    try:
        X_scaled = scaler.transform(df_features.values.astype(np.float32))
    except Exception as e:
        st.error(f"Scaler failed: {e}")
        st.stop()

    # Row selection
    st.header("2) Pick a row to Predict & Explain")
    row_index = st.number_input(
        "Row index (0-based)",
        min_value=0, max_value=max(0, len(df_features)-1), value=0, step=1
    )

    # LSTM is loaded ONLY if selected (keeps RF independent)
    lstm_model = None
    lstm_error = None
    if model_choice == "LSTM":
        lstm_model, lstm_error = load_lstm_model(LSTM_H5_PATH)
        if lstm_model is None:
            st.error("Could not load LSTM model from 'best_lstm.h5'. You can still use Random Forest.")
            if lstm_error:
                with st.expander("Show LSTM load error details"):
                    st.code(lstm_error)

    # Action
    if st.button("🔮 Predict & Explain Selected Row", use_container_width=True):
        try:
            if model_choice == "Random Forest" or (model_choice == "LSTM" and lstm_model is None):
                # RF path (fully independent of TensorFlow)
                preds, probs = predict_rf(rf_model, X_scaled)
                explanation = explain_rf_instance(rf_model, X_scaled[row_index:row_index+1], feature_columns)
            else:
                # LSTM path (TensorFlow loaded only here)
                preds, probs = predict_lstm(lstm_model, X_scaled)
                explanation = explain_lstm_instance(
                    lstm_model, X_scaled, row_index, feature_columns,
                    bg_size=background_size, nsamples=nsamples_kernel
                )
        except Exception as e_pred:
            st.error(f"Prediction/Explanation failed: {e_pred}")
            st.stop()

        # Results
        pred_label = "🚨 Fraudulent" if preds[row_index] == 1 else "✅ Legitimate"
        st.subheader("Prediction")
        st.write(f"**Row {row_index} Prediction:** {pred_label}")
        st.write(f"**Probability of Fraud:** {probs[row_index]:.2%}")

        # SHAP visualizations
        render_shap_plots(explanation, top_k=12)

    st.markdown("---")

if __name__ == "__main__":
    main()
