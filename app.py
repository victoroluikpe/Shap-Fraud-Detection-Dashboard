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
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
RF_MODEL_PATH   = BASE_DIR / "random_forest_tuned.pkl"
SCALER_PATH     = BASE_DIR / "scaler.pkl"
FEATURES_PATH   = BASE_DIR / "feature_columns.json"

# ========= Cached loaders =========
@st.cache_resource
def load_pickle(path: Path):
    if path.exists():
        return joblib.load(path)
    return None

@st.cache_resource
def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None

# --- Improved LSTM loader ---
@st.cache_resource
def load_lstm_model_candidates(base_dir: Path):
    """
    Try common locations and formats for the LSTM model.
    Returns (model_or_None, error_message_or_None).
    """
    candidates = [
        base_dir / "best_lstm.h5",
        base_dir / "models" / "best_lstm.h5",
        base_dir / "best_lstm",           # SavedModel dir
        base_dir / "models" / "best_lstm" # SavedModel dir
    ]
    tried = []
    errors = []
    for p in candidates:
        tried.append(str(p))
        if not p.exists():
            continue
        try:
            m = tf.keras.models.load_model(str(p), compile=False)
            return m, None
        except Exception as e:
            errors.append(f"{p}: {type(e).__name__}: {e}")

    err_msg = (
        "No model loaded. Tried these paths:\n" + "\n".join(tried) +
        ("\n\nErrors:\n" + "\n".join(errors[:5]) if errors else " (no candidate files present)")
    )
    return None, err_msg

# ========= Load artifacts =========
rf_model        = load_pickle(RF_MODEL_PATH)
scaler          = load_pickle(SCALER_PATH)
feature_columns = load_json(FEATURES_PATH)
lstm_model, lstm_load_error = load_lstm_model_candidates(BASE_DIR)

# --- Sidebar model uploader for LSTM ---
uploaded_lstm = st.sidebar.file_uploader("Upload LSTM model (.h5)", type=["h5"])
if uploaded_lstm is not None:
    tmp_path = Path("/tmp") / "uploaded_best_lstm.h5"
    tmp_path.write_bytes(uploaded_lstm.read())
    try:
        lstm_model = tf.keras.models.load_model(str(tmp_path), compile=False)
        st.sidebar.success("✅ Uploaded LSTM loaded successfully.")
        lstm_load_error = None
    except Exception as e:
        st.sidebar.error(f"Uploaded LSTM failed: {type(e).__name__}: {e}")

# ========= Sanity checks =========
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

if isinstance(feature_columns, dict) and "feature_columns" in feature_columns:
    feature_columns = list(feature_columns["feature_columns"])
elif isinstance(feature_columns, list):
    feature_columns = list(feature_columns)
else:
    st.error("feature_columns.json could not be parsed into a list.")
    st.stop()

# ========= Sidebar =========
st.sidebar.header("Model & Options")
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "LSTM"], index=0)
background_size = st.sidebar.slider("Background sample for SHAP (for LSTM explainer)",
                                    min_value=10, max_value=300, value=64, step=8)
nsamples_kernel = st.sidebar.slider("SHAP nsamples (KernelExplainer fallback)",
                                    min_value=50, max_value=400, value=150, step=25)

# ========= Helpers =========
def st_shap(plot_html, height=320):
    components.html(plot_html, height=height)

def safe_str(x):
    try:
        return str(x)
    except Exception:
        return "Other"

def extract_categorical_bases_from_features(features):
    return sorted(list({c.split("_")[0] for c in features if "_" in c}))

def preprocess_uploaded(df_raw: pd.DataFrame, features: list[str], cardinality_threshold: int = 100):
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
        df.drop(columns=extras, inplace=True)
        log["dropped_extra"].extend(extras)

    df = df[features]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df, log

def predict_rf(X_scaled: np.ndarray):
    probs = rf_model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def predict_lstm(X_scaled_2d: np.ndarray):
    X_seq = X_scaled_2d.reshape((X_scaled_2d.shape[0], 1, X_scaled_2d.shape[1]))
    probs = lstm_model.predict(X_seq, verbose=0)
    probs = np.asarray(probs).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    return preds, probs

@st.cache_resource
def get_rf_explainer():
    try:
        return shap.TreeExplainer(rf_model, model_output="probability", feature_perturbation="interventional")
    except Exception:
        return shap.TreeExplainer(rf_model)

def explain_rf_instance(x_row_2d: np.ndarray, feature_names: list[str]):
    explainer = get_rf_explainer()
    out = explainer.shap_values(x_row_2d)
    if isinstance(out, shap.Explanation):
        return out
    if isinstance(out, list):
        return shap.Explanation(values=out[1][0], base_values=explainer.expected_value[1],
                                data=x_row_2d[0], feature_names=feature_names)
    return shap.Explanation(values=out[0], base_values=explainer.expected_value,
                            data=x_row_2d[0], feature_names=feature_names)

def explain_lstm_instance(X_scaled_all: np.ndarray, idx: int, feature_names: list[str],
                          bg_size: int = 64, nsamples: int = 150):
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
        sv_list = explainer.shap_values(x0_3d)
        sv_arr = sv_list[0] if isinstance(sv_list, list) else sv_list
        sv_vec = np.array(sv_arr).reshape(-1)[-X_scaled_all.shape[1]:]
        base_value = float(np.array(explainer.expected_value).reshape(-1)[0])
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

def shap_force_plot_html(explanation: shap.Explanation):
    obj = shap.plots.force(explanation, matplotlib=False)
    return f"<head>{shap.getjs()}</head><body>{obj.html() if hasattr(obj,'html') else str(obj)}</body>"

# ========= UI =========
st.title(" Fraud Detection Dashboard")

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
        preds, probs = predict_rf(X_scaled)
        explanation = explain_rf_instance(X_scaled[row_index:row_index+1], feature_columns)
    else:
        if lstm_model is None:
            st.error("❌ LSTM not loaded. See sidebar debug info.")
            if lstm_load_error:
                with st.expander("LSTM load debug info"):
                    st.code(lstm_load_error)
            st.stop()
        preds, probs = predict_lstm(X_scaled)
        explanation = explain_lstm_instance(X_scaled, row_index, feature_columns,
                                            bg_size=background_size, nsamples=nsamples_kernel)

    pred_label = "🚨 Fraudulent" if preds[row_index] == 1 else "✅ Legitimate"
    st.subheader("Prediction")
    st.write(f"**Row {row_index} Prediction:** {pred_label}")
    st.write(f"**Probability of Fraud:** {probs[row_index]:.2%}")

    st.subheader("Explanation for this Transaction (SHAP)")

    st.markdown("**Waterfall Plot**")
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    st.pyplot(fig, clear_figure=True)

    st.markdown("**Force Plot (interactive)**")
    html = shap_force_plot_html(explanation)
    st_shap(html, height=320)

    with st.expander("Top Feature Contributions (abs SHAP)"):
        vals = pd.Series(explanation.values, index=feature_columns)
        top = vals.abs().sort_values(ascending=False).head(15)
        contrib = pd.DataFrame({
            "feature": top.index,
            "shap_value": vals.loc[top.index].values,
            "abs_contribution": top.values
        })
        st.dataframe(contrib)

st.markdown("---")
