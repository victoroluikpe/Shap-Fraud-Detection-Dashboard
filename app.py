# app.py
"""
Fraud Detection Dashboard (RF + LSTM) with Individual, Colorful SHAP Explanations
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

# ========= Streamlit page config =========
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
shap.initjs()  # ensure JS library loads for force plots

# ========= Paths =========
BASE_DIR = Path(__file__).parent

# All artifacts are now assumed in repo root
RF_MODEL_PATH   = BASE_DIR / "random_forest_tuned.pkl"
LSTM_H5_PATH    = BASE_DIR / "best_lstm.h5"
SCALER_PATH     = BASE_DIR / "scaler.pkl"
FEATURES_PATH   = BASE_DIR / "feature_columns.json"

# ========= Cached loaders =========
@st.cache_resource
def load_pickle(path: Path):
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return joblib.load(f)
    return None

@st.cache_resource
def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None

@st.cache_resource
def load_lstm(path: Path):
    if path.exists():
        try:
            return tf.keras.models.load_model(str(path))
        except Exception:
            return None
    return None

# ========= Load artifacts =========
rf_model        = load_pickle(RF_MODEL_PATH)
scaler          = load_pickle(SCALER_PATH)
feature_columns = load_json(FEATURES_PATH)
lstm_model      = load_lstm(LSTM_H5_PATH)

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
    bases = [c.split("_")[0] for c in features if "_" in c]
    return sorted(list(set(bases)))

def preprocess_uploaded(df_raw: pd.DataFrame, features: list[str], cardinality_threshold: int = 100):
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

# ---- RF: cached explainer
@st.cache_resource
def get_rf_explainer():
    try:
        return shap.TreeExplainer(rf_model, model_output="probability", feature_perturbation="interventional")
    except Exception:
        return shap.TreeExplainer(rf_model)

def shap_force_plot_html(explanation: shap.Explanation):
    obj = shap.plots.force(explanation, matplotlib=False)
    try:
        html = obj.html()
    except Exception:
        html = str(obj)
    return f"<head>{shap.getjs()}</head><body>{html}</body>"

# ========= UI =========
st.title("💳 Fraud Detection (RF + LSTM) with Individual SHAP Explanations")

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
            st.error("No LSTM model found. Please place 'best_lstm.h5' in the repo root.")
            st.stop()
        preds, probs = predict_lstm(X_scaled)
        explanation = explain_lstm_instance(X_scaled, row_index, feature_columns,
                                            bg_size=background_size, nsamples=nsamples_kernel)

    pred_label = "🚨 Fraudulent" if preds[row_index] == 1 else "✅ Legitimate"
    st.subheader("Prediction")
    st.write(f"**Row {row_index} Prediction:** {pred_label}")
    st.write(f"**Probability of Fraud:** {probs[row_index]:.2%}")

    st.subheader("Explanation for this Transaction (SHAP)")
    st.markdown("**Waterfall Plot (feature contributions → prediction)**")
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    st.pyplot(fig, clear_figure=True)

    st.markdown("**Force Plot (interactive, colorful)**")
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
