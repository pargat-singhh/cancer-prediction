"""
Cancer Type Prediction - Web Application
=========================================
A simple Streamlit app to predict cancer type from gene expression data.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page setup ────────────────────────────────────────────────────
st.set_page_config(page_title="Cancer Type Predictor", page_icon="🧬", layout="centered")

st.title("🧬 Cancer Type Prediction")
st.write("Upload gene expression data (CSV) to predict the cancer type using our trained ML model.")

# ── Load model and required files ─────────────────────────────────
MODEL_PATH = "models/best_model.pkl"
FEATURES_PATH = "models/feature_names.csv"
CLASSES_PATH = "models/cancer_classes.csv"

# check if model exists
if not os.path.exists(MODEL_PATH):
    st.error("Model not found! Please run the Jupyter notebook first to train the model.")
    st.stop()

# load once and cache
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_PATH, header=None)[0].tolist()

@st.cache_data
def load_classes():
    return pd.read_csv(CLASSES_PATH, header=None)[0].tolist()

model = load_model()
feature_names = load_features()
cancer_classes = load_classes()

# cancer type full names for display
cancer_fullnames = {
    "BRCA": "Breast Cancer",
    "KIRC": "Kidney Cancer",
    "LUAD": "Lung Cancer",
    "PRAD": "Prostate Cancer",
    "COAD": "Colon Cancer"
}

st.success(f"Model loaded! (expects {len(feature_names)} gene features)")
st.divider()

# ── Sidebar with info ────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write("This app predicts cancer type from gene expression data using a machine learning model trained on the TCGA dataset.")

    st.subheader("Cancer Types")
    for code, name in cancer_fullnames.items():
        st.write(f"**{code}** — {name}")

    st.divider()
    st.caption("For educational purposes only.")

# ── File upload ───────────────────────────────────────────────────
st.subheader("📤 Upload your CSV file")
st.write("The CSV should have gene expression values. Each row = one patient sample.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# option to try with random sample data
use_demo = st.checkbox("Or try with demo data (random values)")

if use_demo:
    # generate random demo data (5 samples)
    np.random.seed(42)
    demo_data = pd.DataFrame(
        np.random.rand(5, len(feature_names)) * 10,
        columns=feature_names
    )
    st.write("Demo data preview (first 6 columns):")
    st.dataframe(demo_data.iloc[:, :6])
    input_data = demo_data

elif uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write(f"Uploaded file: {input_data.shape[0]} samples, {input_data.shape[1]} columns")
    st.dataframe(input_data.iloc[:, :6])
else:
    st.info("Please upload a CSV file or check the demo data box above.")
    st.stop()

# ── Make predictions ──────────────────────────────────────────────
st.divider()
st.subheader("🔮 Prediction Results")

# remove non-gene columns if present (like sample_id)
gene_data = input_data.copy()
for col in ['sample_id', 'Unnamed: 0', 'Class']:
    if col in gene_data.columns:
        gene_data = gene_data.drop(columns=[col])

# check if columns match
if len(gene_data.columns) != len(feature_names):
    st.warning(f"Expected {len(feature_names)} gene columns but got {len(gene_data.columns)}. "
               f"Trying to align columns...")
    # try to match columns
    common = [c for c in feature_names if c in gene_data.columns]
    if len(common) > 0:
        missing = [c for c in feature_names if c not in gene_data.columns]
        gene_data = gene_data.reindex(columns=feature_names, fill_value=0)
        st.write(f"Matched {len(common)} genes. Filled {len(missing)} missing genes with 0.")
    else:
        # assume columns are in correct order
        gene_data.columns = feature_names

# predict
try:
    predictions = model.predict(gene_data)
    probabilities = model.predict_proba(gene_data)

    # show results for each sample
    for i in range(len(predictions)):
        pred = predictions[i]
        prob = probabilities[i]
        confidence = max(prob) * 100
        full_name = cancer_fullnames.get(pred, pred)

        # color based on confidence
        if confidence >= 80:
            color = "🟢"
        elif confidence >= 50:
            color = "🟡"
        else:
            color = "🔴"

        st.markdown(f"### Sample {i+1}: **{pred}** ({full_name}) {color}")
        st.write(f"Confidence: **{confidence:.1f}%**")

        # show probability for each cancer type as a bar chart
        prob_df = pd.DataFrame({
            'Cancer Type': cancer_classes,
            'Probability': prob * 100
        })
        st.bar_chart(prob_df.set_index('Cancer Type'))
        st.divider()

    st.balloons()

except Exception as e:
    st.error(f"Error during prediction: {str(e)}")
    st.write("Make sure your CSV has the correct gene expression format.")
