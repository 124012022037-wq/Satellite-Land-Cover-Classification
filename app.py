from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "land_cover_model.joblib"
FEATURES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI"]

st.set_page_config(page_title="Satellite Land Cover Classification", page_icon="🛰️", layout="centered")
st.title("🛰️ Satellite Land Cover Classification")
st.write("Enter the spectral-band values and NDVI to predict a land-cover category.")
st.info("Educational demonstration using a synthetic 700-row CSV dataset. Predictions are not validated satellite mapping results.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Input satellite features")
col1, col2 = st.columns(2)
with col1:
    blue = st.number_input("Blue", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.3f")
    green = st.number_input("Green", min_value=0.0, max_value=1.0, value=0.15, step=0.01, format="%.3f")
    red = st.number_input("Red", min_value=0.0, max_value=1.0, value=0.12, step=0.01, format="%.3f")
    nir = st.number_input("NIR", min_value=0.0, max_value=1.0, value=0.42, step=0.01, format="%.3f")
with col2:
    swir1 = st.number_input("SWIR1", min_value=0.0, max_value=1.0, value=0.25, step=0.01, format="%.3f")
    swir2 = st.number_input("SWIR2", min_value=0.0, max_value=1.0, value=0.16, step=0.01, format="%.3f")
    ndvi = st.number_input("NDVI", min_value=-1.0, max_value=1.0, value=0.55, step=0.01, format="%.01f")

if st.button("Predict Land Cover", type="primary", use_container_width=True):
    values = pd.DataFrame([[blue, green, red, nir, swir1, swir2, ndvi]], columns=FEATURES)
    prediction = model.predict(values)[0]
    st.success(f"Predicted land-cover category: **{prediction}**")
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(values).max())
        st.metric("Model confidence", f"{confidence:.1%}")
    st.caption("Confidence is the model's estimated probability, not a guarantee of correctness.")

st.divider()
st.subheader("About the project")
st.write("This mini-project compares Decision Tree, Random Forest, SVM and KNN classifiers using seven tabular spectral features.")
st.caption("For real-world use, train and validate with authentic, labelled satellite observations and document the sensor, bands, units and preprocessing.")
