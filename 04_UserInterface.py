# ============================================================
# ImpactSense: Earthquake Impact Prediction - Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# ------------------------------
# Load Models & Encoders
# ------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "pkl_models")

@st.cache_data
def load_models():
    model = joblib.load(os.path.join(MODEL_DIR, 'impact_model.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    soil_encoder = joblib.load(os.path.join(MODEL_DIR, 'soil_encoder.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    soil_types = joblib.load(os.path.join(MODEL_DIR, 'soil_type_names.pkl'))
    return model, label_encoder, soil_encoder, scaler, soil_types

model, label_encoder, soil_encoder, scaler, soil_types = load_models()
st.success("✅ Model and encoders loaded successfully")

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(page_title="ImpactSense: Earthquake Impact Predictor", layout='wide')
st.title("🌍 ImpactSense: Earthquake Impact Prediction")

st.markdown("""
Predict the potential **impact level of an earthquake** based on real seismic parameters  
(Magnitude, Depth, Latitude, Longitude, and Soil Type).
""")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Enter Earthquake Parameters")
magnitude = st.sidebar.number_input("Magnitude", min_value=0.0, max_value=10.0, step=0.1)
depth = st.sidebar.number_input("Depth (km)", min_value=0.0, max_value=700.0, step=1.0)
latitude = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.01)
longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.01)
soil_type = st.sidebar.selectbox("Soil Type", soil_types)

# ------------------------------
# Prepare Input Data
# ------------------------------
input_df = pd.DataFrame([{
    "Magnitude": magnitude,
    "Depth": depth,
    "Latitude": latitude,
    "Longitude": longitude,
    "Soil_Type": soil_type
}])

# Encode Soil_Type
input_df['Soil_Type_Encoded'] = soil_encoder.transform(input_df['Soil_Type'])

# Feature columns used during training
feature_cols = ['Magnitude', 'Depth', 'Latitude', 'Longitude', 'Soil_Type_Encoded']

# Scale input
input_scaled = scaler.transform(input_df[feature_cols])

# ------------------------------
# Prediction
# ------------------------------
if st.sidebar.button("🔍 Predict Impact Level"):
    pred = model.predict(input_scaled)
    pred_class = label_encoder.inverse_transform(pred)[0]

    impact_color = {"Low": "green", "Medium": "orange", "High": "red"}
    color = impact_color.get(pred_class, "black")

    st.markdown(
        f"## Predicted Impact Level: <span style='color:{color}'>{pred_class}</span>",
        unsafe_allow_html=True
    )

    # Contextual alert
    if pred_class == "High":
        st.error("🚨 High Impact Alert! Major damage possible.")
    elif pred_class == "Medium":
        st.warning("⚠️ Moderate Impact. Stay alert.")
    else:
        st.success("✅ Low Impact. No major concerns.")

    # Map visualization
    if latitude and longitude:
        st.subheader("🌐 Earthquake Location Visualization")
        map_df = pd.DataFrame({
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Impact Level': [pred_class]
        })
        fig = px.scatter_map(
            map_df,
            lat='Latitude',
            lon='Longitude',
            color='Impact Level',
            color_discrete_map=impact_color,
            zoom=3,
            height=400,
            size_max=15
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------
        # Download option
        # ------------------------------
        download_df = input_df.drop(columns='Soil_Type_Encoded')
        download_df['Predicted Impact Level'] = pred_class
        csv = download_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Prediction as CSV", data=csv, file_name="impact_prediction.csv")
