import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import base64
import io
import os
import matplotlib.pyplot as plt
from datetime import datetime

# === Load model, scaler, and label encoder ===
@st.cache_resource
def load_assets():
    model = joblib.load("optimized_rf_model.pkl")
    scaler = joblib.load("optimized_scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_assets()

# === Soil and Region Choices ===
SOIL_CHOICES = [
    (0, "Af - Arenosols (Ferric)"), (1, "Ag - Arenosols (Gleyic)"), (2, "Ah - Arenosols (Haplic)"),
    (3, "Ao - Arenosols (Albic)"), (4, "Ap - Arenosols (Protic)"), (5, "Bc - Cambisols (Chromic)")
    # Extend this list as needed
]
REGION_CHOICES = [
    (0, "Cluster 0 - Low Seismic Risk"), (1, "Cluster 1 - Moderate Seismic Risk"),
    (2, "Cluster 2 - High Seismic Risk"), (3, "Cluster 3 - Very High Seismic Risk"),
    (4, "Cluster 4 - Extreme Seismic Risk"), (5, "Cluster 5 - Critical Seismic Risk")
]

SOIL_DICT = {label: idx for idx, label in SOIL_CHOICES}
REGION_DICT = {label: idx for idx, label in REGION_CHOICES}

LAT_BANDS = ['90°S-60°S', '60°S-30°S', '30°S-0°', '0°-30°N', '30°N-60°N', '60°N-90°N']
LON_BANDS = ['180°W-120°W', '120°W-60°W', '60°W-0°', '0°-60°E', '60°E-120°E', '120°E-180°E']

def encode_band(value, bins, labels):
    band = pd.cut([value], bins=bins, labels=labels)[0]
    return labels.index(band) if band in labels else -1

# === Streamlit UI ===
st.set_page_config(page_title="Earthquake Damage Predictor", layout="centered")
st.title("🌍 Earthquake Damage Predictor")
st.markdown("Predict structural damage risk based on seismic parameters using a trained Random Forest model.")

# === Inputs ===
magnitude = st.slider("Magnitude (Richter Scale)", 0.0, 10.0, 5.5, 0.1)
depth = st.slider("Depth (km)", 0.0, 700.0, 10.0, 1.0)
latitude = st.slider("Latitude", -90.0, 90.0, 28.6)
longitude = st.slider("Longitude", -180.0, 180.0, 77.3)
soil_label = st.selectbox("Soil Type", [label for _, label in SOIL_CHOICES])
region_label = st.selectbox("Region Cluster", [label for _, label in REGION_CHOICES])

# === Prediction ===
if st.button("🚀 Predict Damage"):    
    # Encode categorical features
    soil_encoded = SOIL_DICT.get(soil_label, -1)
    region_encoded = REGION_DICT.get(region_label, -1)
    domsoi_encoded = soil_encoded  # Assuming DOMSOI matches soil type

    lat_band_encoded = encode_band(latitude, np.arange(-90, 91, 30), LAT_BANDS)
    lon_band_encoded = encode_band(longitude, np.arange(-180, 181, 60), LON_BANDS)
    year = datetime.now().year
    risk_score = magnitude * 0.7 + np.log1p(depth) * 0.3

    # Create input dataframe
    input_df = pd.DataFrame([{
        "Magnitude": magnitude,
        "Depth": depth,
        "Risk_Score": risk_score,
        "SoilType_Encoded": soil_encoded,
        "Region_Cluster": region_encoded,
        "DOMSOI_Encoded": domsoi_encoded,
        "Latitude_Band_Encoded": lat_band_encoded,
        "Longitude_Band_Encoded": lon_band_encoded,
        "Year": year
    }])

    # Scale numeric features
    numeric_cols = ['Magnitude', 'Depth', 'Risk_Score', 'Year']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    damage_label = label_encoder.inverse_transform([prediction])[0]
    confidence = np.max(model.predict_proba(input_df)[0])
    
    st.markdown("---")
    st.header("📊 Prediction Dashboard")

    log_path = "prediction_log.csv"
    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path, parse_dates=["Timestamp"])
    
        # 1. Line chart: Predictions per day
        df_log["Date"] = df_log["Timestamp"].dt.date
        daily_counts = df_log.groupby("Date").size().reset_index(name="Predictions")
        st.subheader("📈 Predictions Over Time")
        st.line_chart(daily_counts.set_index("Date"))

        # 2. Bar chart: Damage category frequency
        st.subheader("📊 Damage Category Frequency")
        damage_counts = df_log["Predicted_Damage"].value_counts()
        st.bar_chart(damage_counts)
    
        # 3. Boxplot: Magnitude by Damage Category
        st.subheader("📦 Magnitude Distribution by Damage Category")
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_log, x="Predicted_Damage", y="Magnitude", ax=ax)
        st.pyplot(fig)

        # 4. Line chart: Confidence over time
        st.subheader("🔍 Confidence Trend")
        confidence_trend = df_log.groupby("Date")["Confidence"].mean().reset_index()
        st.line_chart(confidence_trend.set_index("Date"))

    else:
        st.warning("No prediction history found. Run a few predictions to populate the dashboard.")       
    

    # === Output ===
    st.success(f"**Predicted Damage Category:** {damage_label}")
    st.markdown(f"**Model Confidence:** {confidence:.2%}")
    st.markdown(f"""
    **Inputs:**
    - Magnitude: {magnitude}
    - Depth: {depth} km
    - Latitude: {latitude}
    - Longitude: {longitude}
    - Soil Type: {soil_label}
    - Region Cluster: {region_label}
    """)

    # === SHAP Visualization ===
    st.markdown("### 🔍 SHAP Feature Impact")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(fig)

    # === Export to CSV ===
    st.markdown("### 📁 Export Prediction")
    export_df = input_df.copy()
    export_df["Predicted_Damage"] = damage_label
    export_df["Confidence"] = confidence
    export_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Download Prediction as CSV",
        data=csv_buffer.getvalue(),
        file_name="earthquake_prediction.csv",
        mime="text/csv"
    )

    # === Log Prediction Locally ===
    log_path = "prediction_log.csv"
    if os.path.exists(log_path):
        export_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        export_df.to_csv(log_path, index=False)
