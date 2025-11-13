import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
from geopy.geocoders import Nominatim
import datetime

st.set_page_config(
    page_title="Quakify",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #edf3fa, #f4f8ff);
            color: #222;
        }
        .main-title {
            font-size: 34px;
            text-align: center;
            font-weight: 800;
            color: #14375a;
        }
        .subtext {
            text-align: center;
            font-size: 17px;
            color: #333;
            margin-bottom: 25px;
        }
        label {
            font-weight: 600 !important;
            color: #14375a !important;
        }
        .stNumberInput input {
            background-color: #ffffff !important;
            color: #111111 !important;
            font-weight: 600 !important;
        }
        .result {
            font-size: 24px;
            text-align: center;
            font-weight: 700;
            color: #0a3a5a;
        }
        .footer {
            text-align: center;
            font-size: 13px;
            color: #555;
            margin-top: 40px;
        }
        div.stButton > button:first-child {
            background-color: #007BFF !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            height: 3em !important;
            width: 100%;
            transition: 0.3s ease-in-out;
        }
        div.stButton > button:hover {
            background-color: #339CFF !important;
            transform: scale(1.05);
        }
        div[data-testid="stDownloadButton"] > button {
            background-color: #007BFF !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            border: none !important;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background-color: #339CFF !important;
        }
    </style>
""", unsafe_allow_html=True)
classification_path = r"C:\Users\Akshaya\OneDrive\Desktop\VI\classification(risk).pkl"
regression_path = r"C:\Users\Akshaya\OneDrive\Desktop\VI\regression.pkl"
clf_model = joblib.load(classification_path)
reg_model = joblib.load(regression_path)
st.sidebar.header("👩 Developer Info")
st.sidebar.write("""
**Name:** Akshaya  
**GitHub:** [github.com/Akshaya](https://github.com/Akshaya)
This app predicts **Earthquake Risk Levels**  
based on:
- 📍 Latitude  
- 🌐 Longitude  
- 🌊 Depth  
- 📅 Year
""")
st.markdown("<p class='main-title'>Quakify - Earthquake Risk Prediction</p>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Enter details below to predict the earthquake risk and visualize the location.</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    latitude_x = st.number_input("📍 Latitude", value=0.00, format="%.2f")
    depth = st.number_input("🌊 Depth (km)", value=0.00, format="%.2f")
with col2:
    longitude_x = st.number_input("🌐 Longitude", value=0.00, format="%.2f")
    year = st.number_input("📅 Year", min_value=1900, max_value=2100, value=1900, step=1)
if st.button("🚀 Predict Risk"):
    input_data = pd.DataFrame({
        'latitude_x': [latitude_x],
        'depth': [depth],
        'longitude_x': [longitude_x],
        'year': [year]
    })
    with st.spinner("Analyzing seismic data..."):
        risk_score = reg_model.predict(input_data)[0]
        prediction = clf_model.predict(input_data)[0]
        risk_labels = {0: "Low", 1: "Medium", 2: "High"}
        risk_label = risk_labels.get(prediction, "Unknown")
        try:
            geolocator = Nominatim(user_agent="geoapi")
            location = geolocator.reverse((latitude_x, longitude_x), language='en')
            if location and "address" in location.raw:
                address = location.raw["address"]
                area = address.get("suburb") or address.get("neighbourhood") or address.get("village") or address.get("town") or address.get("county")
                city = address.get("city") or address.get("municipality") or address.get("town")
                state = address.get("state")
                country = address.get("country")
                formatted_address = ", ".join([x for x in [area, city, state, country] if x])
            else:
                formatted_address = "Unknown Location"
        except Exception:
            formatted_address = "Unable to determine location"
        st.markdown(f"<p class='result'>Predicted Risk Category: <b>{risk_label}</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p class='result'>Predicted Risk Score: <b>{risk_score:.3f}</b></p>", unsafe_allow_html=True)
        st.write(f"📍 **Approximate Location:** {formatted_address}")
        st.write("### 🌍 Earthquake Location Map")
        map_data = pd.DataFrame({'lat': [latitude_x], 'lon': [longitude_x]})
        st.pydeck_chart(pdk.Deck(
            map_provider="mapbox",
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=latitude_x,
                longitude=longitude_x,
                zoom=5,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position='[lon, lat]',
                    get_color='[255, 0, 0, 230]',
                    get_radius=60000,
                ),
            ],
        ))
        st.write("### Safety Recommendations")
        if prediction == 2:
            st.markdown("### 🔴 **High Risk! Take these precautions:**")
            st.markdown("- Move to open areas away from tall structures")
            st.markdown("- Keep emergency supplies ready")
            st.markdown("- Stay alert for official earthquake alerts")
        elif prediction == 1:
            st.markdown("### 🟠 **Medium Risk! Stay alert:**")
            st.markdown("- Secure heavy objects and shelves")
            st.markdown("- Have an evacuation plan ready")
        else:
            st.markdown("### 🟢 **Low Risk! No immediate concerns.**")
            st.markdown("- Continue regular safety checks")
        df_log = pd.DataFrame({
            "Timestamp": [datetime.datetime.now()],
            "Latitude": [latitude_x],
            "Longitude": [longitude_x],
            "Depth": [depth],
            "Year": [year],
            "Location": [formatted_address],
            "Risk_Score": [risk_score],
            "Risk_Category": [risk_label]
        })
        df_log.to_csv("prediction_log.csv", mode='a', header=False, index=False)
        csv = df_log.to_csv(index=False)
        st.download_button("📄 Download Prediction Report", csv, "earthquake_prediction.csv", "text/csv")
with st.expander("💡 Example Input"):
    st.write("""
    **Example (Chennai, India):**
    - Latitude: `13.08`
    - Longitude: `80.27`
    - Depth: `30.00`
    - Year: `2024`
    """)