import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Earthquake Impact Classifier",
    page_icon="🌍"
)

# --- 2. Load Models & Sidebar ---
st.sidebar.header("App Controls")

@st.cache_resource
def load_models():
    """Loads the pipeline and label encoder from disk."""
    try:
        pipeline = joblib.load('earthquake_pipeline.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        return pipeline, label_encoder
    except FileNotFoundError:
        st.sidebar.error("Model files not found! Please run the training script first.")
        return None, None
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        return None, None

pipeline, label_encoder = load_models()

if pipeline and label_encoder:
    st.sidebar.success("Model Pipeline Ready!")
else:
    st.stop()

st.sidebar.header("About This App")
st.sidebar.info(
    "This app classifies the potential impact of an earthquake "
    "(Light, Moderate, Strong, or Major) based on its location, "
    "magnitude, depth, and local soil type. It uses an XGBoost model "
    "trained on historical data from the USGS and the Digital Soil Map of the World."
)

# --- 3. Prediction Function ---
@st.cache_data
def predict_earthquake_class(data, _pipeline, _encoder):
    """
    Predicts the earthquake magnitude class from user input.
    """
    df_new = pd.DataFrame([data])
    
    # Add interaction features
    try:
        df_new['depth_mag_interaction'] = df_new['depth'] * df_new['magnitude']
        df_new['lat_mag_interaction'] = df_new['latitude'] * df_new['magnitude']
    except Exception as e:
        st.error(f"Error creating interaction features: {e}")
        return None
        
    # Predict
    try:
        prediction_encoded = _pipeline.predict(df_new)
        prediction_class = _encoder.inverse_transform(prediction_encoded)
        return prediction_class[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# --- 4. Streamlit User Interface ---
st.title("Earthquake Impact Classification")
st.write("Enter the details of an earthquake event to predict its magnitude class. This model is trained to classify events based on *known* magnitude, location, and geological data.")

# --- Create input form ---
with st.form(key='prediction_form'):
    st.subheader("Event Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        latitude = st.number_input(
            "Latitude", 
            min_value=-90.0, 
            max_value=90.0, 
            value=34.05, 
            step=0.01, 
            help="Enter the latitude (e.g., 34.05 for Los Angeles)."
        )
        longitude = st.number_input(
            "Longitude", 
            min_value=-180.0, 
            max_value=180.0, 
            value=-118.25, 
            step=0.01, 
            help="Enter the longitude (e.g., -118.25 for Los Angeles)."
        )
        
    with col2:
        magnitude = st.number_input(
            "Magnitude (Richter)", 
            min_value=4.0, 
            max_value=10.0, 
            value=5.8, 
            step=0.1, 
            help="Enter the magnitude of the event."
        )
        depth = st.number_input(
            "Depth (in km)", 
            min_value=0.0, 
            value=10.5, 
            step=0.1, 
            help="Enter the depth of the event in kilometers."
        )

    # Note: This list should be dynamically generated from your dataset for a full production app.
    # We use the known values from your training data for this demo.
    soil_type = st.selectbox(
        "Soil Type (from DSMW map)", 
        options=['Unknown', 'Lx', 'I-d-2a', 'Bd', 'Bc', 'Qc', 'Re', 'Rd', 'Ao', 'Je', 'Zt', 'I-e-2b', 'I-b-2c', 'G-d-2a', 'A-o', 'V-a', 'E-c'],
        help="Select the dominant soil type (DOMSOI) from the geological map."
    )

    submit_button = st.form_submit_button(label='Classify Earthquake')

# --- 5. Handle prediction and display results ---
if submit_button:
    input_data = {
        'latitude': latitude,
        'longitude': longitude,
        'depth': depth,
        'magnitude': magnitude,
        'soil_type': soil_type
    }
    
    with st.spinner('Running classification...'):
        prediction = predict_earthquake_class(input_data, pipeline, label_encoder)
    
    if prediction:
        st.subheader("Prediction Result")
        
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            if prediction == "Major":
                st.error(f"**Predicted Class: {prediction}**")
                st.markdown("This event is classified as **Major**. It has the potential for severe damage and widespread impact. Immediate caution is advised for affected areas.")
            elif prediction == "Strong":
                st.warning(f"**Predicted Class: {prediction}**")
                st.markdown("This event is classified as **Strong**. It is likely to cause significant damage, especially to poorly constructed buildings near the epicenter.")
            else:
                st.success(f"**Predicted Class: {prediction}**")
                st.markdown(f"This event is classified as **{prediction}**. It is less likely to cause widespread, severe damage, though it may be felt strongly.")
        
        with res_col2:
            # New Visualization: Map of the location
            map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
            st.map(map_data, zoom=5)
            st.caption(f"Epicenter location: {latitude}, {longitude}")