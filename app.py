import streamlit as st
import pandas as pd
import joblib

# --- Load the saved model and other necessary objects ---

# Load the trained Random Forest model
model = joblib.load('models/random_forest_model.joblib')

# Load the preprocessor
preprocessor = joblib.load('models/preprocessor.joblib')

# Load the list of unique soil types for the dropdown menu
soil_types = joblib.load('models/soil_types.joblib')


# --- Build the User Interface using Streamlit ---

# Set the title of the web app
st.title('ImpactSense: Earthquake Impact Prediction 🌍')

# Create input fields for the user
st.header('Enter Earthquake Details')

# Input for Magnitude
magnitude = st.number_input(
    'Magnitude (mag)',
    min_value=0.0,
    max_value=10.0,
    value=5.5, # Default value
    step=0.1
)

# Input for Latitude and Longitude
col1, col2 = st.columns(2)
with col1:
    latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=0.0)
with col2:
    longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=0.0)

# Dropdown for Soil Type
soil_type = st.selectbox('Dominant Soil Type (DOMSOI)', options=soil_types)

# --- Prediction Logic ---

# Create a button that triggers the model prediction
if st.button('Predict Impact Risk'):
    # Create a DataFrame from the user's inputs
    # The structure must match the one used for training
    input_data = pd.DataFrame({
        'mag': [magnitude],
        'latitude': [latitude],
        'longitude': [longitude],
        'DOMSOI': [soil_type]
    })

    # Use the preprocessor to transform the input data
    input_data_processed = preprocessor.transform(input_data)

    # Use the model to make a prediction
    prediction = model.predict(input_data_processed)
    prediction_proba = model.predict_proba(input_data_processed)

    # --- Display the Prediction ---
    st.header('Prediction Result')
    
    risk_level = prediction[0]
    
    if risk_level == 'High':
        st.error(f'Predicted Risk Level: High 🚨')
    elif risk_level == 'Medium':
        st.warning(f'Predicted Risk Level: Medium ⚠️')
    else:
        st.success(f'Predicted Risk Level: Low ✅')

    # Display prediction probabilities for more detail
    st.write("Prediction Confidence:")
    st.write(f"- Low: {prediction_proba[0][1]:.2%}")
    st.write(f"- Medium: {prediction_proba[0][2]:.2%}")
    st.write(f"- High: {prediction_proba[0][0]:.2%}")