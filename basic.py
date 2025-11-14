import streamlit as st
import pandas as pd
import joblib
model_path = r"C:\Users\Akshaya\OneDrive\Desktop\xgboost_final.pkl"
xgb_model = joblib.load(model_path)
st.title("Earthquake Risk Category Prediction")
st.write("Enter the details below to predict the **risk category**")
longitude_x = st.number_input("Longitude", format="%.3f")
latitude_x = st.number_input("Latitude", format="%.3f")
depth = st.number_input("Depth (km)", format="%.3f")
year = st.number_input("Year", min_value=1900, max_value=2100, step=1)

if st.button("Predict Risk Category"):
    data = pd.DataFrame({
        'longitude_x': [longitude_x],
        'latitude_x': [latitude_x],
        'depth': [depth],
        'year': [year]
    })
    prediction = xgb_model.predict(data)[0]
    labels = {0: "Low", 1: "Medium", 2: "High"}
    risk_label = labels.get(prediction, "Unknown")
    st.success(f"Predicted Risk Category: **{risk_label} ({prediction})**")
