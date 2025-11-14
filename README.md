# ImpactSense – Earthquake Impact Prediction

**Author:** Sakthi Balaji A
**Tools:** Python, Scikit-learn, XGBoost, Geopandas, Pandas, NumPy, Gradio

## Project Overview

ImpactSense is a machine learning project that predicts earthquake severity by integrating seismic data (magnitude, depth) with geospatial soil data. It explores Logistic Regression, Random Forest, and XGBoost models and includes a Gradio app for real-time predictions. [cite: springboardmentor3847a-cloud/impactsense-earthquake-impact-prediction/ImpactSense-Earthquake-Impact-Prediction-Sakthibalaji-branch/ImpactSense.ipynb]

## Objective

To build and evaluate models that classify an earthquake's Magnitude Class or Damage Risk based on its seismic and soil properties. [cite: springboardmentor3847a-cloud/impactsense-earthquake-impact-prediction/ImpactSense-Earthquake-Impact-Prediction-Sakthibalaji-branch/ImpactSense.ipynb]

## Setup & How to Run

Get Files: Ensure you have ImpactSense.ipynb, the Significant_Earthquakes.xlsx dataset, and the Soil_data/ directory (with all DSMW shapefiles). [cite: springboardmentor3847a-cloud/impactsense-earthquake-impact-prediction/ImpactSense-Earthquake-Impact-Prediction-Sakthibalaji-branch/ImpactSense.ipynb]

## Install Dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn geopandas xgboost gradio jupyter joblib


Run Notebook: Start Jupyter (jupyter notebook) and run the cells in ImpactSense.ipynb. The notebook will clean data, train models, and launch the Gradio UI. [cite: springboardmentor3847a-cloud/impactsense-earthquake-impact-prediction/ImpactSense-Earthquake-Impact-Prediction-Sakthibalaji-branch/ImpactSense.ipynb]

## Modules Implemented

Data Cleaning: Loads and cleans raw earthquake .xlsx data. [cite: springboardmentor3847a-cloud/impactsense-earthquake-impact-prediction/ImpactSense-Earthquake-Impact-Prediction-Sakthibalaji-branch/ImpactSense.ipynb]

Feature Engineering: Creates Magnitude_Class, Depth_Class, and Risk_Score from mag and depth. [cite: springboardmentor3847a-cloud/impactsense-earthquake-impact-prediction/ImpactSense-Earthquake-Impact-Prediction-Sakthibalaji-branch/ImpactSense.ipynb]

Geospatial Enrichment: Uses Geopandas to join earthquake locations with DSMW.shx soil map data. [cite: springboardmentor3847a-cloud/impactsense-earthquake-impact-prediction/ImpactSense-Earthquake-Impact-Prediction-Sakthibalaji-branch/ImpactSense.ipynb]

Data Visualization: Plots magnitude distribution and a global map of earthquake locations. [cite: springboardmentor3847a-cloud/impactsense-earthquake-impact-prediction/ImpactSense-Earthquake-Impact-Prediction-Sakthibalaji-branch/ImpactSense.ipynb]

Model Development: Trains and evaluates LogisticRegression, RandomForestClassifier, and XGBClassifier. [cite: springboardmentor3847a-cloud/impactsense-earthquake-impact-prediction/ImpactSense-Earthquake-Impact-Prediction-Sakthibalaji-branch/ImpactSense.ipynb]

User Interface: Loads the saved XGBClassifier into a Gradio web app for live predictions. [cite: springboardmentor3847a-cloud/impactsense-earthquake-impact-prediction/ImpactSense-Earthquake-Impact-Prediction-Sakthibalaji-branch/ImpactSense.ipynb]

Developed By
Sakthibalaji A
