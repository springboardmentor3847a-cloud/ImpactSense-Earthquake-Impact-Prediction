# ImpactSense – Earthquake Impact Prediction

**Author:** Sakthi Balaji A

**Tools:** Python, Scikit-learn, XGBoost, Geopandas, Pandas, NumPy, Gradio

## Project Overview

ImpactSense is a machine learning project that predicts earthquake severity by integrating seismic data (magnitude, depth) with geospatial soil data. It explores Logistic Regression, Random Forest, and XGBoost models and includes a Gradio app for real-time predictions.

## Objective

To build and evaluate models that classify an earthquake's Magnitude Class or Damage Risk based on its seismic and soil properties.

## Setup & How to Run

Get Files: Ensure you have ImpactSense.ipynb, the Significant_Earthquakes.xlsx dataset, and the Soil_data/ directory (with all DSMW shapefiles).

### Install Dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn geopandas xgboost gradio jupyter joblib


Run Notebook: Start Jupyter (jupyter notebook) and run the cells in ImpactSense.ipynb. The notebook will clean data, train models, and launch the Gradio UI.

## Modules Implemented

Data Cleaning: Loads and cleans raw earthquake .xlsx data.

Feature Engineering: Creates Magnitude_Class, Depth_Class, and Risk_Score from mag and depth.

Geospatial Enrichment: Uses Geopandas to join earthquake locations with DSMW.shx soil map data.

Data Visualization: Plots magnitude distribution and a global map of earthquake locations.

Model Development: Trains and evaluates LogisticRegression, RandomForestClassifier, and XGBClassifier.

User Interface: Loads the saved XGBClassifier into a Gradio web app for live predictions.

Developed By

Sakthibalaji A
