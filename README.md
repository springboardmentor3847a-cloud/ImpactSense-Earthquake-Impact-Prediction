# 🌍 Earthquake Impact App

A data-driven pipeline for analyzing and predicting earthquake damage using historical records, real-time API data, soil classification, and machine learning models.

## 🚀 Features

- 🔎 Real-time earthquake data from USGS API
- 📊 Exploratory data analysis and geospatial mapping
- 🧠 Feature engineering with soil types, depth, magnitude, and clustering
- 🤖 ML models: Random Forest, XGBoost, Logistic Regression, Decision Tree
- 📈 SHAP-based model explainability
- 🧪 Streamlit dashboard for interactive predictions

## 📁 Folder Structure
earthquake-impact-app/ ├── home.py ├── pages/ │   ├── 1_Predictor.py │   └── 2_Dashboard.py ├── models/                # Saved ML models (.pkl) ├── notebooks/             # EDA and preprocessing pipeline ├── requirements.txt ├── README.md └── .gitignore


## 📂 Data Access

Due to GitHub's 100MB file limit, large datasets are hosted externally:

- [earthquake_features_day4.csv](https://drive.google.com/your-link)
- [earthquake_preprocessed_week2.csv](https://drive.google.com/your-link)
- [optimized_rf_model.pkl](https://drive.google.com/your-link)

## 🛠️ Setup

```bash
pip install -r requirements.txt
streamlit run home.py