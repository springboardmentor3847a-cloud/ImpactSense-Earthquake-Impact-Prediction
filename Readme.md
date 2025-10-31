# 🌍 ImpactSense – Earthquake Impact Prediction  

**Author:** Sarthak Singh  
**Tools Used:** Python, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Gradio / Streamlit (optional UI)  

---

## 📖 Project Overview  

**ImpactSense** is a machine learning–based system designed to predict the *impact and severity* of earthquakes using geophysical and environmental data such as magnitude, depth, latitude, longitude, and soil characteristics.  

The model helps in **urban planning**, **disaster management**, and **emergency response** by classifying earthquake risks or predicting potential damage levels.  

---

## 🎯 Objective  

To build a predictive model that estimates the **impact zone**, **risk level**, or **damage severity** of earthquakes based on seismic and soil data.  

---

## 🧱 Project Structure  

```text
IMPACTSENSE_PROJECT/
│
├── Dataset/
│   ├── dmsw/
│   │   └── DSMW/
│   │       ├── BasicFilesSC.xls
│   │       ├── DSMW.dbf
│   │       ├── DSMW.lyr
│   │       ├── DSMW_noborder.lyr
│   │       ├── DSMW.sbn
│   │       ├── DSMW.sbx
│   │       ├── DSMW.shp
│   │       ├── DSMW.shp.xml
│   │       ├── Generalized_SU_Info.xls
│   │       ├── SU_Info.xls
│   │       ├── WORLD764.xls
│   │       ├── SoilData.doc
│   │       └── DSMW.shx
│   ├── baseline_results_week3.csv
│   ├── earthquake_cleaned_day1.csv
│   ├── earthquake_encoded_day3.csv
│   ├── earthquake_features_day4.csv
│   ├── earthquake_preprocessed_week2.csv
│   ├── earthquake_scaled_day2.csv
│   ├── Significant_Earthquakes.csv
│   ├── unified_earthquake_data.csv
│   ├── label_encoder.pkl
│   ├── random_forest_day1.pkl
│   ├── random_forest_day1_balanced.pkl
│   ├── random_forest_best.pkl
│   └── xgboost_day2.pkl
├── Preview.png 
├── ImpactSense.ipynb
├── random_forest_tuned.pkl
├── xgboost_tuned.pkl
├── xgboost_tuned_weighted.pkl
├── model_comparison_summary.csv
├── model_evaluation_summary.csv
├── xgboost_best.pkl
└── week5_model_evaluation_summary.csv
```

## ⚙️ Modules Implemented  

### 🧹 1. Data Exploration & Cleaning  
- Load raw datasets  
- Handle missing values, duplicates, and anomalies  
- Visualize earthquake frequency, magnitude, and depth distribution  

### 🧩 2. Feature Engineering  
- Scaling and normalization  
- Geospatial clustering and encoding  
- Feature extraction from soil and seismic attributes  

### 🤖 3. Model Development  
- Algorithms used: **Logistic Regression**, **Decision Tree**, **Random Forest**, **XGBoost**  
- Hyperparameter tuning for optimized performance  

### 📊 4. Model Evaluation  
- **Classification metrics:** Accuracy, Precision, Recall, F1-Score  
- **Regression metrics:** MAE, MSE, R²  
- **Visualization:** Confusion matrix, feature importance, SHAP plots  

### 💻 5. User Interface (Optional)  
- Built with **Gradio** or **Streamlit**  
- **Input:** Magnitude, depth, region, soil type  
- **Output:** Predicted risk category or impact level  

---

## 📈 Model Performance Metrics  

| Metric | Description |
|:--------|:-------------|
| **Accuracy** | Correct classification percentage |
| **Precision** | Reliability of positive predictions |
| **Recall** | Coverage of actual high-risk cases |
| **F1-Score** | Balance between precision & recall |
| **MAE/MSE** | Average prediction error (for regression) |
| **R² Score** | Model explanatory power |

---

## 🧠 Key Outcomes  

✅ Built and tuned ML models for earthquake impact prediction  
✅ Achieved explainability via feature importance and SHAP  
✅ Developed reproducible code in Jupyter Notebook  
✅ Optional prototype UI for real-time prediction  

---

## 🧩 Datasets Used  

- **Significant_Earthquakes.csv** – Core dataset (magnitude, depth, lat, long, region)  
- **SoilData.doc / DSMW Files** – Soil and geological properties  
- **WORLD764.xls / SU_Info.xls** – Supplementary location and seismic zone data  

**Source:** Kaggle and publicly available geospatial datasets  

---


## 📚 References  

- [📘 Kaggle Earthquake Dataset](https://www.kaggle.com)  
- [🌐 USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/)  
- [🗺️ DSMW (Digital Soil Map of the World)](https://data.apps.fao.org/map/catalog/srv/eng/catalog.search#/metadata/22b99b60-4b4a-11db-b8c6-000d939bc5d8)  

---

## 🪶 Acknowledgement  

This project was developed as part of the **Infosys Springboard Internship Program**.  

I would like to express my sincere gratitude to my **Spring Mentor** for their constant guidance, support, and valuable feedback throughout the project.  

Their mentorship played a crucial role in shaping the development of **ImpactSense**, helping me gain hands-on experience in **machine learning**, **data analysis**, and **real-world problem-solving** in the domain of **disaster management and geospatial analytics**.  

---

---

## 👨‍💻 Developed By  

**✨ Sarthak Singh**  
💼 *Project completed during internship at Infosys Springboard*  

🔗 **Connect with me:**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sarthak%20Singh-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sarthak-singh-cse/)  

---



