ImpactSense – Earthquake Impact Prediction

ImpactSense: AI-Powered Earthquake Risk Analyzer

ImpactSense is an end-to-end data science and full-stack project that predicts and visualizes the potential impact of earthquakes.
It integrates live data pipelines, machine learning models, explainable AI (XAI), and a web-based interface for real-time risk assessment.

The project demonstrates a complete MLOps-style workflow — from data preprocessing and model training to explainability and deployment using a React + Express web app.

Project Overview

ImpactSense analyzes earthquake characteristics such as magnitude, depth, and population density to predict their likely impact.
It uses advanced feature engineering and explainability techniques to ensure transparent predictions.

System Components:

Machine Learning Pipeline built using Python and scikit-learn

Explainability Tools using SHAP for feature contribution insights

Web Interface developed with React frontend and Express backend API for real-time predictions

Milestone 1: Data Preparation and Preprocessing

*This milestone focused on creating a reliable dataset for model training and validation.

Key Steps:

Fetched or simulated earthquake data including Magnitude, Depth, Latitude, Longitude, Population Density, and Soil Type

Engineered an ImpactScore feature capturing interactions between parameters

Applied preprocessing including normalization and missing value handling

Conducted exploratory data analysis (EDA) to understand distributions and correlations

Milestone 2: Model Training and Evaluation

This milestone developed a predictive model to estimate earthquake impact.

Approach:

Split data into training (80%) and testing (20%) sets

Implemented a Random Forest Regressor

Computed evaluation metrics:

Mean Absolute Error (MAE): 3.401

Mean Squared Error (MSE): 18.569

R² Score: 0.985

Analyzed predicted vs actual ImpactScores using histograms

Key Insights:

Achieved high model accuracy (R² ≈ 0.98)

Feature engineering (Depth influence and Population Density weighting) improved performance

Derived percentile-based thresholds for risk classification (Low, Medium, High)

Milestone 3: Explainability and Model Interpretation

This milestone enhanced the interpretability of the model using explainable AI.

Tools Used:

SHAP (SHapley Additive exPlanations) to measure feature importance

Generated summary and force plots showing influence of magnitude, depth, and population density

Created classification metrics such as confusion matrix and report

Results:

Model maintained over 93% classification accuracy when mapping regression outputs to risk categories

Precision and recall above 0.90 for all classes (Low, Medium, High)

SHAP confirmed Magnitude and Depth as dominant predictors

Milestone 4: UI Development and Full-Stack Integration

Implemented a complete frontend-backend system for real-time predictions.

Backend (Express.js)

Endpoint: POST /api/predict

Inputs: Magnitude, Depth, Soil Type, Population Density, Latitude, Longitude

Logic: Calculates an impact score (0–100) using a scaled formula

Returns: { "score": value, "risk": "Low" | "Medium" | "High" }

Frontend (React)

Component: ImpactForm.js

Features:

Input fields for all parameters

Validations for realistic ranges (magnitude 0–10, depth ≥ 0)

Preset buttons for quick “Urban” and “Rural” test cases

Displays predicted risk level dynamically

Risk Level Computation

The backend computes risk using the formula:
Score = Magnitude Factor × Depth Factor × Soil Factor × Population Factor

Risk Classification:

0–34 → Low (minimal surface damage)

35–69 → Medium (moderate infrastructure effects)

70–100 → High (severe urban damage potential)

Model Explainability and Insights

Top Influencing Factors:

Magnitude: Directly increases impact score

Depth: Shallow earthquakes amplify risk

Population Density: Reflects human and infrastructure exposure

Soil Type: Affects ground shaking intensity

Visualizations such as bar charts, histograms, and SHAP plots were used to interpret and justify predictions.

Conclusion

ImpactSense demonstrates the full lifecycle of a modern AI project — from data creation and model training to explainability and deployment.
It highlights how data-driven modeling and XAI contribute to transparent and actionable earthquake risk assessments.

Future improvements may include:

Integration with live seismic APIs

Enhanced geospatial visualization

Automated retraining pipelines for continuous real-time analysis
