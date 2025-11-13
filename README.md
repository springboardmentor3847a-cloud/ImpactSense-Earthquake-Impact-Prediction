# ImpactSense-Earthquake-Impact-Prediction
ImpactSense: AI-Powered Earthquake Risk Analyzer

ImpactSense is an end-to-end data science and full-stack project that predicts and visualizes the potential impact of earthquakes. It integrates live data pipelines, machine learning models, explainable AI (XAI), and a web-based interface for real-time risk assessment.

The project demonstrates a complete MLOps-style workflow, from data preprocessing and model training to explainability and deployment via a React + Express web app.

Project Overview

ImpactSense is designed to analyze earthquake characteristics such as magnitude, depth, and population density to predict their likely impact. The project uses advanced feature engineering and explainability techniques to ensure transparent predictions.

The system is composed of:

   Machine Learning Pipeline: Built using Python and scikit-learn.

   Explainability Tools: SHAP for feature contribution insights.

   Web Interface: React-based frontend with an Express backend API for real-time predictions.

Milestone 1: Data Preparation and Preprocessing

This milestone focused on creating a reliable dataset for model training and validation.

Key Steps:

Fetched or simulated earthquake data including parameters like Magnitude, Depth, Latitude, Longitude, Population Density, and Soil Type.

Engineered an ImpactScore feature that captures the interaction between these parameters.

Handled data preprocessing tasks including normalization and handling missing values.

Conducted exploratory data analysis (EDA) to understand feature distributions and correlations.

Milestone 2: Model Training and Evaluation

This milestone covered the development of a predictive model to estimate earthquake impact based on multiple input features.

Approach:

Split the data into training and testing sets.

Implemented a Random Forest Regressor trained on 80% of the data.

Computed evaluation metrics including:

Mean Absolute Error (MAE): 3.401

Mean Squared Error (MSE): 18.569

R² Score: 0.985

Created histograms to analyze the distribution of predicted vs actual ImpactScores.

Key Insights:

The Random Forest model achieved high accuracy (R² ≈ 0.98).

Feature engineering (Depth influence and Population Density weighting) significantly improved performance.

Thresholds for classifying risk levels (Low, Medium, High) were derived using percentile-based thresholds from training data.

Milestone 3: Explainability and Model Interpretation

This milestone focused on enhancing the interpretability of the model.

Tools Used:

SHAP (SHapley Additive exPlanations): To measure feature importance and visualize model reasoning.

Generated summary and force plots showing how each input (magnitude, depth, population density) affects the prediction.

Derived class-based performance metrics using a confusion matrix and classification report.

Results:

Model maintained over 93% classification accuracy when converting regression outputs to categorical risk levels.

Precision and recall were above 0.90 across all classes (Low, Medium, High).

SHAP analysis confirmed that Magnitude and Depth were the dominant predictors of impact severity.

Milestone 4: UI Development and Full-Stack Integration

This milestone implemented a complete frontend-backend architecture to make predictions accessible via a user-friendly interface.

Backend (Express.js):

Endpoint: POST /api/predict

Inputs: Magnitude, Depth, Soil Type, Population Density, Latitude, Longitude

Logic: Calculates an impact score (0–100) using a scaled function of all parameters.

Returns: A JSON response { "score": value, "risk": "Low | Medium | High" }

Frontend (React):

Component: ImpactForm.js

Features:

User input fields for all model parameters

Validation for realistic ranges (e.g., magnitude 0–10, depth ≥ 0)

Preset buttons for quick “Urban” and “Rural” test scenarios

Displays predicted risk level and visual feedback on submission

The web interface communicates with the backend in real-time and displays model predictions dynamically.

Risk Level Computation

The backend uses a scoring formula to assess risk:

Score = Magnitude Factor × Depth Factor × Soil Factor × Population Factor

Risk Classification:

0–34: Low risk (minimal surface damage)

35–69: Medium risk (moderate infrastructure effects)

70–100: High risk (severe urban damage potential)

Model Explainability and Insights

The project integrates SHAP explainability to ensure the predictions are transparent and justifiable.

Top Influencing Factors:

Magnitude: Directly increases the impact score.

Depth: Shallow quakes amplify risk.

Population Density: Indicates human exposure.

Soil Type: Modulates ground shaking intensity.

Visualization tools (bar charts, histograms, SHAP plots) make it easy to understand model decisions.
