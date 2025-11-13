###ImpactSense – Earthquake Impact Prediction

ImpactSense: AI-Powered Earthquake Risk Analyzer

ImpactSense is an end-to-end data science and full-stack project that predicts and visualizes the potential impact of earthquakes.
It integrates live data pipelines, machine learning models, explainable AI (XAI), and a web-based interface for real-time risk assessment.

The project demonstrates a complete MLOps-style workflow — from data preprocessing and model training to explainability and deployment using a React + Express web app.

Project Overview

ImpactSense analyzes earthquake characteristics such as magnitude, depth, and population density to predict their likely impact.
It uses advanced feature engineering and explainability techniques to ensure transparent predictions.

The system is composed of:

Machine Learning Pipeline: Built using Python and scikit-learn.

Explainability Tools: SHAP for feature contribution insights.

Web Interface: React frontend and Express backend API for real-time predictions.

Milestone 1: Data Preparation and Preprocessing

This milestone focused on creating a reliable dataset for model training and validation.

Key Steps

Fetched or simulated earthquake data including Magnitude, Depth, Latitude, Longitude, Population Density, and Soil Type.

Engineered an ImpactScore feature capturing interactions between parameters.

Applied preprocessing including normalization and handling missing values.

Conducted exploratory data analysis (EDA) to understand feature distributions and correlations.

Milestone 2: Model Training and Evaluation

This milestone developed a predictive model to estimate earthquake impact.

Approach

Split the dataset into training (80%) and testing (20%) sets.

Implemented a Random Forest Regressor for prediction.

Computed evaluation metrics as follows:

Mean Absolute Error (MAE): 3.401

Mean Squared Error (MSE): 18.569

R² Score: 0.985

Analyzed predicted vs actual ImpactScores using histograms.

Key Insights

The Random Forest model achieved high accuracy (R² ≈ 0.98).

Feature engineering (Depth influence and Population Density weighting) improved performance.

Thresholds for risk classification (Low, Medium, High) were derived using percentile-based thresholds from training data.

Milestone 3: Explainability and Web Interface Integration

This milestone combined explainable AI with frontend-backend integration to make model insights accessible through a user interface.

Model Explainability

Used SHAP (SHapley Additive exPlanations) to interpret model predictions.

Generated summary and force plots showing how each feature (Magnitude, Depth, Population Density, Soil Type) contributes to impact prediction.

Created a confusion matrix and classification report to evaluate categorical accuracy.

Model maintained over 93% classification accuracy, with balanced precision and recall across all risk levels.

Web Application Integration

Developed a full-stack setup with:

Backend (Express.js): POST /api/predict endpoint returning impact score and risk category.

Frontend (React): Interactive form for user input.

Real-time communication between frontend and backend to display prediction results.

Users can:

Input parameters such as Magnitude, Depth, Soil Type, Population Density, Latitude, and Longitude.

Use preset scenarios for Urban and Rural earthquake simulations.

View predicted risk levels instantly on submission.

Milestone 4: Results and Visual Outputs

This milestone presented the final results and visualizations generated through the model and UI.

Key Outcomes

SHAP analysis confirmed Magnitude and Depth as dominant predictors.

Confusion matrix and classification reports demonstrated model consistency.

Histograms and bar plots visualized error metrics (MAE, MSE) and score distributions.

The web interface displayed accurate and responsive predictions.

Screenshots captured UI forms for urban/rural test cases, output panels, and visual maps showing predicted impact zones.

These outcomes validate the system’s usability and interpretability, confirming its real-world potential.

Risk Level Computation

The backend uses a scoring formula to assess risk:
Score = Magnitude Factor × Depth Factor × Soil Factor × Population Factor

Risk Classification

0–34: Low risk (minimal surface damage)

35–69: Medium risk (moderate infrastructure effects)

70–100: High risk (severe urban damage potential)

How to Run the Project
Prerequisites

Make sure you have the following installed:

Node.js and npm

Python 3.x

Git

A code editor (e.g., VS Code)

1. Clone the Repository
git clone https://github.com/yourusername/ImpactSense-Earthquake-Impact-Prediction.git
cd ImpactSense-Earthquake-Impact-Prediction

2. Backend Setup (Express API)
cd backend
npm install
npm start


The backend will start on http://localhost:5000

3. Frontend Setup (React App)

In a new terminal:

cd frontend
npm install
npm start


The frontend will run on http://localhost:3000

4. Running the Model Notebook

You can explore or retrain the model using Jupyter Notebook:

jupyter notebook


Then open the file ImpactSense_Model.ipynb (or your milestone notebooks) and execute the cells.

5. Access the Web App

Once both servers are running:
Open http://localhost:3000
 in your browser and enter parameters to get real-time earthquake impact predictions.

Conclusion

ImpactSense demonstrates the full lifecycle of a modern AI project — from data creation and model training to explainability and deployment.
It highlights how machine learning and explainable AI can produce transparent, actionable earthquake risk assessments.

Future Enhancements

Integration with live seismic APIs for real-time updates.

Enhanced geospatial visualizations for risk mapping.

Automated retraining pipelines to keep models updated.
