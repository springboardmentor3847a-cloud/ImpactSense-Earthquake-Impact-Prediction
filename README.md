ImpactSense – Earthquake Impact Prediction
ImpactSense: AI-Powered Earthquake Risk Analyzer

ImpactSense is an end-to-end data science and full-stack project that predicts and visualizes the potential impact of earthquakes.
It integrates live data pipelines, machine learning models, explainable AI (XAI), and a web-based interface for real-time risk assessment.

The project demonstrates a complete MLOps-style workflow — from data preprocessing and model training to explainability and deployment using a React + Express web app.

Project Overview

ImpactSense analyzes earthquake characteristics such as magnitude, depth, and population density to predict their likely impact.
It uses advanced feature engineering and explainability techniques to ensure transparent predictions.

The system is composed of:
• Machine Learning Pipeline built using Python and scikit-learn.
• Explainability Tools using SHAP for feature contribution insights.
• Web Interface developed with React frontend and Express backend API for real-time predictions.

Milestone 1: Data Preparation and Preprocessing

This milestone focused on creating a reliable dataset for model training and validation.

Key Steps:
• Fetched or simulated earthquake data including Magnitude, Depth, Latitude, Longitude, Population Density, and Soil Type.
• Engineered an ImpactScore feature capturing interactions between parameters.
• Applied preprocessing including normalization and handling missing values.
• Conducted exploratory data analysis (EDA) to understand feature distributions and correlations.

Milestone 2: Model Training and Evaluation

This milestone developed a predictive model to estimate earthquake impact.

Approach:
• Split the dataset into training (80%) and testing (20%) sets.
• Implemented a Random Forest Regressor for prediction.
• Computed evaluation metrics as follows:

Mean Absolute Error (MAE): 3.401

Mean Squared Error (MSE): 18.569

R² Score: 0.985
• Analyzed predicted vs actual ImpactScores using histograms.

Key Insights:
• The Random Forest model achieved high accuracy (R² ≈ 0.98).
• Feature engineering (Depth influence and Population Density weighting) improved performance.
• Thresholds for risk classification (Low, Medium, High) were derived using percentile-based thresholds from training data.

Milestone 3: Explainability and Web Interface Integration

This milestone combined explainable AI with frontend-backend integration to make model insights accessible through a user interface.

Model Explainability

• Used SHAP (SHapley Additive exPlanations) to interpret model predictions.
• Generated summary and force plots showing how each feature (magnitude, depth, population density, soil type) contributes to impact prediction.
• Created a confusion matrix and classification report to evaluate categorical accuracy.

Results showed over 93% classification accuracy with balanced precision and recall across all risk levels.

Web Application Integration

Developed a full-stack setup with:
• Backend (Express.js) exposing a POST /api/predict endpoint that returns impact score and risk category.
• Frontend (React) implementing an interactive form for user input.
• Real-time communication between frontend and backend to display prediction results.

The UI allowed users to:
• Input parameters such as Magnitude, Depth, Soil Type, Population Density, Latitude, and Longitude.
• Use preset scenarios for “Urban” and “Rural” earthquake simulations.
• View predicted risk levels (Low, Medium, High) instantly on submission.

Milestone 4: Results and Visual Outputs

This milestone presented the final results and visualizations generated through the model and user interface.

Key Outcomes:
• The SHAP analysis confirmed that Magnitude and Depth are the most influential features.
• Confusion matrix and classification reports demonstrated high model consistency.
• Histograms and bar plots visualized the error metrics (MAE, MSE) and score distributions.
• The web interface displayed accurate and responsive predictions for multiple scenarios.
• Screenshots captured the UI forms for urban and rural cases, prediction output panels, and visual maps showing predicted impact zones.

These visual outcomes validate the usability and interpretability of the system, confirming its real-world application potential.

Risk Level Computation

The backend uses a scoring formula to assess risk:
Score = Magnitude Factor × Depth Factor × Soil Factor × Population Factor

Risk Classification:
• 0–34 → Low risk (minimal surface damage)
• 35–69 → Medium risk (moderate infrastructure effects)
• 70–100 → High risk (severe urban damage potential)

How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/ImpactSense-Earthquake-Impact-Prediction.git
cd ImpactSense-Earthquake-Impact-Prediction

2. Run the Backend

Navigate to the backend folder and install dependencies:

cd server
npm install
node index.js


The backend will run on http://localhost:5050.

3. Run the Frontend

In a new terminal:

cd client
npm install
npm start


The React app will open on http://localhost:3000.

4. Run Jupyter Notebooks

To explore model training, evaluation, and explainability:

jupyter notebook


Open the notebooks for each milestone and run them in order.

Conclusion

ImpactSense demonstrates the complete lifecycle of a modern AI project — from data creation and model training to explainability and deployment.
It highlights how machine learning and explainable AI can produce transparent and actionable earthquake risk assessments.

Future improvements can include:
• Integration with live seismic APIs for real-time updates.
• Enhanced geospatial visualizations for risk mapping.
• Automated retraining pipelines to keep the model up to date.
