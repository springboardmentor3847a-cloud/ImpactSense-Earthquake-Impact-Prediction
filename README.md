# ImpactSense-Earthquake-Impact-Prediction
🌍 ImpactSense – Earthquake Impact Prediction
📘 Project Overview

ImpactSense is a machine learning-based project designed to predict the impact or severity of earthquakes using geophysical and environmental data such as latitude, longitude, depth, and magnitude.
The system helps identify high-risk zones and supports urban planning, emergency response, and disaster preparedness through data-driven insights.

🎯 Objectives

Predict earthquake impact or risk zone using seismic data.

Train and evaluate machine learning models for classification and regression.

Visualize seismic patterns and model results.

(Optional) Deploy a simple Streamlit-based UI for user input and prediction.

🧩 Use Cases
Use Case	Description	Example
Urban Risk Assessment	Predict earthquake impact in densely populated areas.	Identify which regions face higher risk during a 5.5 magnitude event.
Infrastructure Planning	Help planners enforce building safety policies in risky zones.	Predict risk level based on soil density and proximity to fault lines.
Disaster Response	Support emergency services in prioritizing aid.	Rank locations for rescue response after an earthquake.
🧠 System Architecture

Data Collection & Understanding – Load and explore dataset (Kaggle source).

Preprocessing & Feature Engineering – Clean missing data, scale values, and encode categorical variables.

Model Development – Build predictive models (Logistic Regression, Decision Tree, Random Forest, XGBoost).

Evaluation & Visualization – Use metrics like Accuracy, F1-score, MAE, MSE, and feature importance plots.

(Optional) UI Deployment – Create a Streamlit app to predict risk levels interactively.

📊 Dataset

Source: Kaggle

Features: Latitude, Longitude, Depth, Magnitude, Geological Parameters, etc.

Target: Earthquake impact or risk category.

⚙️ Technologies Used

Python 3.x

Libraries:
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, streamlit (optional)

Tools: Jupyter Notebook / Google Colab

📈 Model Evaluation
Classification Metrics

Accuracy – Overall prediction correctness

Precision & Recall – Balance between predicted and actual high-risk cases

F1-Score – Harmonic mean of precision and recall

Confusion Matrix – Visualization of correct vs. misclassified cases

Regression Metrics (if applicable)

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

R² Score (Coefficient of Determination)

📅 Implementation Timeline
Week	Task	Deliverables
1	Project setup, data loading	Dataset understanding
2	Preprocessing & feature engineering	Cleaned dataset
3	Baseline model training	Logistic Regression, Decision Tree
4	Advanced models	Random Forest, XGBoost
5	Evaluation & explainability	Metrics, SHAP plots
6	UI prototype	Streamlit/FastAPI interface
7–8	Final testing & report	Visuals, documentation, presentation
🚀 How to Run
# Clone the repository
git clone https://github.com/<your-username>/ImpactSense.git

# Navigate to project folder
cd ImpactSense

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook "earthquake_project final.ipynb"
🏁 Future Enhancements
Integrate real-time earthquake data APIs.

Improve prediction accuracy using deep learning models.

Expand model to global data coverage.

Deploy fully interactive web dashboard.
# (Optional) Run Streamlit app
streamlit run app.py

🧾 Results

Successfully predicted earthquake impact zones based on input features.

Demonstrated strong correlation between magnitude, depth, and risk levels.

Model performance visualized using confusion matrix and feature importance charts.

