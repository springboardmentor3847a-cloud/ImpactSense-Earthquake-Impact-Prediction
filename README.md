# ImpactSense-Earthquake-Impact-Prediction

Project Description.
This project aims to predict the impact severity of earthquakes using machine learning models.
The workflow is divided into 4 structured milestones (8 weeks) covering dataset exploration, preprocessing, model development, evaluation, UI creation, testing, and final documentation.
The final outcome is a predictive system that can estimate earthquake impact levels based on geospatial and seismic features.
A Gradio-based web interface is included for real-time earthquake impact prediction.

Instructions to set up and run the project.
Install Dependencies
If you have a requirements.txt:
pip install -r requirements.txt

Common libraries used:
pandas, numpy
matplotlib, seaborn, plotly
scikit-learn
xgboost, lightgbm (optional)
shap

Run the Notebooks
Open Jupyter/Colab and run the notebook for each milestone.

Milestone-wise Deliverables & Insights

Milestone 1 – Data Understanding & Preprocessing
Week 1: Dataset Exploration
Loaded dataset
Checked missing values and data types
Visualized distributions
Mapped earthquake locations
Insight: Several features were skewed; spatial clusters observed.

Week 2: Preprocessing & Feature Engineering
Handled missing values
Normalized/standardized features
Created derived features (e.g., depth bin, soil category)
Encoded categorical variables
Insight: Feature engineering improved baseline performance.

Milestone 2 – Model Development
Week 3: Baseline Models
Logistic Regression
Decision Tree Classifier
Train-test evaluation
Basic accuracy/MAE results
Insight: Decision Tree performed better than Logistic Regression.

Week 4: Advanced Models
Random Forest
Gradient Boosting / XGBoost
Hyperparameter tuning (GridSearch/RandomizedSearch)
Insight: Random Forest achieved the best overall metrics.

Milestone 3 – Evaluation & UI
Week 5: Model Evaluation & Explainability
Confusion matrix
MAE / MSE / RMSE
ROC-AUC (for classification)
SHAP values for interpretability
Feature importance charts
Insight: Magnitude and Depth were the most influential features.

Week 6: Gradio UI Prototype
Built a fully interactive prediction interface using Gradio
Inputs: Magnitude, Depth, Soil Type, Region Cluster

Outputs:
Predicted damage level
Probability bar chart
Insight: The UI enables non-technical users to interact with the model easily.

Milestone 4 – Testing & Finalization
Week 7: Testing & Refinements
Tested edge cases (extreme magnitude, invalid numbers, empty inputs)
Added safe input handling
Improved UI design and prediction reliability
Insight: Added validation reduced incorrect predictions.

Week 8: Final Report & Presentation
Prepared final performance charts
Included SHAP summary plots
Generated PDF report and slides
Insight: Clear visualization improved the storytelling in presentation.

Results Summary
Best Model: Random Forest
Top Contributing Features: Magnitude, Depth, SoilType, RegionCluster
Use Case: Disaster preparedness + early impact estimate tool
