# 🌍 ImpactSense – Earthquake Magnitude Prediction (ML Pipeline)

This project builds a complete machine learning pipeline to predict earthquake magnitude using global earthquake data. The work is organized week-wise according to milestones, and all details from dataset analysis, preprocessing, modeling, evaluation, visualization, and explainability are included inside the weekly progress.

# 🗓 WEEK-WISE PROJECT DETAILS
# 📌 Milestone 1 — Week 1 & Week 2
# ✅ Week 1 — Project Setup & Dataset Understanding

During Week 1, the project began with loading and understanding the global earthquake dataset. The dataset used was earthquakes_2023_global.csv.
The initial steps included:

Viewing the first few rows of the dataset

Checking the dataset shape and structure

Reviewing column details and data types

Identifying missing values

Exploring unique earthquake event types

Unnecessary columns such as type, id, time, updated, place, and the magnitude column (which is the target variable) were removed.
The final target output of the model was set as Magnitude (mag), while the remaining numerical and categorical columns formed the feature set for training.

# ✅ Week 2 — Preprocessing & Feature Engineering

In Week 2, preprocessing and feature engineering tasks were completed.

A structured preprocessing system was created using pipelines.
Categorical columns were cleaned by filling missing values with the most frequent category and converting them into numerical form through one-hot encoding.
Numerical columns were cleaned using median imputation and scaled for uniformity.

All preprocessing steps were combined into a single transformation pipeline and applied to the training and testing datasets.
The dataset was then split into training and testing subsets to prepare for the modeling phase.

# 📌 Milestone 2 — Week 3 & Week 4
# ✅ Week 3 — Baseline Model Training

Week 3 focused on building multiple baseline regression models to compare performance.
The models trained and evaluated included:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Support Vector Regressor

The evaluation metric used was Mean Squared Error (MSE).
This comparison helped understand which model provided the best starting performance before any tuning.

# ✅ Week 4 — Advanced Modeling & Hyperparameter Tuning

In Week 4, the Random Forest model was selected for advanced training because of its strong baseline performance.

A detailed hyperparameter search was performed to identify the best model configuration.
The tuning involved adjusting the number of trees, maximum depth, feature selection method, and minimum samples required for splits.

After tuning, the optimized model was evaluated using:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

R-Squared Score (R²)

Root Mean Squared Error (RMSE)

A visualization comparing actual vs predicted magnitudes was generated.
The scatter plot included a diagonal reference line showing perfect predictions, making it easy to see how closely the model matched real earthquake magnitudes.

# 📌 Milestone 3 — Week 5
# ✅ Week 5 — Evaluation & Explainability

Week 5 focused on deeper evaluation and understanding of the model’s behavior.

A comparison chart was created to display the error values (MSE and MAE) of the best-performing Random Forest model.
This helped confirm the model’s reliability and performance consistency.

To understand why the model was making certain predictions, SHAP explainability was applied:

A SHAP TreeExplainer was created for the final model

SHAP values were generated to analyze feature contributions

A SHAP summary plot was produced to visually show which features influenced magnitude prediction the most

This step provided clear insights into which earthquake characteristics played the most important role in determining the magnitude value predicted by the model.

# 📌 Milestone 4 — (Not Included)

Weeks 6 to 8 (UI development, testing, deployment, final reporting) were not executed in the notebook, so they are intentionally excluded from this README.

# 🧪 Final Outcome of the Project

By the end of the executed work:

A cleaned and preprocessed dataset was prepared

Multiple baseline models were built and compared

A Random Forest model was fine-tuned for best performance

Evaluation metrics and visual comparisons were completed

SHAP explainability provided insights into feature importance

This completes a full machine learning pipeline for earthquake magnitude prediction, covering data preparation, modeling, evaluation, and interpretation.
