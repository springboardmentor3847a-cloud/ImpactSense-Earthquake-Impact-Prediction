# ImpactSense-Earthquake-Impact-Prediction

ImpactSense is an end-to-end data science application designed to classify earthquake severity using geophysical parameters.  
The project showcases a complete ML workflow—data preprocessing, model development, evaluation, and deployment—implemented using Logistic Regression and Decision Tree Classifier.  
An interactive Gradio UI is included for real-time predictions. 

---

# Key Features

**Structured Data Pipeline:**  
Processes earthquake-related features, handles missing values, encodes labels, and prepares data for ML modeling.

**Dual ML Models:**  
Implements **Logistic Regression** and **Decision Tree Classifier** for severity classification, allowing comparison between a linear and a non-linear model.

**Balanced & Clean Dataset:**  
Includes preprocessing, feature scaling (if needed), and clean splitting into train/test sets.

**Interactive Gradio Interface:**  
The UI allows users to input parameters and get:
- Predicted severity level  
- Model confidence (probabilities)  
- SHAP-based interpretability dashboard  

**Model Explainability:**  
Uses SHAP to visually explain which features influenced each prediction.

---

# Project Workflow (Milestones)

This project follows a structured milestone-based workflow, implemented in the notebook **Milestone 1 – 4.ipynb**.

---

## **Milestone 1: Project Setup & Preprocessing**

- Loaded the dataset and performed initial cleaning  
- Handled missing values and applied encoding  
- Split data into features (`X`) and target (`y`)  
- Scaled/normalized features (if applicable)  
- Conducted basic exploratory analysis  

---

## **Milestone 2: Model Development**

- Implemented **Logistic Regression** and **Decision Tree Classifier**  
- Trained both models on preprocessed data  
- Tuned hyperparameters for optimal results  
- Compared baseline vs improved performance  

---

## **Milestone 3: Evaluation & Visualization**

- Generated **Classification Report** (Accuracy, Precision, Recall, F1-score)  
- Visualized **Confusion Matrix**  
- Extracted **Feature Importance** (Decision Tree)  
- Selected the most stable model for deployment
- Fine tuning is done using XGBoost.

---

## **Milestone 4: Final UI & Deployment**

- Built a complete **Gradio UI application**  
- Integrated trained model + label encoder   
- Finalized project files for GitHub submission  

---

Dataset Link - https://drive.google.com/drive/folders/1fg53n4tOzIMRJc0NBE_5PZby0EJYn7AO
API - https://earthquake.usgs.gov/fdsnws/event/1/query

## How to Use the Earthquake Severity Prediction System

1. **Open the Application**  
   Click on the provided link to launch the Earthquake Severity Prediction UI.

2. **Set the Input Parameters**  
   Adjust the values shown on the screen:
   - **Magnitude (Richter Scale):** Move the slider or type a value.
   - **Depth (km):** Use the slider to choose the earthquake depth.
   - **Soil Type (Encoded):** Select the soil type from the dropdown.
   - **Region Cluster (Encoded):** Choose the region cluster value.

3. **Click on "🔎 Predict Severity"**  
   After entering all parameters, press the **Predict Severity** button.

4. **View the Prediction**  
   The result will appear in the **Prediction Result** section below, showing the predicted earthquake severity (e.g., *Low*, *Moderate*, *High*).

5. **Interpret the Output**  
   The prediction is based on the trained machine learning models (Logistic Regression / Decision Tree / XGBoost).

