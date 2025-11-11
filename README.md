# ImpactSense: AI-Powered Earthquake Risk Analyzer


ImpactSense is an end-to-end data science application that demonstrates a complete MLOps workflow. It fetches live data from the USGS, uses a high-accuracy (90%+) machine learning model to predict earthquake impact, and serves these insights through an interactive Streamlit dashboard. The application is enhanced with a Generative AI assistant (Mistral-7B) to provide human-readable situation reports and an interactive chatbot for disaster response advice.



#  Key Features


**Live Data Pipeline:** Fetches real-time earthquake data (Magnitude 2.5+) from the USGS API in robust, monthly batches.

**High-Accuracy ML Model:** Utilizes a Stacking Classifier (combining XGBoost and Random Forest) trained on balanced data (using SMOTE) to achieve >90% accuracy in predicting risk.

**Advanced Feature Engineering:** Enriches raw data by:

    Deriving a is_populated feature using the Geopy library.

    Engineering a synthetic impact_score to create a realistic, class-balanced training target.

**Dynamic UI Dashboard:** Built with Streamlit, the UI features:

    An interactive map to visualize the event's epicenter.

    A model confidence chart showing the probability for each risk level.

**AI-Powered Analyst:** Uses a Mistral-7B LLM (via OpenRouter) to:

    Generate a clean, point-by-point situation report for each event.

    Identify the human-readable location name from raw coordinates.

**Conversational AI Assistant:** A fully interactive chatbot that uses the event's context to answer follow-up questions about safety, aftershocks, and tsunami risks.




# Project Workflow (Milestones)


This project was built following a structured plan. The main.ipynb notebook reflects this workflow:

**Milestone 1: Project Setup & Preprocessing**

    Project setup, library imports, and robust data fetching from the USGS API.

    Preprocessing, imputation, and advanced feature/target engineering (creating is_populated, impact_score, and balanced risk_level classes).

**Milestone 2: Model Training**

    Splitting data and encoding labels (LabelEncoder).

    Balancing the training data using SMOTE to handle class imbalance.

    Implementing and training an advanced Stacking Classifier (XGBoost + Random Forest) to maximize accuracy.

**Milestone 3: Evaluation & UI Prototyping**

    Evaluating the model with a Classification Report and Confusion Matrix, achieving >90% accuracy.

    Generating a Feature Importance plot to explain model decisions.

    Saving the final model (clf), encoder (le), and feature list (features) for the Streamlit app.

**Milestone 4: Final Product & Deployment**

    The app.py file represents the final, tested UI.

    This README.md and the GitHub repository complete the final reporting requirement.