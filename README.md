# 🌍 Earthquake Impact Classifier & Response App

This project is a complete end-to-end machine learning application that predicts the potential impact of an earthquake—classifying it as **Light, Moderate, Strong, or Major**.

It goes beyond simple prediction to provide a fully-interactive web application built with Streamlit. The app gives users actionable, life-saving information, including **dynamic safety checklists** and **geo-located emergency helplines** based on the event's coordinates.



## 🚀 Key Features

* **Impact Classification:** Uses a fine-tuned **XGBoost** model to classify earthquake severity.
* **Geospatial Data Enrichment:** Enriches earthquake data with local **soil type** information for more accurate predictions.
* **Dynamic Response UI:** The Streamlit front-end is more than just a form; it provides a full response kit:
    * **Color-Coded Results:** Instantly see the severity with professional green, yellow, orange, and red headers.
    * **Actionable Checklists:** Dynamically generates safety information for what to do *before*, *during*, and *after* the event.
    * **Geo-Located Helplines:** Automatically performs reverse-geocoding to provide local emergency numbers (e.g., 911 in the US, 112 in India) for the event's location.
* **Professional UI/UX:** Features a custom-themed UI with a dark sidebar, modern "card" elements, and a clean, responsive layout.

---

## 🛠️ Project Workflow & Technical Deep-Dive

This project is split into two main components: the **Machine Learning Pipeline** (the "brain") and the **Streamlit App** (the "face").

### 1. The Data & Preprocessing Pipeline

Our goal is to build a robust dataset that includes not just *what* the earthquake is, but *where* it is and *what* it's on.

* **Data Sources:** We combine two datasets for a comprehensive history:
    1.  **USGS Earthquake API:** Fetches recent (2023) earthquake data.
    2.  **Historical CSV:** A large local dataset (`database.csv`) provides a deep historical context.
* **Geospatial Enrichment (The "Secret Sauce"):**
    * We use **`geopandas`** to perform a **spatial join**.
    * Each earthquake (a point) is mapped onto the **Digital Soil Map of the World (DSMW)** shapefile (a set of polygons).
    * This critical step adds the `soil_type` feature to our dataset, a key predictor of how much the ground will shake (liquefaction).
* **Feature Engineering:**
    * **Target Variable:** The continuous `magnitude` is binned into four categorical classes to frame this as a classification problem: **Light, Moderate, Strong, and Major**.
    * **Interaction Features:** We create new features like `depth_mag_interaction` to help the model find complex patterns (e.g., a shallow, strong quake is different from a deep, strong quake).
* **Handling Imbalance:**
    * Earthquake data is naturally imbalanced (many `Light` events, very few `Major` events).
    * We use **SMOTE (Synthetic Minority Over-sampling Technique)** on the *training data only*. This synthetically creates new examples of the rare classes, forcing the model to learn their patterns.

### 2. Model Training & Selection

We held a "competition" to find the best model for the job.

* **Candidate Models:** Logistic Regression, Random Forest, and XGBoost.
* **Hyperparameter Tuning:** We used `GridSearchCV` to systematically test different combinations of settings (e.g., `n_estimators`, `max_depth`) for each model to find its peak performance.
* **Best Model:** The **XGBoost Classifier** was the clear winner. It provided the best balance of accuracy and, most importantly, a high F1-score, meaning it was much better at correctly identifying the rare but critical `Strong` and `Major` classes.

### 3. The Final ML Pipeline

For a professional and reproducible result, the entire process is bundled.

1.  A `ColumnTransformer` is created to automatically apply `StandardScaler` to numeric features and `OneHotEncoder` to categorical features (`soil_type`).
2.  This "preprocessor" is bundled with the best-tuned XGBoost model into a single `scikit-learn Pipeline` object.
3.  This pipeline, along with the `LabelEncoder` (for turning "Strong" into `2`), is saved to disk using `joblib` (`earthquake_pipeline.joblib`).

### 4. The Streamlit Web Application (UI)

The UI's job is to be a user-friendly "face" for the powerful pipeline.

* **Loading:** On startup, the app loads the `earthquake_pipeline.joblib` and `label_encoder.joblib` files using `@st.cache_resource`. This loads the model into memory *once* for high-speed predictions.
* **User Input:** A clean, two-column form (styled as a "card" with custom CSS) captures the user's inputs.
* **Prediction & Response:** When the user clicks "Classify":
    1.  The app runs the prediction using the loaded `pipeline.predict()` method.
    2.  It uses `reverse_geocoder` to find the country for the given latitude and longitude. *Note: This may take a moment on the first run as it loads its local geo-database.*
    3.  A Python dictionary (`EMERGENCY_NUMBERS`) maps the country code (e.g., "IN") to its specific helplines (e.g., "112").
    4.  The entire UI dynamically updates to show the **color-coded result** and the four-tab response section, now populated with both generic safety info and the **specific, geo-located emergency numbers**.

---

## 💻 Technology Stack

* **Data Science:** Pandas, NumPy, Scikit-learn, XGBoost, Geopandas, Imbalanced-learn (SMOTE)
* **Web Application:** Streamlit
* **Utilities:** Joblib, Requests, Reverse-Geocoder

---

## ⚙️ How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/earthquake-classifier.git](https://github.com/your-username/earthquake-classifier.git)
    cd earthquake-classifier
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install streamlit pandas numpy scikit-learn xgboost geopandas imblearn joblib requests reverse-geocoder
    ```

4.  **Download Geospatial Data:**
    * This project requires the **Digital Soil Map of the World (DSMW)** shapefile.
    * You must download it manually (e.g., from a source like the FAO) and place the `.shp` file and its companion files (`.dbf`, `.shx`, etc.) in your project.
    * Update the `SHAPEFILE_PATH` variable in the `training_script.py` to point to your downloaded `.shp` file.
    * You will also need the historical earthquake CSV (`database.csv`).

5.  **Run the Training Pipeline:**
    * Run the training script (`Milestone_2.ipynb`) to fetch data, process it, train the model, and save the `.joblib` files.
    * This only needs to be done once.
    ```bash
    python training_script.py 
    ```
    * This will create `earthquake_pipeline.joblib` and `label_encoder.joblib`.

6.  **Run the Streamlit App:**
    * Make sure your UI file is named correctly, as you mentioned `Milestone_3_UI.py`.
    ```bash
    streamlit run Milestone_3_UI.py
    ```
    Your application will now be running and accessible in your browser!
   ---
   ## 📊 Dataset Links

This project relies on publicly available, high-quality data:

* **USGS Earthquake Hazards Program:** Real-time data is fetched from the [USGS FDSN Event Web Service API](https://earthquake.usgs.gov/fdsnws/event/1/).
* **Historical Earthquake Database:** The `database.csv` file is based on the [Significant Earthquakes, 1900-Present Dataset](https://www.kaggle.com/datasets/usamabuttar/significant-earthquakes) available on Kaggle, which is also sourced from the USGS.
* **Digital Soil Map of the World (DSMW):** The geospatial `soil_type` data is sourced from the [FAO/UNESCO Digital Soil Map of the World](https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/faounesco-soil-map-of-the-world/en/).
---
## 🙏 Acknowledgements

* This project would not be possible without the invaluable, open-access data provided by the **U.S. Geological Survey (USGS)** and the **Food and Agriculture Organization (FAO) of the United Nations**.
* A special thanks to the maintainers of the **Scikit-learn**, **XGBoost**, **Streamlit**, and **Pandas** libraries, whose open-source tools form the backbone of this application.
