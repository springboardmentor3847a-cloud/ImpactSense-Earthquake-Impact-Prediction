# ImpactSense-Earthquake-Impact-Prediction
# Milestone 1
Week 1: Data Exploration and Cleaning
Loaded the dataset: Imported the earthquake data and previewed initial records to understand its structure.

Data inspection: Checked data types, column names, and summary statistics for all variables.

Missing value analysis: Identified columns with missing values and assessed their extent.

Duplicate removal: Confirmed there were zero duplicate records in the provided dataset.

Missing value treatment: Numeric columns with missing values were imputed using the median; categorical columns with minimal missingness were filled with the mode.

Datetime handling: Identified the correct time column and parsed it as a datetime object, tested for consistency, and extracted date/time features as needed.

Export: Saved the cleaned dataset as earthquake_data_clean.csv for further use.

Week 2: Feature Engineering and Preprocessing
Datetime feature extraction: Extracted year, month, day, hour, and minute from the parsed datetime column.

Categorical encoding: Converted categorical variables (such as Type, Source, Status, etc.) into numerical representations using Label Encoding.

Numerical feature scaling: Standardized all relevant numerical features to ensure consistent scaling for machine learning models.

Feature/target definition: Defined feature set (X) and target variable (y, e.g., Magnitude).

Train-test split: Split the dataset into training and test sets with an 80/20 ratio, readying the data for model development.
