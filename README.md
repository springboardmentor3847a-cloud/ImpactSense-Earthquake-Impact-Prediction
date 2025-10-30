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
