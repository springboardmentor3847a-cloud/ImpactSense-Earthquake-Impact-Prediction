# Load unified dataset
import pandas as pd

df = pd.read_csv(r"C:\Users\sanka\Desktop\new peoject\unified_earthquake_data.csv")

print("Missing values per column before cleaning:")
print(df.isnull().sum())

# --- Strategy ---
# Numeric: fill with median
num_cols = ['Magnitude', 'Depth']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical: fill with mode
# Removed 'Location' as it's not present in the CSV according to previous cell output
cat_cols = ['DOMSOI']
for col in cat_cols:
    # Check if the column exists before trying to fill
    if col in df.columns:
        # Use .mode() only if there's at least one non-null value to avoid errors on all-NaN columns
        if not df[col].isnull().all():
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            # If all values are NaN, mode() will fail, handle accordingly (e.g., fill with a placeholder or leave as NaN)
            print(f"Warning: Column '{col}' contains only missing values. Cannot fill with mode.")
            # Optionally, fill with a placeholder like 'Unknown' or 'Missing'
            # df[col] = df[col].fillna('Unknown')


print("\nMissing values after imputation:")
print(df.isnull().sum())

# Save intermediate version
df.to_csv(r"C:\Users\sanka\Desktop\new peoject\earthquake_cleaned1.csv", index=False)
print("\n"r"C:\Users\sanka\Desktop\new peoject\earthquake_cleaned1.csv")