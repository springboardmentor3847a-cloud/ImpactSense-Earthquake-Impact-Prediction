
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\sanka\Desktop\new peoject\earthquake_cleaned1.csv")

# Select numeric features
numeric_features = ['Magnitude', 'Depth']

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])

print(df_scaled[numeric_features].describe())

# Save scaled dataset
df_scaled.to_csv(r"C:\Users\sanka\Desktop\new peoject\earthquake_scaled2.csv", index=False)