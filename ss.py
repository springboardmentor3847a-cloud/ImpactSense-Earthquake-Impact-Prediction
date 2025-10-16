import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load a historical dataset (Kaggle/NOAA/USGS archive)
df_hist = pd.read_csv(r"C:\Users\sanka\Desktop\new peoject\Significant_Earthquakes.csv")
print("Columns in the CSV file:", df_hist.columns.tolist())

# Select & clean - using the correct column names from your dataset
df_hist = df_hist[['Date', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
df_hist['Date'] = pd.to_datetime(df_hist['Date'], errors='coerce')
df_hist = df_hist.dropna()

# Rename columns to more descriptive names (optional but recommended)
df_hist = df_hist.rename(columns={
    'time': 'Date',
    'latitude': 'Latitude', 
    'longitude': 'Longitude',
    'depth': 'Depth',
    'mag': 'Magnitude',
    'place': 'Location'
})

# --- EDA ---
print(df_hist.describe())
print("Missing values:", df_hist.isnull().sum())

# Time series trend
df_hist['Year'] = df_hist['Date'].dt.year
plt.figure(figsize=(10,5))
sns.lineplot(data=df_hist.groupby('Year')['Magnitude'].mean().reset_index(), x='Year', y='Magnitude')
plt.title("Average Magnitude of Earthquakes Over Time")
plt.show()

# Magnitude vs Depth
plt.figure(figsize=(8,5))
sns.scatterplot(data=df_hist, x="Depth", y="Magnitude", alpha=0.5)
plt.title("Depth vs Magnitude (Historical Data)")
plt.show()