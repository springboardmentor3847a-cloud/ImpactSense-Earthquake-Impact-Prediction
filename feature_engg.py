import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\sanka\Desktop\new peoject\earthquake_encoded3.csv")

# 1. Magnitude Category
df['Magnitude_Class'] = pd.cut(
    df['Magnitude'],
    bins=[0, 4, 6, 7, 10],
    labels=['Low', 'Moderate', 'Strong', 'Major']
)

# 2. Depth Category
df['Depth_Class'] = pd.cut(
    df['Depth'],
    bins=[-1, 70, 300, 700],
    labels=['Shallow', 'Intermediate', 'Deep']
)

# 3. Risk Score (simple weighted sum)
df['Risk_Score'] = (
    (df['Magnitude'] * 0.7) +
    (np.log1p(df['Depth']) * 0.3)
)

# 4. Spatial Clusters
coords = df[['Latitude', 'Longitude']]
kmeans = KMeans(n_clusters=5, random_state=42)
df['Region_Cluster'] = kmeans.fit_predict(coords)

print(df[['Magnitude', 'Depth', 'Risk_Score', 'Region_Cluster']].head())

# Save engineered dataset
df.to_csv(r"C:\Users\sanka\Desktop\new peoject\earthquake_features4.csv", index=False)

# Load latest version
df_final = pd.read_csv(r"C:\Users\sanka\Desktop\new peoject\earthquake_features4.csv")

print("Final dataset shape:", df_final.shape)
print("Final columns:", df_final.columns.tolist())

# Save final dataset for modeling
df_final.to_csv(r"C:\Users\sanka\Desktop\new peoject\earthquake_preprocessed_wwek2.csv", index=False)