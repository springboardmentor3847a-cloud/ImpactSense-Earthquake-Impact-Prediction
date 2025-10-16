import pandas as pd

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\sanka\Desktop\new peoject\earthquake_scaled2.csv")

df['Location'] = df['Latitude'].astype(str) + ',' + df['Longitude'].astype(str)
# Encode Soil Type (DOMSOI)
if 'DOMSOI' in df.columns:
    encoder = LabelEncoder()
    df['SoilType_Encoded'] = encoder.fit_transform(df['DOMSOI'])
else:
    df['SoilType_Encoded'] = -1  # fallback if missing

# Encode Location (optional, may be too high-cardinality)
df['Location_Encoded'] = LabelEncoder().fit_transform(df['Location'])

print(df[['DOMSOI', 'SoilType_Encoded']].head())

# Save encoded dataset
df.to_csv(r"C:\Users\sanka\Desktop\new peoject\earthquake_encoded3.csv", index=False)
