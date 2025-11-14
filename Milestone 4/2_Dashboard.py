import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 Earthquake Prediction Dashboard")

log_path = "prediction_log.csv"

if not os.path.exists(log_path):
    st.warning("No prediction history found. Run predictions first.")
    st.stop()

df = pd.read_csv(log_path, parse_dates=["Timestamp"])
df["Date"] = df["Timestamp"].dt.date

# === Sidebar Filters ===
st.sidebar.header("🔎 Filter Data")
regions = st.sidebar.multiselect("Region Cluster", sorted(df["Region_Cluster"].unique()))
soils = st.sidebar.multiselect("Soil Type", sorted(df["SoilType_Encoded"].unique()))

filtered = df.copy()
if regions:
    filtered = filtered[filtered["Region_Cluster"].isin(regions)]
if soils:
    filtered = filtered[filtered["SoilType_Encoded"].isin(soils)]

# === 1. Predictions Over Time ===
st.subheader("📈 Predictions Per Day")
daily = filtered.groupby("Date").size().reset_index(name="Predictions")
fig1, ax1 = plt.subplots()
sns.lineplot(data=daily, x="Date", y="Predictions", marker="o", ax=ax1)
ax1.set_title("Predictions Over Time")
st.pyplot(fig1)

# === 2. Damage Category Frequency ===
st.subheader("📊 Damage Category Frequency")
fig2, ax2 = plt.subplots()
filtered["Predicted_Damage"].value_counts().plot(kind="bar", ax=ax2, color="skyblue")
ax2.set_title("Predicted Damage Categories")
st.pyplot(fig2)

# === 3. Magnitude by Damage Category ===
st.subheader("📦 Magnitude Distribution by Category")
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=filtered, x="Predicted_Damage", y="Magnitude", ax=ax3)
ax3.set_title("Magnitude by Predicted Damage")
st.pyplot(fig3)

# === 4. Confidence Trend ===
st.subheader("🔍 Confidence Over Time")
conf = filtered.groupby("Date")["Confidence"].mean().reset_index()
fig4, ax4 = plt.subplots()
sns.lineplot(data=conf, x="Date", y="Confidence", marker="o", ax=ax4)
ax4.set_title("Average Confidence Over Time")
st.pyplot(fig4)

# === Export All Plots ===
st.markdown("### 📁 Export Visuals")
for i, (fig, name) in enumerate(zip([fig1, fig2, fig3, fig4],
                                    ["predictions_over_time", "damage_frequency", "magnitude_boxplot", "confidence_trend"])):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label=f"📥 Download {name}.png",
        data=buf.getvalue(),
        file_name=f"{name}.png",
        mime="image/png"
    )
