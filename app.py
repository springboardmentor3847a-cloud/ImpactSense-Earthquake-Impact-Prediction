import gradio as gr
import pickle
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 🎯 Load model and encoder
model_path = "/content/best_random_forest.pkl"
le_path = "/content/drive/MyDrive/datasets/lable_encoder.pkl"

try:
    model = pickle.load(open(model_path, "rb"))
    label_encoder = joblib.load(le_path)
except FileNotFoundError as e:
    raise FileNotFoundError(f"⚠️ Missing file: {e}")

# 🔮 Prediction Function
def predict_damage(magnitude, depth, soil_id, region_id):
    # Prepare data
    df = pd.DataFrame([{
        "Magnitude": magnitude,
        "Depth": depth,
        "SoilType_Encoded": int(soil_id),
        "Region_Cluster": int(region_id)
    }])

    # Predict
    pred = model.predict(df)[0]
    decoded = label_encoder.inverse_transform([pred])[0]

    # 🎨 Create probability plot (if available)
    probs_fig = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(df)[0]
            class_indices = list(range(len(probs)))
            try:
                class_labels = label_encoder.inverse_transform(class_indices)
            except Exception:
                class_labels = [str(i) for i in class_indices]

            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.bar(class_labels, probs, color="#4F75FF")
            ax.set_ylim(0, 1)
            ax.set_ylabel("📊 Probability")
            ax.set_xlabel("🏠 Damage Category")
            ax.set_title("🌋 Predicted Class Probabilities")

            for b in bars:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.01,
                    f"{b.get_height():.2f}",
                    ha="center",
                    fontsize=9,
                    color="black"
                )
            plt.tight_layout()
            probs_fig = fig
        except Exception as e:
            probs_fig = f"⚠️ Could not create probability chart: {e}"

    # ✨ Final Message
    result = f"🧭 **Predicted Damage Level:** `{decoded}`"
    advice = "\n\n💡 *Tip:* Regular maintenance and safety drills can reduce future earthquake damage."
    return result + advice, probs_fig


# 🧩 Gradio Interface Setup
iface = gr.Interface(
    fn=predict_damage,
    inputs=[
        gr.Slider(0.0, 10.0, value=5.5, step=0.1, label="🌋 Magnitude (Richter Scale)"),
        gr.Slider(0.0, 700.0, value=10.0, step=1.0, label="🌎 Depth (km)"),
        gr.Number(value=0, label="🧱 Soil Type (Encoded)"),
        gr.Number(value=0, label="📍 Region Cluster")
    ],
    outputs=[
        gr.Markdown(label="🎯 Prediction Result"),
        gr.Plot(label="📊 Probability Chart")
    ],
    title="🌍 Earthquake Damage Prediction App",
    description=(
        "🧠 **Welcome to the Earthquake Damage Predictor!**\n\n"
        "Estimate the level of building damage based on earthquake parameters.\n"
        "Powered by a trained **Random Forest Model** 🤖 and smart data preprocessing ⚙️.\n\n"
        "Move the sliders, press **Predict**, and watch the results in real time! 🚀"
    ),
    theme="soft",
    allow_flagging="never"
)

# 🚀 Launch with both Local and Public URLs
iface.launch(share=True)
