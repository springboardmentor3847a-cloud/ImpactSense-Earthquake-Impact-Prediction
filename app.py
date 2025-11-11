import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import json
import time
from openai import OpenAI
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="ImpactSense",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Helper Functions ---

def get_openai_client():
    """Safely tries to get the OpenAI client."""
    try:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets['OPENROUTER_API_KEY'])
    except: return None

@st.cache_data
def get_location_context(lat, lon):
    """Uses Geopy for accurate location names."""
    location_name = f"Unknown ({lat:.2f}, {lon:.2f})"
    is_populated = 0
    try:
        geolocator = Nominatim(user_agent="impactsense_edu_app")
        location = geolocator.reverse((lat, lon), exactly_one=True, language='en', timeout=5)
        if location:
            address = location.raw.get('address', {})
            city = address.get('city') or address.get('town') or address.get('village') or address.get('county')
            state = address.get('state')
            country = address.get('country')
            parts = [p for p in [city, state, country] if p]
            location_name = ", ".join(parts) if parts else location.address
            is_populated = 1
        else:
            location_name = f"Remote Region ({lat:.1f}, {lon:.1f})"
    except:
        pass
    return {"location_name": location_name, "is_populated": is_populated}

@st.cache_data
def get_ai_summary(mag, depth, loc, risk):
    """Get initial situation report from LLM as bullet points."""
    client = get_openai_client()
    if not client: return json.dumps({"location_name": loc, "summary_points": ["AI summary unavailable."]})
    try:
        system_prompt = """
        You are a senior seismologist at the USGS.
        Your response MUST be a valid JSON object with two keys:
        1. "location_name": A short location name (e.g., "Mojave Desert, CA").
        2. "summary_points": A JSON list of 2-3 short, bulleted strings explaining the risk.
        """
        user_prompt = f"Analyze earthquake: Mag {mag:.1f}, Depth {depth:.1f}km, at {loc}. Predicted Risk: {risk}"
        resp = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.6, max_tokens=250, response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content
    except Exception as e: 
        return json.dumps({"location_name": loc, "summary_points": [f"AI summary error: {e}"]})

def get_ai_chat_response(messages_history):
    """Gets a JSON response with an answer and follow-up questions."""
    client = get_openai_client()
    if not client: 
        return json.dumps({"answer": "Chat unavailable (API key not configured).", "follow_up_questions": []})
    
    try:
        system_prompt = f"""
        You are a helpful disaster response expert.
        Context: {st.session_state.ctx['risk']} risk at {st.session_state.ctx['loc']}.
        Your response MUST be a valid JSON object with two keys:
        1. "answer": (String) A concise, 2-4 sentence answer to the user's question.
        2. "follow_up_questions": (List of 3 strings) 3 short, relevant follow-up questions.
        """
        api_messages = [{"role": "system", "content": system_prompt}, messages_history[-1]] 
        
        resp = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=api_messages,
            temperature=0.7, max_tokens=400, stream=False, response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content
    except Exception as e: 
        return json.dumps({"answer": f"Sorry, the AI assistant ran into an error: {e}", "follow_up_questions": []})

@st.cache_resource
def load_assets():
    """Loads model assets safely."""
    try:
        base = 'saved_model'
        return {
            "m": joblib.load(os.path.join(base, 'earthquake_clf_model.pkl')),
            "le": joblib.load(os.path.join(base, 'label_encoder.pkl')),
            "f": joblib.load(os.path.join(base, 'feature_names.pkl'))
        }
    except: return None

# --- 3. Main App Logic ---
if "messages" not in st.session_state: st.session_state.messages = []
if "ctx" not in st.session_state: st.session_state.ctx = None
if "follow_ups" not in st.session_state: st.session_state.follow_ups = []

assets = load_assets()

# --- 4. Sidebar ---
st.sidebar.header("🌊 ImpactSense")
if assets:
    with st.sidebar.form("pred"):
        st.write("Parameters:")
        c1, c2 = st.columns(2)
        mag = c1.number_input("Magnitude", 2.5, 10.0, 6.0, 0.1)
        lon = c1.number_input("Longitude", -180.0, 180.0, -118.2, 0.1)
        depth = c2.number_input("Depth (km)", 0.0, 700.0, 10.0, 1.0)
        lat = c2.number_input("Latitude", -90.0, 90.0, 34.0, 0.1)
        run = st.form_submit_button("Analyze", type="primary", use_container_width=True)

# --- 5. Main UI (Tabs) ---
if assets:
    tab1, tab2 = st.tabs(["📊 Dashboard & AI Assistant", "ℹ️ About"])

    # === TAB 1: DASHBOARD & CHAT ===
    with tab1:
        if run:
            with st.spinner("Locating and analyzing event..."):
                loc_data = get_location_context(lat, lon)
                input_data = pd.DataFrame([[mag, lon, lat, depth, loc_data['is_populated']]], columns=assets['f'])
                probs = assets['m'].predict_proba(input_data)[0]
                risk = assets['le'].inverse_transform([np.argmax(probs)])[0]
                
                st.session_state.ctx = {"mag": mag, "depth": depth, "loc": loc_data['location_name'], "risk": risk, "lat": lat, "lon": lon, "probs": probs, "is_pop": loc_data['is_populated']}
                st.session_state.messages = []
                st.session_state.follow_ups = []
                
                summary_json_str = get_ai_summary(mag, depth, loc_data['location_name'], risk)
                st.session_state.summary_data = json.loads(summary_json_str)

        if st.session_state.ctx:
            ctx = st.session_state.ctx
            risk = ctx['risk']
            loc_data = st.session_state.summary_data
            
            # --- 1. Header ---
            c_hex, emoji = ("#FF4B4B", "🔴") if risk == "High" else ("#FFA500", "🟡") if risk == "Medium" else ("#00C851", "🟢")
            st.markdown(f"<h2 style='text-align: center; color: {c_hex};'>{emoji} {risk} Risk Event</h2>", unsafe_allow_html=True)
            pop_text = "Populated Area" if ctx['is_pop'] else "Remote Area"
            pop_emoji = "🏙️" if ctx['is_pop'] else "🏞️"
            st.markdown(f"<h3 style='text-align: center;'>📍 {ctx['loc']} | {pop_emoji} {pop_text}</h3>", unsafe_allow_html=True)
            st.divider()
            
            # --- 2. Map & Confidence ---
            cm, cs = st.columns([1.5, 1])
            cm.map(pd.DataFrame({'lat': [ctx['lat']], 'lon': [ctx['lon']]}), zoom=6, use_container_width=True)
            cs.subheader("Confidence")
            cs.bar_chart(pd.DataFrame({'Risk': assets['le'].classes_, 'Prob': ctx['probs']}).set_index('Risk'), color=c_hex)
            
            st.divider()
            
            # --- 3. AI Summary ---
            st.subheader(f"📝 AI Situation Report")
            summary_points = loc_data.get('summary_points', ["Summary not available."])
            summary_markdown = "\n".join(f"- {point}" for point in summary_points)
            if risk == "High": st.error(summary_markdown, icon="🚨")
            elif risk == "Medium": st.warning(summary_markdown, icon="⚠️")
            else: st.success(summary_markdown, icon="✅")

            # --- 4. AI ASSISTANT (MOVED HERE) ---
            st.divider()
            st.subheader("💬 AI Assistant")
            st.caption(f"Ask follow-up questions about the {ctx['loc']} event.")

            # --- This is the function that handles the button logic ---
            def ask_question(question):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.follow_ups = [] # Clear old suggestions

            # Chat history container
            chat_container = st.container(height=350, border=False)
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                
                # This logic runs if the LAST message was from USER
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response_json_str = get_ai_chat_response(st.session_state.messages)
                            try:
                                response_data = json.loads(response_json_str)
                                answer = response_data.get("answer", "Sorry, I couldn't understand that.")
                                follow_ups = response_data.get("follow_up_questions", [])
                                st.markdown(answer)
                                # Save response and new follow-ups
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                                st.session_state.follow_ups = follow_ups
                                st.rerun() # Rerun to show new follow-ups
                            except json.JSONDecodeError:
                                st.error("AI returned an invalid response. Please try again.")
                                st.session_state.follow_ups = []
                                st.rerun()

            # --- NEW: Suggestion Area (Gemini-style) ---
            st.write("")
            suggestion_container = st.container()
            with suggestion_container:
                # If chat is empty, show initial prompts
                if len(st.session_state.messages) == 0:
                    st.caption("Start the conversation:")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💡 **What are the first 3 things I should do?**\n*Ask about immediate safety steps.*", use_container_width=True):
                            ask_question("What are the first 3 things I should do?")
                            st.rerun()
                        if st.button("🏠 **How do I check my home for damage?**\n*Ask about structural and utility safety.*", use_container_width=True):
                            ask_question("How do I check my home for damage safely?")
                            st.rerun()
                    with col2:
                        if st.button("🌊 **Is there a tsunami risk for this area?**\n*Ask about coastal and secondary dangers.*", use_container_width=True):
                            ask_question("Is there a tsunami risk for this area?")
                            st.rerun()
                        if st.button("📉 **What are aftershocks?**\n*Ask about follow-on tremors and risks.*", use_container_width=True):
                            ask_question("What are aftershocks and should I expect them?")
                            st.rerun()
                
                # If we have follow-ups from the AI, show them
                elif len(st.session_state.follow_ups) > 0:
                    st.caption("Suggested follow-ups:")
                    cols = st.columns(len(st.session_state.follow_ups))
                    for i, question in enumerate(st.session_state.follow_ups[:3]): # Show max 3
                        if cols[i].button(question, use_container_width=True, key=f"q_{i}"):
                            ask_question(question)
                            st.rerun()

            # Main chat input (this stays at the bottom)
            if user_input := st.chat_input("Ask a follow-up question..."):
                ask_question(user_input)
                st.rerun() # Rerun to trigger the AI response logic

        else:
            st.info("👈 Select parameters in the sidebar and click **Analyze** to start.")


   # === TAB 2: ABOUT (with new content) ===
    with tab2:
        st.header("ℹ️ About ImpactSense")
        
        st.markdown("""
        **ImpactSense** is an end-to-end data science application that demonstrates how to combine machine learning and large language models (LLMs) to analyze and respond to real-time events. This tool predicts the potential impact of an earthquake and provides an interactive AI assistant for immediate, context-aware advice.
        """)

        st.subheader("Core Technology Stack")
        st.markdown("""
        * **Machine Learning Model:** An **XGBoost Classifier** predicts the event's risk level ("Low", "Medium", "High").
        * **AI Analyst & Assistant:** A **Mistral-7B** model (via OpenRouter) provides dynamic, human-readable summaries and powers the conversational chatbot.
        * **Data Pipeline:**
            * **Seismic Data:** Fetched live from the **U.S. Geological Survey (USGS)** API.
            * **Geospatial Data:** Location names are accurately identified using the **Geopy (Nominatim/OpenStreetMap)** library.
        * **Frontend:** The application is built and served with **Streamlit**.
        """)

        st.subheader("How It Works: From Data to Dashboard")
        st.markdown("""
        1.  **Data Fetching:** The app's model was trained on 3+ years of global earthquake data (M2.5+) fetched in monthly batches from the USGS.
        2.  **Feature Engineering:** A crucial feature, `is_populated`, was created by analyzing the location text to determine if an event occurred near a populated area. This gives the model essential geographic context.
        3.  **Data Balancing (SMOTE):** To ensure the model could accurately identify rare, high-risk events, the imbalanced training data was balanced using the **SMOTE** (Synthetic Minority Over-sampling Technique).
        4.  **Prediction:** When you enter parameters, the app first uses Geopy to find the location name and `is_populated` status. It then feeds all 5 features (`mag`, `lon`, `lat`, `depth`, `is_populated`) into the trained XGBoost model to get a risk probability.
        5.  **AI Analysis:** The model's output (e.g., "High Risk") and the location data are fed to the Mistral-7B LLM, which generates the easy-to-read situation report and prepares the AI assistant for your questions.
        """)
        
        st.warning("""
        **Project Disclaimer:**
        The "Risk Level" is a synthetic target created for this project to demonstrate a complete machine learning workflow. This application is for educational and portfolio purposes **only** and should not be used for actual disaster response or personal safety decisions.
        """)

else:
    st.error("🚨 Models missing. Please run the notebook first.")