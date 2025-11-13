import streamlit as st
import pandas as pd
import joblib
import numpy as np
import reverse_geocoder as rg

# --- 1. Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Earthquake Impact Classifier",
    page_icon="🌍"
)

# --- 2. Custom CSS for a Professional Look ---
st.markdown("""
    <style>
    /* Style the main content "cards" (inputs and results) */
    [data-testid="stVerticalBlockBorder"] {
        border: 1px solid rgba(0,0,0,0.05);
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 25px 25px 25px 25px;
        background-color: #ffffff;
    }

    /* Style the main title */
    h1 {
        color: #31333F;
        font-weight: 600;
    }

    /* ------------------- DARK SIDEBAR ------------------- */
    [data-testid="stSidebar"] {
        background-color: #262730;
        color: #E0E0E0;
        padding-left: 20px;
        padding-right: 20px;
    }
    [data-testid="stSidebar"] h1 {
        color: #FFFFFF;
        font-size: 28px;
        font-weight: 600;
        margin-bottom: -10px;
    }
    [data-testid="stSidebar"] .st-subheader { color: #FFFFFF; }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary { color: #FFFFFF; }
    [data-testid="stSidebar"] .st-caption { color: #A0A0A0; }
    [data-testid="stSidebar"] .st-info {
        background-color: #31333F;
        color: #E0E0E0;
        border-color: #4A4A4A;
    }
    /* ----------------- END DARK SIDEBAR ----------------- */

    
    /* ----------------- NEW PREDICTION COLORS ----------------- */
    
    /* Base style for the prediction header (matches st.header) */
    .pred-header {
        font-size: 1.75rem; /* 28px */
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    /* Specific color classes */
    .pred-light {
        color: #33AE84; /* Professional Green */
    }
    .pred-moderate {
        color: #F3B61D; /* Professional Gold/Yellow */
    }
    .pred-strong {
        color: #E67E22; /* Professional Orange */
    }
    .pred-major {
        color: #D93025; /* Professional Red */
    }
    /* ----------------- END PREDICTION COLORS ----------------- */

    </style>
    """, unsafe_allow_html=True)


# --- 3. Load Models & Sidebar ---
@st.cache_resource
def load_models():
    """Loads the pipeline and label encoder from disk."""
    try:
        pipeline = joblib.load('earthquake_pipeline.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        return pipeline, label_encoder
    except FileNotFoundError:
        st.error("Model files not found! Please run the training script first.")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

pipeline, label_encoder = load_models()

# --- Sidebar ---
st.sidebar.title("🌍 Earthquake Classifier")

st.sidebar.header("ℹ️ About This App", divider='blue')
st.sidebar.info(
    "This app classifies the potential impact of an earthquake "
    "(Light, Moderate, Strong, or Major) based on its location, "
    "magnitude, depth, and local soil type."
)

with st.sidebar.expander("🛠️ Model & Data Details"):
    st.markdown(f"**Model Type:** `{type(pipeline.named_steps['model']).__name__}`")
    st.markdown(f"**Classes:** `{', '.join(label_encoder.classes_)}`")
    st.caption("Model trained on data from USGS and the Digital Soil Map of the World (DSMW).")

if not pipeline or not label_encoder:
    st.error("Models not loaded. App is non-functional.")
    st.stop()


# --- 4. Emergency Number Database ---
EMERGENCY_NUMBERS = {
    'US': {'All Services': '911'}, 'CA': {'All Services': '911'}, 'MX': {'All Services': '911'},
    'GB': {'All Services': '999 or 112'},
    'FR': {'All Services': '112', 'Police': '17', 'Ambulance': '15', 'Fire': '18'},
    'DE': {'All Services': '112', 'Police': '110'},
    'IT': {'All Services': '112', 'Police': '113', 'Ambulance': '118', 'Fire': '115'},
    'ES': {'All Services': '112'}, 'AU': {'All Services': '000'}, 'NZ': {'All Services': '111'},
    'JP': {'Police': '110', 'Ambulance / Fire': '119'},
    'CN': {'Police': '110', 'Ambulance': '120', 'Fire': '119'},
    'IN': {'All Services': '112', 'Police': '100', 'Ambulance': '102', 'Fire': '101'},
    'BR': {'Police': '190', 'Ambulance': '192', 'Fire': '193'}, 'RU': {'All Services': '112'},
    'ZA': {'Police': '10111', 'Ambulance / Fire': '10177'},
    'ID': {'All Services': '112', 'Ambulance': '118', 'Police': '110'},
    'PH': {'All Services': '911'}, 'TR': {'All Services': '112'},
    'CL': {'Police': '133', 'Ambulance': '131', 'Fire': '132'},
    'AR': {'Police': '101', 'Ambulance': '107', 'Fire': '100'},
    'CO': {'All Services': '123'},
    'PE': {'Police': '105', 'Ambulance': '106', 'Fire': '116'},
    'DEFAULT': {
        'Global Standard': '112 (Check local carrier)',
        'USGS Earthquake Info': 'https://earthquake.usgs.gov/'
    }
}

# --- 5. Prediction & Geocoding Functions ---
@st.cache_data
def get_prediction(data, _pipeline, _encoder):
    """
    Predicts the earthquake class from user input.
    """
    df_new = pd.DataFrame([data])
    try:
        df_new['depth_mag_interaction'] = df_new['depth'] * df_new['magnitude']
        df_new['lat_mag_interaction'] = df_new['latitude'] * df_new['magnitude']
        prediction_encoded = _pipeline.predict(df_new)
        prediction_class = _encoder.inverse_transform(prediction_encoded)[0]
        return prediction_class
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

@st.cache_data
def get_country_and_helplines(lat, lon):
    """
    Performs reverse geocoding and fetches helpline info.
    This function is cached, so it will only be slow on the *first* run.
    """
    country_code, country_name = None, None
    helplines = EMERGENCY_NUMBERS['DEFAULT'] # Default
    try:
        results = rg.search((lat, lon), mode=1)
        if results:
            info = results[0]
            country_code = info['cc']
            country_name = info['name']
            helplines = EMERGENCY_NUMBERS.get(country_code, EMERGENCY_NUMBERS['DEFAULT'])
    except Exception as e:
        st.warning(f"Could not perform reverse geocoding: {e}")
        
    return country_code, country_name, helplines

# --- 6. Enhanced Safety Info Function ---
def display_safety_info(prediction, country_code, country_name, helplines):
    """
    Displays dynamic tabs with safety info and helplines.
    """
    st.subheader("Recommended Actions & Information")
    
    tab1, tab2, tab3, tab4 = st.tabs(["✅ Safety Checklist", "📖 Expected Impact", "📉 Aftershock Advisory", "🚨 Emergency Helplines"])

    with tab1:
        st.subheader("Immediate Safety Checklist")
        if prediction == "Major":
            st.error("**IMMEDIATE DANGER: This is a catastrophic event.**", icon="🚨")
            st.markdown("""
                - **IF INDOORS:** **DROP, COVER, AND HOLD ON.** Move away from windows, outer walls, and anything that can fall.
                - **IF OUTDOORS:** Move to an open area away from buildings, streetlights, and utility wires.
                - **IF IN A VEHICLE:** Stop in a clear area, set the parking brake, and stay in the vehicle. Avoid bridges and overpasses.
                - **TSUNAMI RISK:** If in a low-lying coastal area, **EVACUATE TO HIGH GROUND** as soon as shaking stops.
            """)
        elif prediction == "Strong":
            st.warning("**HIGH RISK: Significant shaking.**", icon="🟧")
            st.markdown("""
                - **DROP, COVER, AND HOLD ON.**
                - After shaking, check yourself and others for injuries.
                - **CHECK UTILITIES:** Smell for gas leaks. If you suspect one, turn off the main valve. Shut off water and electricity if you suspect damage.
                - **DO NOT USE** elevators or matches/lighters.
            """)
        elif prediction == "Moderate":
            st.info("**BE AWARE: This will be clearly felt.**", icon="ℹ️")
            st.markdown("""
                - **STAY INDOORS** and away from windows or objects that could fall from shelves.
                - Be prepared for the possibility of stronger aftershocks.
            """)
        else: # Light
            st.success("**LOW RISK: Event felt, but damage is rare.**", icon="✅")
            st.markdown("""
                - No immediate action is required.
                - **Use this as a reminder:** Check your emergency kit and review your family's safety plan.
            """)

    with tab2:
        st.subheader("What to Expect")
        if prediction == "Major":
            st.markdown("- **Feeling:** Violent, chaotic shaking. Difficult to stand.\n- **Structural Damage:** Widespread, catastrophic damage. Most buildings collapse.\n- **Infrastructure:** Bridges, roads, and utilities destroyed.")
        elif prediction == "Strong":
            st.markdown("- **Feeling:** Strong, frightening shaking.\n- **Structural Damage:** Significant damage to unreinforced buildings. Chimneys fall.\n- **Non-Structural:** Furniture overturned. Most items knocked from shelves.")
        elif prediction == "Moderate":
            st.markdown("- **Feeling:** Felt by most; can awaken those asleep.\n- **Structural Damage:** Very unlikely. Minor cracks in plaster.\n- **Non-Structural:** Items knocked from shelves, windows may break.")
        else: # Light
            st.markdown("- **Feeling:** A light 'jolt' or 'roll.'\n- **Structural Damage:** None.\n- **Non-Structural:** None.")

    with tab3:
        st.subheader("Aftershock Advisory")
        if prediction == "Major" or prediction == "Strong":
            st.warning("**High probability of numerous, strong aftershocks.**", icon="⚠️")
            st.markdown("- Be prepared to **DROP, COVER, AND HOLD ON** repeatedly.\n- Do not re-enter damaged buildings.")
        elif prediction == "Moderate":
            st.info("**Aftershocks are possible and will be felt.**", icon="ℹ️")
        else: # Light
            st.success("**Aftershocks are not a significant concern.**", icon="✅")

    with tab4:
        st.subheader("Emergency Helplines & Resources")
        
        if country_name and country_name != 'Unknown':
            st.info(f"Displaying localized numbers for: **{country_name} ({country_code})**", icon="📍")
            for service, number in helplines.items():
                st.markdown(f"- **{service}**: `{number}`")
            
            if country_code not in ['US', 'CA', 'MX'] and '112' not in helplines.values():
                 st.caption("Note: **112** is the standard emergency number in the EU and many other nations. It may still work.")
        
        else:
            st.warning("Could not determine specific country. Displaying generic resources.", icon="⚠️")
            for service, number in helplines.items(): # This will be the DEFAULT dict
                st.markdown(f"- **{service}**: `{number}`")
        
        st.divider()
        st.markdown("""
            **Global Resources:**
            - **[List of Emergency Numbers Worldwide](https://en.wikipedia.org/wiki/List_of_emergency_telephone_numbers)**
            - **[International Red Cross and Red Crescent](https://www.ifrc.org/)**
            - **[Ready.gov Earthquake Page](https://www.ready.gov/earthquakes)**
        """)


# --- 7. Main Page UI ---
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("https://emojicdn.elk.sh/🌍?style=twitter", width=80) 
with col2:
    st.title("Earthquake Impact Classifier")
    st.caption("Predicting event impact based on location, magnitude, depth, and soil type.")

st.divider()

# --- Input Form ---
with st.container(border=True):
    with st.form(key='prediction_form'):
        st.subheader("Event Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=34.05, step=0.01)
            longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-118.25, step=0.01)
        with col2:
            magnitude = st.number_input("Magnitude (Richter)", min_value=4.0, max_value=10.0, value=5.8, step=0.1)
            depth = st.number_input("Depth (in km)", min_value=0.0, value=10.5, step=0.1)

        soil_type = st.selectbox(
            "Soil Type (from DSMW map)", 
            options=['Unknown', 'Lx', 'I-d-2a', 'Bd', 'Bc', 'Qc', 'Re', 'Rd', 'Ao', 'Je', 'Zt', 'I-e-2b', 'I-b-2c', 'G-d-2a', 'A-o', 'V-a', 'E-c'],
            help="Select the dominant soil type (DOMSOI) from the geological map."
        )

        submit_button = st.form_submit_button(label='Classify Earthquake Impact', use_container_width=True, type="primary")

# --- 8. Handle prediction and display results ---
if submit_button:
    input_data = {
        'latitude': latitude,
        'longitude': longitude,
        'depth': depth,
        'magnitude': magnitude,
        'soil_type': soil_type
    }
    
    with st.spinner('Running classification model...'):
        prediction = get_prediction(input_data, pipeline, label_encoder)
    
    with st.spinner('Fetching localized helplines... (This may take a moment on first run)'):
        country_code, country_name, helplines = get_country_and_helplines(latitude, longitude)
    
    if prediction:
        st.divider()
        
        with st.container(border=True):
            
            # --- MODIFIED: Dynamic Color-Coded Header ---
            if prediction == "Major":
                st.markdown('<h2 class="pred-header pred-major">🚨 Predicted Class: Major</h2>', unsafe_allow_html=True)
            elif prediction == "Strong":
                st.markdown('<h2 class="pred-header pred-strong">🟧 Predicted Class: Strong</h2>', unsafe_allow_html=True)
            elif prediction == "Moderate":
                st.markdown('<h2 class="pred-header pred-moderate">ℹ️ Predicted Class: Moderate</h2>', unsafe_allow_html=True)
            else:
                st.markdown('<h2 class="pred-header pred-light">✅ Predicted Class: Light</h2>', unsafe_allow_html=True)

            st.divider()

            # --- Map & Input Summary ---
            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                st.subheader("Event Location")
                loc_caption = f"Epicenter: {latitude}, {longitude}"
                if country_name:
                    loc_caption += f" (near {country_name})"
                
                map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
                st.map(map_data, zoom=5)
                st.caption(loc_caption)
            
            with res_col2:
                st.subheader("Input Summary")
                st.markdown(f"**Magnitude:** `{magnitude}`")
                st.markdown(f"**Depth:** `{depth} km`")
                st.markdown(f"**Soil Type:** `{soil_type}`")
                with st.expander("Show Full Input Data"):
                    st.json(input_data)

            st.divider()
            
            # --- Actionable Info Section ---
            display_safety_info(prediction, country_code, country_name, helplines)
