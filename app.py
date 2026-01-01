import os
import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

# --------------------------------------------------
# BASE DIRECTORY
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
COURSE_NAME = "CMP7005 – Data Analytics and Visualisation"

# --------------------------------------------------
# CITY COORDINATES
# --------------------------------------------------
CITY_COORDS = {
    "Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "Kolkata": [22.5726, 88.3639],
    "Chennai": [13.0827, 80.2707],
    "Bengaluru": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
    "Pune": [18.5204, 73.8567],
    "Ahmedabad": [23.0225, 72.5714],
    "Jaipur": [26.9124, 75.7873],
    "Lucknow": [26.8467, 80.9462]
}

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Air Quality Analysis – CMP7005",
    layout="wide",
    page_icon="🌫"
)

# --------------------------------------------------
# SIDEBAR – NAVIGATION
# --------------------------------------------------
st.sidebar.title("📊 Dashboard Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home Dashboard",
        "🔮 PM2.5 Prediction",
        "📊 EDA & Visualisation",
        "🗺 India AQ Map",
        "🏙 City AQ Map"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📚 {COURSE_NAME}")

with st.sidebar.expander("ℹ️ Project Info"):
    st.markdown(f"""
    **Academic Project**

    **Course:** {COURSE_NAME}

    **Objective:**  
    - Analyze air quality data across Indian cities  
    - Predict PM2.5 levels using ML  
    - Visualize pollution patterns  

    **Dataset:**  
    - Multi-city air quality measurements  
    - Parameters: PM2.5, PM10, SO₂, NO₂, CO, O₃, NH₃
    """)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "merged_data.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1N317Atsm71Is04H_P711V3Dk-jr5y1ou"
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇ Downloading ML model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Model error: {e}")
    model = None

# --------------------------------------------------
# AQI FUNCTIONS
# --------------------------------------------------
def aqi_category(pm):
    if pm <= 30: return "Good"
    elif pm <= 60: return "Satisfactory"
    elif pm <= 90: return "Moderate"
    elif pm <= 120: return "Poor"
    elif pm <= 250: return "Very Poor"
    else: return "Severe"

def get_aqi_color(category):
    return {
        "Good": "#00E400",
        "Satisfactory": "#FFFF00",
        "Moderate": "#FF7E00",
        "Poor": "#FF0000",
        "Very Poor": "#8F3F97",
        "Severe": "#7E0023"
    }.get(category, "#808080")

df["AQI Category"] = df["PM2.5"].apply(aqi_category)

# ==================================================
# HOME DASHBOARD
# ==================================================
if page == "🏠 Home Dashboard":
    st.title("🌫 Air Quality Analysis & Prediction Dashboard")
    st.markdown(f"### {COURSE_NAME} Project")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Cities", df["City"].nunique())
    with c2:
        st.metric("Records", f"{len(df):,}")

    st.markdown("---")

# ==================================================
# PREDICTION PAGE
# ==================================================
elif page == "🔮 PM2.5 Prediction":
    st.title("🔮 PM2.5 Concentration Predictor")
    st.markdown("Predict PM2.5 using Machine Learning")

    if model:
        so2 = st.number_input("SO₂", 0.0, 500.0, 10.0)
        no2 = st.number_input("NO₂", 0.0, 500.0, 20.0)
        co = st.number_input("CO", 0.0, 10.0, 1.0)
        o3 = st.number_input("O₃", 0.0, 500.0, 30.0)
        pm10 = st.number_input("PM10", 0.0, 1000.0, 50.0)
        nh3 = st.number_input("NH₃", 0.0, 500.0, 15.0)

        if st.button("Predict"):
            pred = model.predict([[so2, no2, co, o3, pm10, nh3]])[0]
            category = aqi_category(pred)
            st.success(f"PM2.5: {pred:.2f} µg/m³ ({category})")

# ==================================================
# EDA PAGE
# ==================================================
elif page == "📊 EDA & Visualisation":
    st.title("📊 Exploratory Data Analysis")
    st.dataframe(df.head(50))

# ==================================================
# INDIA MAP
# ==================================================
elif page == "🗺 India AQ Map":
    st.title("🗺 India Air Quality Map")

    city_pm = df.groupby("City")["PM2.5"].mean().reset_index()
    india_map = folium.Map(location=[22.5, 80], zoom_start=5)

    for _, row in city_pm.iterrows():
        if row["City"] in CITY_COORDS:
            lat, lon = CITY_COORDS[row["City"]]
            folium.CircleMarker(
                [lat, lon],
                radius=8,
                color="red",
                fill=True,
                tooltip=f"{row['City']} – {row['PM2.5']:.1f}"
            ).add_to(india_map)

    st_folium(india_map, width=1000, height=600)

# ==================================================
# CITY MAP
# ==================================================
elif page == "🏙 City AQ Map":
    st.title("🏙 City Air Quality")

    city = st.selectbox("Select City", sorted(df["City"].unique()))
    city_df = df[df["City"] == city]

    st.metric("Average PM2.5", f"{city_df['PM2.5'].mean():.2f}")
    st.dataframe(city_df.head(100))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(f"""
**📘 Course:** {COURSE_NAME}  
**🎓 Academic Project**  
**🌍 Making environmental data accessible and actionable**
""")
