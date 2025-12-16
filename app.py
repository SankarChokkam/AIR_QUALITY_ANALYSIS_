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
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Air Quality Analysis – CMP7005",
    layout="wide",
    page_icon="🌫"
)

# --------------------------------------------------
# CUSTOM CSS (ANIMATIONS + UI)
# --------------------------------------------------
st.markdown("""
<style>

/* Global */
html, body {
    font-family: 'Segoe UI', sans-serif;
}

/* Animated title */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.hero-title {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    animation: fadeInUp 1.2s ease-out;
}

.hero-subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1.2rem;
    margin-bottom: 2rem;
    animation: fadeInUp 1.6s ease-out;
}

/* Floating cards */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
    100% { transform: translateY(0px); }
}

.card {
    background: linear-gradient(135deg, #f8fafc, #e5e7eb);
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    animation: float 4s ease-in-out infinite;
}

.card h3 {
    margin-bottom: 0.5rem;
}

.card p {
    color: #374151;
    font-size: 0.95rem;
}

/* Section spacing */
.section {
    margin-top: 3rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 4rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# BASE DIRECTORY
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
# SIDEBAR – NAVIGATION
# --------------------------------------------------
st.sidebar.title("🌫 Air Quality Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🔮 PM2.5 Prediction",
        "📊 EDA & Visualisation",
        "🗺 India AQ Map",
        "🏙 City AQ Map"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("CMP7005 · Streamlit Cloud")

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
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# AQI FUNCTION
# --------------------------------------------------
def aqi_category(pm):
    if pm <= 30: return "Good"
    elif pm <= 60: return "Satisfactory"
    elif pm <= 90: return "Moderate"
    elif pm <= 120: return "Poor"
    elif pm <= 250: return "Very Poor"
    else: return "Severe"

df["AQI Category"] = df["PM2.5"].apply(aqi_category)

# ==================================================
# 🏠 HOME PAGE (ANIMATED)
# ==================================================
if page == "🏠 Home":

    st.markdown('<div class="hero-title">Air Quality Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Predict, analyze, and visualize air pollution trends across Indian cities</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
            <h3>🔮 Prediction</h3>
            <p>Machine learning model to predict PM2.5 concentration using key pollutants.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <h3>📊 Data Insights</h3>
            <p>Interactive exploratory analysis with filters for city, AQI, and pollution range.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
            <h3>🗺 Geospatial View</h3>
            <p>City-wise and India-level air quality maps for spatial understanding.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        Use the sidebar to explore predictions, analysis, and maps.<br>
        Designed for CMP7005 · Streamlit Cloud Deployment
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# 🔮 PREDICTION PAGE
# ==================================================
elif page == "🔮 PM2.5 Prediction":
    st.subheader("Predict PM2.5 Concentration")

    col1, col2 = st.columns(2)
    with col1:
        so2 = st.number_input("SO₂", 0.0, value=10.0)
        no2 = st.number_input("NO₂", 0.0, value=20.0)
        co = st.number_input("CO", 0.0, value=1.0)
    with col2:
        o3 = st.number_input("O₃", 0.0, value=30.0)
        pm10 = st.number_input("PM10", 0.0, value=50.0)
        nh3 = st.number_input("NH₃", 0.0, value=15.0)

    if st.button("Predict PM2.5"):
        pred = model.predict(np.array([[so2, no2, co, o3, pm10, nh3]]))[0]
        st.success(f"Predicted PM2.5: {pred:.2f} µg/m³")

# ==================================================
# 📊 EDA PAGE
# ==================================================
elif page == "📊 EDA & Visualisation":
    st.subheader("Exploratory Data Analysis")

    city = st.selectbox("City", ["All"] + sorted(df["City"].unique()))
    data = df if city == "All" else df[df["City"] == city]

    st.dataframe(data.head(20))
    st.bar_chart(data["AQI Category"].value_counts())

    fig, ax = plt.subplots()
    ax.hist(data["PM2.5"], bins=30)
    st.pyplot(fig)

# ==================================================
# 🗺 INDIA MAP
# ==================================================
elif page == "🗺 India AQ Map":
    st.subheader("India Air Quality Map")

    city_pm = df.groupby("City")["PM2.5"].mean().reset_index()
    m = folium.Map(location=[22.5, 80.0], zoom_start=5)

    for _, row in city_pm.iterrows():
        if row["City"] in CITY_COORDS:
            lat, lon = CITY_COORDS[row["City"]]
            folium.CircleMarker([lat, lon], radius=8, fill=True).add_to(m)

    st_folium(m, width=1000, height=500)

# ==================================================
# 🏙 CITY MAP
# ==================================================
elif page == "🏙 City AQ Map":
    st.subheader("City Air Quality Map")

    city = st.selectbox("Select City", sorted(df["City"].unique()))
    avg_pm = df[df["City"] == city]["PM2.5"].mean()

    lat, lon = CITY_COORDS[city]
    m = folium.Map(location=[lat, lon], zoom_start=11)

    folium.CircleMarker([lat, lon], radius=15, fill=True).add_to(m)

    st.markdown(f"**Average PM2.5:** {avg_pm:.2f}")
    st.markdown(f"**AQI Category:** {aqi_category(avg_pm)}")
    st_folium(m, width=900, height=500)
