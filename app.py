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
# CUSTOM CSS (CLEAN DARK UI)
# --------------------------------------------------
st.markdown("""
<style>
html, body {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0b1220;
}

section[data-testid="stSidebar"] {
    background-color: #020617;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #111827, #020617);
    padding: 3.5rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 3rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}

.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    color: #f9fafb;
}

.hero p {
    font-size: 1.15rem;
    color: #cbd5f5;
}

/* Cards */
.card {
    background: #111827;
    border-radius: 16px;
    padding: 1.8rem;
    height: 100%;
    box-shadow: 0 12px 25px rgba(0,0,0,0.35);
}

.card h3 {
    color: #f9fafb;
}

.card p {
    color: #9ca3af;
    font-size: 0.95rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 3rem;
    font-size: 0.85rem;
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
# SIDEBAR NAVIGATION
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
    if pm <= 30:
        return "Good"
    elif pm <= 60:
        return "Satisfactory"
    elif pm <= 90:
        return "Moderate"
    elif pm <= 120:
        return "Poor"
    elif pm <= 250:
        return "Very Poor"
    else:
        return "Severe"

df["AQI Category"] = df["PM2.5"].apply(aqi_category)

# ==================================================
# HOME PAGE
# ==================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>Air Quality Analysis Dashboard</h1>
        <p>Predict, analyze, and visualize air pollution trends across Indian cities</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
            <h3>🔮 PM2.5 Prediction</h3>
            <p>Predict PM2.5 concentration using a trained machine learning model.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <h3>📊 Exploratory Analysis</h3>
            <p>Filter and analyze air quality data across cities and AQI categories.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
            <h3>🗺 Geospatial Insights</h3>
            <p>Visualize air quality spatially using India-level and city-level maps.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="footer">CMP7005 · Streamlit Cloud Deployment</div>', unsafe_allow_html=True)

# ==================================================
# PREDICTION PAGE
# ==================================================
elif page == "🔮 PM2.5 Prediction":
    st.subheader("Predict PM2.5 Concentration")

    c1, c2 = st.columns(2)
    with c1:
        so2 = st.number_input("SO₂", 0.0, value=10.0)
        no2 = st.number_input("NO₂", 0.0, value=20.0)
        co = st.number_input("CO", 0.0, value=1.0)
    with c2:
        o3 = st.number_input("O₃", 0.0, value=30.0)
        pm10 = st.number_input("PM10", 0.0, value=50.0)
        nh3 = st.number_input("NH₃", 0.0, value=15.0)

    if st.button("Predict PM2.5"):
        pred = model.predict(np.array([[so2, no2, co, o3, pm10, nh3]]))[0]
        st.success(f"Predicted PM2.5: {pred:.2f} µg/m³")

# ==================================================
# EDA PAGE (FULLY RESTORED)
# ==================================================
elif page == "📊 EDA & Visualisation":
    st.subheader("Exploratory Data Analysis (Filter-Driven)")

    f1, f2, f3 = st.columns(3)

    with f1:
        city_filter = st.selectbox("City", ["All"] + sorted(df["City"].unique()))

    with f2:
        pm25_range = st.slider(
            "PM2.5 Range",
            float(df["PM2.5"].min()),
            float(df["PM2.5"].max()),
            (float(df["PM2.5"].min()), float(df["PM2.5"].max()))
        )

    with f3:
        aqi_filter = st.multiselect(
            "AQI Category",
            df["AQI Category"].unique(),
            default=df["AQI Category"].unique()
        )

    filtered_df = df.copy()

    if city_filter != "All":
        filtered_df = filtered_df[filtered_df["City"] == city_filter]

    filtered_df = filtered_df[
        (filtered_df["PM2.5"] >= pm25_range[0]) &
        (filtered_df["PM2.5"] <= pm25_range[1])
    ]

    filtered_df = filtered_df[
        filtered_df["AQI Category"].isin(aqi_filter)
    ]

    st.dataframe(filtered_df.head(20))

    st.subheader("AQI Category Distribution")
    st.bar_chart(filtered_df["AQI Category"].value_counts())

    st.subheader("PM2.5 Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered_df["PM2.5"], bins=30)
    st.pyplot(fig)

    st.subheader("PM2.5 vs PM10")
    fig, ax = plt.subplots()
    ax.scatter(filtered_df["PM10"], filtered_df["PM2.5"], alpha=0.5)
    st.pyplot(fig)

# ==================================================
# INDIA MAP
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
# CITY MAP
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
