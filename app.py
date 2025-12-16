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
    page_title="Air Quality Dashboard",
    layout="wide",
    page_icon="🌫"
)

# --------------------------------------------------
# CUSTOM CSS (UI ENHANCEMENT)
# --------------------------------------------------
st.markdown("""
<style>
/* General */
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
    color: white;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: #e5e7eb;
}

/* Title */
.dashboard-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.dashboard-subtitle {
    color: #6b7280;
    margin-bottom: 1.5rem;
}

/* KPI Cards */
.kpi-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.2rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    text-align: center;
}
.kpi-title {
    color: #6b7280;
    font-size: 0.9rem;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0f172a;
}

/* Section Headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 600;
    margin: 1.5rem 0 0.5rem 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.85rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

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
    "Navigation",
    [
        "🔮 Prediction",
        "📊 EDA",
        "🗺 India Map",
        "🏙 City Map"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("CMP7005 · Streamlit Cloud")

# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------
st.markdown('<div class="dashboard-title">Air Quality Analysis & Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Interactive dashboard for air pollution insights</div>', unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "merged_data.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

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

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Average PM2.5</div>
        <div class="kpi-value">{df['PM2.5'].mean():.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Maximum PM2.5</div>
        <div class="kpi-value">{df['PM2.5'].max():.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Cities Covered</div>
        <div class="kpi-value">{df['City'].nunique()}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# PAGE: PREDICTION
# --------------------------------------------------
if page == "🔮 Prediction":
    st.markdown('<div class="section-header">PM2.5 Prediction</div>', unsafe_allow_html=True)

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
        model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
        pred = model.predict(np.array([[so2, no2, co, o3, pm10, nh3]]))[0]
        st.success(f"Predicted PM2.5: {pred:.2f} µg/m³")

# --------------------------------------------------
# PAGE: EDA
# --------------------------------------------------
elif page == "📊 EDA":
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    city = st.selectbox("Filter by City", ["All"] + sorted(df["City"].unique()))
    data = df if city == "All" else df[df["City"] == city]

    st.dataframe(data.head(20))

    st.bar_chart(data["AQI Category"].value_counts())

    fig, ax = plt.subplots()
    ax.hist(data["PM2.5"], bins=30)
    st.pyplot(fig)

# --------------------------------------------------
# PAGE: INDIA MAP
# --------------------------------------------------
elif page == "🗺 India Map":
    st.markdown('<div class="section-header">India Air Quality Map</div>', unsafe_allow_html=True)

    city_pm = df.groupby("City")["PM2.5"].mean().reset_index()
    m = folium.Map(location=[22.5, 80.0], zoom_start=5)

    for _, row in city_pm.iterrows():
        if row["City"] in CITY_COORDS:
            lat, lon = CITY_COORDS[row["City"]]
            color = "green" if row["PM2.5"] <= 30 else "orange" if row["PM2.5"] <= 60 else "red"
            folium.CircleMarker([lat, lon], radius=8, color=color, fill=True).add_to(m)

    st_folium(m, width=1000, height=500)

# --------------------------------------------------
# PAGE: CITY MAP
# --------------------------------------------------
elif page == "🏙 City Map":
    st.markdown('<div class="section-header">City Air Quality View</div>', unsafe_allow_html=True)

    city = st.selectbox("Select City", sorted(df["City"].unique()))
    avg_pm = df[df["City"] == city]["PM2.5"].mean()

    lat, lon = CITY_COORDS[city]
    m = folium.Map(location=[lat, lon], zoom_start=11)

    folium.CircleMarker(
        [lat, lon],
        radius=15,
        color="red",
        fill=True
    ).add_to(m)

    st.markdown(f"**Average PM2.5:** {avg_pm:.2f} µg/m³")
    st.markdown(f"**AQI Category:** {aqi_category(avg_pm)}")
    st_folium(m, width=900, height=500)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown('<div class="footer">CMP7005 · Air Quality Analysis · Streamlit Cloud</div>', unsafe_allow_html=True)
