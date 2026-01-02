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
# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)
# --------------------------------------------------
st.set_page_config(
    page_title="Air Quality Analysis – Data Analytics and Visualisation",
    layout="wide",
    page_icon="🌫"
)

# --------------------------------------------------
# BASE DIRECTORY
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# AQI FUNCTION (MUST BE DEFINED BEFORE USE)
# --------------------------------------------------
def aqi_category(pm):
    if pd.isna(pm):
        return "Unknown"
    elif pm <= 30:
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

def get_aqi_color(category):
    return {
        "Good": "#00E400",
        "Satisfactory": "#FFFF00",
        "Moderate": "#FF7E00",
        "Poor": "#FF0000",
        "Very Poor": "#8F3F97",
        "Severe": "#7E0023",
        "Unknown": "#999999"
    }.get(category, "#999999")

# --------------------------------------------------
# LOAD DATA (NO ROW DROPPING)
# --------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "merged_data.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Normalize city names
    if "City" in df.columns:
        df["City"] = df["City"].astype(str).str.strip()

    return df

df = load_data()

# Apply AQI AFTER function definition
df["AQI Category"] = df["PM2.5"].apply(aqi_category)

# --------------------------------------------------
# CITY COORDINATES (ONLY USED IF CITY EXISTS)
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
    "Lucknow": [26.8467, 80.9462],
}

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("📊 Dashboard Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home Dashboard",
        "📊 EDA & Visualisation",
        "🗺 India AQ Map",
        "🏙 City AQ Map",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Data Analytics and Visualisation")

with st.sidebar.expander("ℹ️ Project Info"):
    st.markdown("""
**Academic Project**

**Course:** Data Analytics and Visualisation  
**Dataset:** Indian city air-quality measurements  
**Cities:** 26 (dataset-driven)  
""")

# ==================================================
# HOME DASHBOARD
# ==================================================
if page == "🏠 Home Dashboard":
    st.title("🌫 Air Quality Analysis Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Cities", df["City"].nunique())
    c3.metric("Avg PM2.5", f"{df['PM2.5'].mean():.1f}")

    st.markdown("---")

    st.subheader("📊 City-wise Average PM2.5")
    city_avg = df.groupby("City")["PM2.5"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(city_avg.index, city_avg.values)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xticklabels(city_avg.index, rotation=45, ha="right")
    st.pyplot(fig)

# ==================================================
# EDA PAGE
# ==================================================
elif page == "📊 EDA & Visualisation":
    st.title("📊 Exploratory Data Analysis")

    city = st.selectbox("City", ["All"] + sorted(df["City"].unique()))

    plot_df = df.copy()
    if city != "All":
        plot_df = plot_df[plot_df["City"] == city]

    plot_df = plot_df.dropna(subset=["PM2.5", "PM10"])

    st.write(f"Records shown: {len(plot_df)}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(plot_df["PM2.5"], bins=30, edgecolor="black")
    ax.set_title("PM2.5 Distribution")
    st.pyplot(fig)

# ==================================================
# INDIA MAP
# ==================================================
elif page == "🗺 India AQ Map":
    st.title("🗺 India Air Quality Map")

    city_avg = df.groupby("City")["PM2.5"].mean().reset_index()

    india_map = folium.Map(location=[22.5, 80.0], zoom_start=5)

    for _, row in city_avg.iterrows():
        city = row["City"]

        if city not in CITY_COORDS:
            continue  # SAFE SKIP

        lat, lon = CITY_COORDS[city]
        pm = row["PM2.5"]
        cat = aqi_category(pm)

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=f"{city}<br>PM2.5: {pm:.1f}<br>{cat}",
            color=get_aqi_color(cat),
            fill=True,
            fill_opacity=0.7
        ).add_to(india_map)

    st_folium(india_map, width=1000, height=600)

# ==================================================
# CITY MAP
# ==================================================
elif page == "🏙 City AQ Map":
    st.title("🏙 City-wise Analysis")

    city = st.selectbox("Select City", sorted(df["City"].unique()))
    city_df = df[df["City"] == city].dropna(subset=["PM2.5"])

    st.metric("Avg PM2.5", f"{city_df['PM2.5'].mean():.1f}")

    if city in CITY_COORDS:
        lat, lon = CITY_COORDS[city]
    else:
        lat, lon = 22.5, 80.0  # fallback

    city_map = folium.Map(location=[lat, lon], zoom_start=10)
    folium.Marker([lat, lon], popup=city).add_to(city_map)

    st_folium(city_map, width=700, height=400)

    st.dataframe(city_df.head(100))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
**Course:** Data Analytics and Visualisation  
**Deployment:** Streamlit Cloud  
**Project:** Air Quality Analysis  
""")
