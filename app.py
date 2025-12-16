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
# BASE DIRECTORY
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# CITY COORDINATES (KNOWN & VERIFIED)
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
# COMPUTE VALID CITIES (CRITICAL FIX)
# --------------------------------------------------
dataset_cities = set(df["City"].unique())
coord_cities = set(CITY_COORDS.keys())

VALID_CITIES = sorted(dataset_cities.intersection(coord_cities))

# ==================================================
# HOME
# ==================================================
if page == "🏠 Home":
    st.title("Air Quality Analysis Dashboard")
    st.write(
        "Predict, analyze, and visualize air pollution trends across Indian cities "
        "using machine learning, exploratory data analysis, and geospatial mapping."
    )

# ==================================================
# PREDICTION
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
        X = np.array([[so2, no2, co, o3, pm10, nh3]])
        pred = model.predict(X)[0]
        st.success(f"Predicted PM2.5: {pred:.2f} µg/m³")

# ==================================================
# EDA (UNCHANGED, FILTER-DRIVEN)
# ==================================================
elif page == "📊 EDA & Visualisation":
    st.subheader("Exploratory Data Analysis")

    city_filter = st.selectbox("City", ["All"] + sorted(dataset_cities))
    pm25_range = st.slider(
        "PM2.5 Range",
        float(df["PM2.5"].min()),
        float(df["PM2.5"].max()),
        (float(df["PM2.5"].min()), float(df["PM2.5"].max()))
    )

    filtered_df = df.copy()

    if city_filter != "All":
        filtered_df = filtered_df[filtered_df["City"] == city_filter]

    filtered_df = filtered_df[
        (filtered_df["PM2.5"] >= pm25_range[0]) &
        (filtered_df["PM2.5"] <= pm25_range[1])
    ]

    st.dataframe(filtered_df.head(20))
    st.bar_chart(filtered_df["AQI Category"].value_counts())

    fig, ax = plt.subplots()
    ax.hist(filtered_df["PM2.5"], bins=30)
    st.pyplot(fig)

# ==================================================
# INDIA MAP
# ==================================================
elif page == "🗺 India AQ Map":
    st.subheader("India Air Quality Map (Average PM2.5)")

    city_pm = (
        df[df["City"].isin(VALID_CITIES)]
        .groupby("City")["PM2.5"]
        .mean()
        .reset_index()
    )

    india_map = folium.Map(
        location=[22.5, 80.0],
        zoom_start=5,
        tiles="CartoDB positron"
    )

    for _, row in city_pm.iterrows():
        lat, lon = CITY_COORDS[row["City"]]

        folium.CircleMarker(
            [lat, lon],
            radius=8,
            fill=True,
            fill_opacity=0.8,
            popup=f"{row['City']} | Avg PM2.5: {row['PM2.5']:.2f}"
        ).add_to(india_map)

    st_folium(india_map, width=1000, height=500)

# ==================================================
# CITY MAP (FULLY FIXED)
# ==================================================
elif page == "🏙 City AQ Map":
    st.subheader("City-Specific Air Quality Map")

    if not VALID_CITIES:
        st.error("No cities with valid coordinates available.")
        st.stop()

    city = st.selectbox(
        "Select City",
        VALID_CITIES,
        key="city_map_select"
    )

    city_df = df[df["City"] == city]
    avg_pm = city_df["PM2.5"].mean()
    category = aqi_category(avg_pm)

    lat, lon = CITY_COORDS[city]

    city_map = folium.Map(
        location=[lat, lon],
        zoom_start=10,
        tiles="CartoDB positron",
        control_scale=True
    )

    folium.CircleMarker(
        [lat, lon],
        radius=18,
        color="red",
        fill=True,
        fill_opacity=0.85,
        popup=f"""
        <b>{city}</b><br>
        Avg PM2.5: {avg_pm:.2f} µg/m³<br>
        AQI Category: {category}
        """
    ).add_to(city_map)

    st.markdown(f"**Average PM2.5:** {avg_pm:.2f} µg/m³")
    st.markdown(f"**AQI Category:** {category}")

    st_folium(city_map, width=900, height=500)
