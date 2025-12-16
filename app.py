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
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙ Controls")
st.sidebar.markdown("Interactive air quality analysis dashboard")

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🌫 Air Quality Analysis & Prediction")
st.markdown("CMP7005 Practical – Streamlit Cloud Deployment")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "merged_data.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("🌫 Avg PM2.5", f"{df['PM2.5'].mean():.2f}")
c2.metric("🚨 Max PM2.5", f"{df['PM2.5'].max():.2f}")
c3.metric("🏙 Cities Covered", df["City"].nunique())

st.markdown("---")

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

model = load_model()

# --------------------------------------------------
# AQI CATEGORY FUNCTION
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
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 PM2.5 Prediction", "📊 EDA & Visualisation", "🗺 India AQ Map", "🏙 City AQ Map"]
)

# ==================================================
# TAB 1 – PREDICTION
# ==================================================
with tab1:
    st.subheader("Predict PM2.5 Concentration")

    col1, col2 = st.columns(2)

    with col1:
        so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=10.0)
        no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=20.0)
        co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.0)

    with col2:
        o3 = st.number_input("O₃ (µg/m³)", min_value=0.0, value=30.0)
        pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=50.0)
        nh3 = st.number_input("NH₃ (µg/m³)", min_value=0.0, value=15.0)

    if st.button("🔮 Predict PM2.5"):
        X = np.array([[so2, no2, co, o3, pm10, nh3]])
        pred = model.predict(X)[0]

        st.success(f"Predicted PM2.5: {pred:.2f} µg/m³")
        st.progress(int(min(pred, 300) / 300 * 100))

# ==================================================
# TAB 2 – FILTERED EDA
# ==================================================
with tab2:
    st.subheader("Exploratory Data Analysis ")

    st.markdown("### 🔍 Filters")

    f1, f2, f3 = st.columns(3)

    with f1:
        city_filter = st.selectbox(
            "City",
            ["All"] + sorted(df["City"].unique())
        )

    with f2:
        pm25_range = st.slider(
            "PM2.5 Range (µg/m³)",
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

    # Apply filters
    filtered_df = df.copy()

    if city_filter != "All":
        filtered_df = filtered_df[filtered_df["City"] == city_filter]

    filtered_df = filtered_df[
        (filtered_df["PM2.5"] >= pm25_range[0]) &
        (filtered_df["PM2.5"] <= pm25_range[1])
    ]

    filtered_df = filtered_df[filtered_df["AQI Category"].isin(aqi_filter)]

    st.markdown("---")

    st.dataframe(filtered_df.head(20))

    # AQI Distribution
    st.subheader("AQI Category Distribution")
    st.bar_chart(filtered_df["AQI Category"].value_counts())

    # PM2.5 Histogram
    st.subheader("PM2.5 Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered_df["PM2.5"], bins=30)
    ax.set_xlabel("PM2.5")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Scatter Plot
    st.subheader("PM2.5 vs PM10")
    fig, ax = plt.subplots()
    ax.scatter(filtered_df["PM10"], filtered_df["PM2.5"], alpha=0.5)
    ax.set_xlabel("PM10")
    ax.set_ylabel("PM2.5")
    st.pyplot(fig)

    # Pollutant Comparison
    st.subheader("Average Pollutant Levels")
    pollutants = st.multiselect(
        "Select Pollutants",
        ["PM10", "SO2", "NO2", "CO", "O3", "NH3"],
        default=["PM10", "NO2", "SO2"]
    )

    if pollutants:
        avg_vals = filtered_df[pollutants].mean()
        fig, ax = plt.subplots()
        ax.bar(avg_vals.index, avg_vals.values)
        ax.set_ylabel("Average Concentration")
        st.pyplot(fig)

    # Stats
    st.subheader("Statistical Summary")
    st.dataframe(filtered_df.describe())

# ==================================================
# TAB 3 – INDIA MAP
# ==================================================
with tab3:
    st.subheader("India Air Quality Map (Average PM2.5)")

    city_pm = df.groupby("City")["PM2.5"].mean().reset_index()
    india_map = folium.Map(location=[22.5, 80.0], zoom_start=5)

    for _, row in city_pm.iterrows():
        if row["City"] in CITY_COORDS:
            lat, lon = CITY_COORDS[row["City"]]
            color = "green" if row["PM2.5"] <= 30 else "orange" if row["PM2.5"] <= 60 else "red"

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                tooltip=f"{row['City']} | PM2.5: {row['PM2.5']:.2f}",
                color=color,
                fill=True,
                fill_opacity=0.8
            ).add_to(india_map)

    st_folium(india_map, width=1000, height=500)

# ==================================================
# TAB 4 – CITY MAP
# ==================================================
with tab4:
    st.subheader("City-Specific Air Quality Map")

    city = st.selectbox("Select City", sorted(df["City"].unique()))

    city_df = df[df["City"] == city]
    avg_pm = city_df["PM2.5"].mean()

    lat, lon = CITY_COORDS.get(city, [22.5, 80.0])

    city_map = folium.Map(location=[lat, lon], zoom_start=11)

    color = "green" if avg_pm <= 30 else "orange" if avg_pm <= 60 else "red"

    folium.CircleMarker(
        location=[lat, lon],
        radius=15,
        tooltip=f"{city} | Avg PM2.5: {avg_pm:.2f}",
        color=color,
        fill=True,
        fill_opacity=0.85
    ).add_to(city_map)

    st.markdown(f"**Average PM2.5:** {avg_pm:.2f} µg/m³")
    st.markdown(f"**AQI Category:** {aqi_category(avg_pm)}")

    st_folium(city_map, width=900, height=500)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("📘 Course: CMP7005 – Air Quality Analysis")
st.markdown("☁ Deployed on Streamlit Cloud")
