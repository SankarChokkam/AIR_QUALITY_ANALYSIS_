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
# SIDEBAR (VISUAL ONLY)
# --------------------------------------------------
st.sidebar.title("⚙ Controls")
st.sidebar.markdown("Explore air quality data interactively")

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
# KPI METRICS (STYLED)
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
# AQI FUNCTIONS
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

def aqi_badge(category):
    colors = {
        "Good": "#22c55e",
        "Satisfactory": "#a3e635",
        "Moderate": "#facc15",
        "Poor": "#fb923c",
        "Very Poor": "#ef4444",
        "Severe": "#7f1d1d"
    }
    return f"<span style='color:white;background:{colors[category]};padding:6px 14px;border-radius:14px;font-weight:600'>{category}</span>"

df["AQI Category"] = df["PM2.5"].apply(aqi_category)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🔮 PM2.5 Prediction", "📊 EDA & Visualisation", "🗺 Air Quality Map"]
)

# ==================================================
# 🔮 TAB 1 – PREDICTION
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

        st.markdown(
            f"""
            <div style="padding:20px;border-radius:14px;
                        background-color:#0E1117;
                        border-left:6px solid #22c55e">
                <h3>Predicted PM2.5: {pred:.2f} µg/m³</h3>
                <p>AQI Category: {aqi_badge(aqi_category(pred))}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(int(min(pred, 300) / 300 * 100))

# ==================================================
# 📊 TAB 2 – EDA & VISUALISATION
# ==================================================
with tab2:
    st.subheader("Exploratory Data Analysis (EDA)")

    selected_city = st.sidebar.selectbox(
        "🌆 Filter by City",
        ["All"] + sorted(df["City"].unique().tolist())
    )

    filtered_df = df if selected_city == "All" else df[df["City"] == selected_city]

    with st.expander("📄 Sample Records", expanded=True):
        st.dataframe(filtered_df.head())

    with st.expander("📊 AQI Category Distribution"):
        st.bar_chart(filtered_df["AQI Category"].value_counts())

    with st.expander("📈 PM2.5 Distribution"):
        min_pm, max_pm = st.slider(
            "Select PM2.5 Range",
            int(df["PM2.5"].min()),
            int(df["PM2.5"].max()),
            (0, 200)
        )
        range_df = filtered_df[
            (filtered_df["PM2.5"] >= min_pm) &
            (filtered_df["PM2.5"] <= max_pm)
        ]

        fig, ax = plt.subplots()
        ax.hist(range_df["PM2.5"], bins=30)
        ax.set_xlabel("PM2.5")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with st.expander("🏙 City-wise Average PM2.5"):
    city_avg = (
        filtered_df
        .groupby("City", as_index=False)["PM2.5"]
        .mean()
        .set_index("City")
    )
    st.bar_chart(city_avg)

    with st.expander("🔗 Correlation Heatmap"):
        corr = filtered_df[
            ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "NH3"]
        ].corr()

        fig, ax = plt.subplots()
        im = ax.imshow(corr)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45)
        ax.set_yticklabels(corr.columns)
        plt.colorbar(im)
        st.pyplot(fig)

    with st.expander("🔬 Pollutant vs PM2.5"):
        pollutant = st.selectbox(
            "Select Pollutant",
            ["PM10", "SO2", "NO2", "CO", "O3", "NH3"]
        )

        fig, ax = plt.subplots()
        ax.scatter(filtered_df[pollutant], filtered_df["PM2.5"], alpha=0.5)
        ax.set_xlabel(pollutant)
        ax.set_ylabel("PM2.5")
        st.pyplot(fig)

    with st.expander("📑 Dataset Statistics"):
        st.dataframe(filtered_df.describe())

# ==================================================
# 🗺 TAB 3 – MAP
# ==================================================
with tab3:
    st.subheader("India Air Quality Map (Average PM2.5)")
    st.caption("🟢 Good | 🟠 Moderate | 🔴 Poor")

    city_pm = df.groupby("City")["PM2.5"].mean().reset_index()
    m = folium.Map(location=[22.5, 80.0], zoom_start=5)

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
            ).add_to(m)

    st_folium(m, width=1000, height=500)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("📘 Course: CMP7005 – Air Quality Analysis")
st.markdown("☁ Deployed on Streamlit Cloud")
