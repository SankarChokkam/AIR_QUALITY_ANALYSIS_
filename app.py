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
# CLEAR CACHES (at the beginning)
# --------------------------------------------------
st.cache_data.clear()
st.cache_resource.clear()

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
# PAGE CONFIG (MUST be first Streamlit command)
# --------------------------------------------------
st.set_page_config(
    page_title="Data Analytics and Visualisation – CMP7005",
    layout="wide",
    page_icon="🌫"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "merged_data.csv")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.error(f"❌ Data file not found at: {DATA_PATH}")
        # Create empty DataFrame with expected columns to prevent crashes
        return pd.DataFrame(columns=["City", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "NH3"])

df = load_data()

# --------------------------------------------------
# AQI FUNCTIONS
# --------------------------------------------------
def aqi_category(pm):
    if pd.isna(pm):
        return "Unknown"
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

def get_aqi_color(category):
    return {
        "Good": "#00E400",
        "Satisfactory": "#FFFF00",
        "Moderate": "#FF7E00",
        "Poor": "#FF0000",
        "Very Poor": "#8F3F97",
        "Severe": "#7E0023",
        "Unknown": "#808080"
    }.get(category, "#808080")

# Only apply AQI category if DataFrame has PM2.5 column
if not df.empty and "PM2.5" in df.columns:
    df["AQI Category"] = df["PM2.5"].apply(aqi_category)
else:
    df["AQI Category"] = "Unknown"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1N317Atsm71Is04H_P711V3Dk-jr5y1ou"
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("⬇ Downloading ML model..."):
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        else:
            st.error("Failed to download model file")
            return None
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        return None

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model error: {e}")
    model = None

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

# ==================================================
# HOME DASHBOARD
# ==================================================
if page == "🏠 Home Dashboard":
    st.title("🌫 Air Quality Analysis & Prediction Dashboard")
    st.markdown(f"### {COURSE_NAME} Project")
    
    st.markdown("---")
    
    # Check if data is loaded
    if df.empty:
        st.warning("⚠️ No data loaded. Please ensure 'merged_data.csv' exists in the project directory.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Cities", df["City"].nunique())
        with c2:
            st.metric("Records", f"{len(df):,}")
        with c3:
            if "PM2.5" in df.columns:
                avg_pm25 = df["PM2.5"].mean()
                st.metric("Avg PM2.5", f"{avg_pm25:.2f} µg/m³")
            else:
                st.metric("Avg PM2.5", "N/A")
        
        st.markdown("---")
        
        # Show data summary
        with st.expander("📋 Data Overview"):
            st.write(f"**Data Shape:** {df.shape}")
            st.write("**Columns:**", list(df.columns))
            st.write("**Data Types:**")
            st.write(df.dtypes)
        
        # Show sample data
        with st.expander("👀 Sample Data (First 10 rows)"):
            st.dataframe(df.head(10))

# ==================================================
# PREDICTION PAGE
# ==================================================
elif page == "🔮 PM2.5 Prediction":
    st.title("🔮 PM2.5 Concentration Predictor")
    st.markdown("Predict PM2.5 using Machine Learning")
    
    if model is None:
        st.error("❌ Model not available. Please check if model.pkl exists or can be downloaded.")
        st.info("If you're running this locally, ensure you have an internet connection to download the model.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            so2 = st.number_input("SO₂ (µg/m³)", 0.0, 500.0, 10.0, step=0.1)
            no2 = st.number_input("NO₂ (µg/m³)", 0.0, 500.0, 20.0, step=0.1)
            co = st.number_input("CO (mg/m³)", 0.0, 10.0, 1.0, step=0.1)
        
        with col2:
            o3 = st.number_input("O₃ (µg/m³)", 0.0, 500.0, 30.0, step=0.1)
            pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 50.0, step=1.0)
            nh3 = st.number_input("NH₃ (µg/m³)", 0.0, 500.0, 15.0, step=0.1)
        
        if st.button("Predict PM2.5", type="primary"):
            try:
                pred = model.predict([[so2, no2, co, o3, pm10, nh3]])[0]
                category = aqi_category(pred)
                color = get_aqi_color(category)
                
                st.success(f"### Predicted PM2.5: **{pred:.2f} µg/m³**")
                st.markdown(f'<h3 style="color:{color};">AQI Category: {category}</h3>', unsafe_allow_html=True)
                
                # Show AQI scale reference
                with st.expander("📊 AQI Scale Reference"):
                    st.markdown("""
                    | PM2.5 (µg/m³) | Category | Color |
                    |---------------|----------|-------|
                    | 0-30 | Good | 🟢 |
                    | 31-60 | Satisfactory | 🟡 |
                    | 61-90 | Moderate | 🟠 |
                    | 91-120 | Poor | 🔴 |
                    | 121-250 | Very Poor | 🟣 |
                    | 251+ | Severe | 🟤 |
                    """)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ==================================================
# EDA PAGE
# ==================================================
elif page == "📊 EDA & Visualisation":
    st.title("📊 Exploratory Data Analysis")
    
    if df.empty:
        st.warning("No data available for EDA")
    else:
        # Data preview
        st.subheader("Data Preview")
        rows_to_show = st.slider("Number of rows to display", 10, 100, 50)
        st.dataframe(df.head(rows_to_show))
        
        # Statistics
        st.subheader("📈 Basic Statistics")
        st.write(df.describe())
        
        # City distribution
        st.subheader("🏙 City Distribution")
        city_counts = df["City"].value_counts()
        fig, ax = plt.subplots()
        city_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel("City")
        ax.set_ylabel("Number of Records")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
        # PM2.5 distribution
        if "PM2.5" in df.columns:
            st.subheader("📊 PM2.5 Distribution")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram
            df["PM2.5"].plot(kind='hist', bins=30, ax=ax1, edgecolor='black')
            ax1.set_xlabel("PM2.5 (µg/m³)")
            ax1.set_ylabel("Frequency")
            ax1.set_title("PM2.5 Distribution")
            
            # Box plot
            df["PM2.5"].plot(kind='box', ax=ax2)
            ax2.set_title("PM2.5 Box Plot")
            
            st.pyplot(fig)

# ==================================================
# INDIA MAP
# ==================================================
elif page == "🗺 India AQ Map":
    st.title("🗺 India Air Quality Map")
    
    if df.empty:
        st.warning("No data available for map visualization")
    else:
        # Calculate average PM2.5 per city
        if "PM2.5" in df.columns:
            city_pm = df.groupby("City")["PM2.5"].mean().reset_index()
            
            # Create map
            india_map = folium.Map(location=[22.5, 80], zoom_start=5, tiles='CartoDB positron')
            
            for _, row in city_pm.iterrows():
                city_name = row["City"]
                pm_value = row["PM2.5"]
                
                if city_name in CITY_COORDS:
                    lat, lon = CITY_COORDS[city_name]
                    category = aqi_category(pm_value)
                    color = get_aqi_color(category)
                    
                    # Create popup content
                    popup_text = f"""
                    <b>{city_name}</b><br>
                    PM2.5: {pm_value:.1f} µg/m³<br>
                    AQI: {category}<br>
                    <div style="width:20px;height:20px;background-color:{color};display:inline-block;"></div>
                    """
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=10 + (pm_value / 20),  # Scale radius with PM2.5
                        popup=folium.Popup(popup_text, max_width=300),
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        tooltip=f"{city_name}: {pm_value:.1f} µg/m³"
                    ).add_to(india_map)
            
            # Add legend
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 220px; height: 280px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px; border-radius: 5px;">
            <b>AQI Categories</b><br>
            <div style="width:20px;height:20px;background-color:#00E400;display:inline-block;"></div> Good (0-30)<br>
            <div style="width:20px;height:20px;background-color:#FFFF00;display:inline-block;"></div> Satisfactory (31-60)<br>
            <div style="width:20px;height:20px;background-color:#FF7E00;display:inline-block;"></div> Moderate (61-90)<br>
            <div style="width:20px;height:20px;background-color:#FF0000;display:inline-block;"></div> Poor (91-120)<br>
            <div style="width:20px;height:20px;background-color:#8F3F97;display:inline-block;"></div> Very Poor (121-250)<br>
            <div style="width:20px;height:20px;background-color:#7E0023;display:inline-block;"></div> Severe (251+)<br>
            <br>
            <i>Marker size indicates pollution level</i>
            </div>
            '''
            
            india_map.get_root().html.add_child(folium.Element(legend_html))
            
            # Display map
            st_folium(india_map, width=1000, height=600, returned_objects=[])
        else:
            st.error("PM2.5 column not found in data")

# ==================================================
# CITY MAP
# ==================================================
elif page == "🏙 City AQ Map":
    st.title("🏙 City Air Quality Analysis")
    
    if df.empty:
        st.warning("No data available")
    else:
        city = st.selectbox("Select City", sorted(df["City"].unique()))
        city_df = df[df["City"] == city]
        
        if not city_df.empty:
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                if "PM2.5" in city_df.columns:
                    avg_pm25 = city_df["PM2.5"].mean()
                    category = aqi_category(avg_pm25)
                    st.metric("Average PM2.5", f"{avg_pm25:.2f} µg/m³", category)
            
            with col2:
                if "PM10" in city_df.columns:
                    avg_pm10 = city_df["PM10"].mean()
                    st.metric("Average PM10", f"{avg_pm10:.2f} µg/m³")
            
            with col3:
                st.metric("Records", len(city_df))
            
            # Show city data
            st.subheader(f"📋 Data for {city}")
            
            # Filter columns to show
            pollutant_cols = [col for col in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "NH3"] 
                            if col in city_df.columns]
            
            if pollutant_cols:
                display_df = city_df[["City"] + pollutant_cols].head(100)
                st.dataframe(display_df)
                
                # Time series plot if date column exists
                date_cols = [col for col in city_df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols and "PM2.5" in city_df.columns:
                    st.subheader("📈 PM2.5 Over Time")
                    # Use first date column found
                    date_col = date_cols[0]
                    try:
                        # Try to convert to datetime
                        city_df[date_col] = pd.to_datetime(city_df[date_col])
                        time_series = city_df.sort_values(date_col)
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(time_series[date_col], time_series["PM2.5"], marker='o', linestyle='-', markersize=3)
                        ax.set_xlabel("Date")
                        ax.set_ylabel("PM2.5 (µg/m³)")
                        ax.set_title(f"PM2.5 Levels in {city}")
                        ax.tick_params(axis='x', rotation=45)
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    except:
                        st.info("Could not create time series plot. Date format may be incompatible.")
            else:
                st.warning("No pollutant columns found in data")
        else:
            st.warning(f"No data found for {city}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(f"""
**📘 Course:** {COURSE_NAME}  
**🎓 Academic Project**  
**🌍 Making environmental data accessible and actionable**  
*Note: This is a demonstration application for educational purposes.*
""")
