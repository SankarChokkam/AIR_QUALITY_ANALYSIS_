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
    page_title="Air Quality Analysis – Data analytics and Visualisation",
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
st.sidebar.markdown("### 📚 Data analytics and Visualisation – Air Quality Analysis")

# Add project info in sidebar
with st.sidebar.expander("ℹ️ Project Info"):
    st.markdown("""
    **Academic Project**
    
    **Course:** Data analytics and Visualisation
    
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
    df = pd.read_csv(DATA_PATH)
    # Clean data - remove NaN values in essential columns
    df = df.dropna(subset=['PM2.5', 'PM10'], how='any')
    return df

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
    # Test if model can predict
    test_input = np.array([[10.0, 20.0, 1.0, 30.0, 50.0, 15.0]]).reshape(1, -1)
    test_pred = model.predict(test_input)
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Model error: {str(e)}")
    model = None

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

# Apply AQI category to dataframe
df["AQI Category"] = df["PM2.5"].apply(aqi_category)

# --------------------------------------------------
# COMMON FUNCTIONS
# --------------------------------------------------
def get_aqi_color(category):
    """Return color for AQI category"""
    colors = {
        "Good": "#00E400",
        "Satisfactory": "#FFFF00",
        "Moderate": "#FF7E00",
        "Poor": "#FF0000",
        "Very Poor": "#8F3F97",
        "Severe": "#7E0023"
    }
    return colors.get(category, "#808080")

# ==================================================
# PAGE 1 – HOME DASHBOARD
# ==================================================
if page == "🏠 Home Dashboard":
    # Hero Section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🌫 Air Quality Analysis & Prediction Dashboard")
        st.markdown("### Data analytics and Visualisation - Advanced Data Analytics Project")
    
    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("**📊 Live Stats**")
        st.metric("Cities", df["City"].nunique())
        st.metric("Records", f"{len(df):,}")
    
    st.markdown("---")
    
    # Project Overview
    st.markdown("## 📋 Project Overview")
    
    overview_col1, overview_col2 = st.columns(2)
    
    with overview_col1:
        st.markdown("""
        ### 🎯 Objectives
        
        This project aims to:
        
        1. **Analyze** air quality patterns across major Indian cities
        2. **Predict** PM2.5 concentrations using machine learning
        3. **Visualize** pollution hotspots and trends
        4. **Provide** actionable insights for environmental monitoring
        
        ### 🔧 Features
        
        - **Real-time PM2.5 prediction** using Random Forest model
        - **Interactive visualizations** with filtering capabilities
        - **Geospatial analysis** on interactive maps
        - **Comprehensive EDA** tools for data exploration
        - **City-specific analysis** with detailed statistics
        """)
    
    with overview_col2:
        st.markdown("""
        ### 📁 Dataset Information
        
        **Source:** Multi-city air quality monitoring stations
        
        **Parameters:**
        - PM2.5 (Primary target)
        - PM10
        - SO₂ (Sulfur Dioxide)
        - NO₂ (Nitrogen Dioxide)
        - CO (Carbon Monoxide)
        - O₃ (Ozone)
        - NH₃ (Ammonia)
        
        **Cities Covered:** 10 major Indian cities
        
        **Time Period:** Multi-year measurements
        
        ### 🛠️ Technologies Used
        
        - **Python** (Streamlit, Pandas, NumPy)
        - **Machine Learning** (Scikit-learn)
        - **Visualization** (Matplotlib, Folium)
        - **Cloud Deployment** (Streamlit Cloud)
        """)
    
    st.markdown("---")
    
    # Quick Stats Dashboard
    st.markdown("## 📈 Air Quality Quick Stats")
    
    # Overall statistics in columns
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        avg_pm25 = df['PM2.5'].mean()
        st.metric("🌫 Average PM2.5", f"{avg_pm25:.1f} µg/m³", 
                 delta=f"{aqi_category(avg_pm25)}")
    
    with stats_col2:
        max_pm25 = df['PM2.5'].max()
        st.metric("🔥 Maximum PM2.5", f"{max_pm25:.1f} µg/m³",
                 delta="Most polluted reading")
    
    with stats_col3:
        min_pm25 = df['PM2.5'].min()
        st.metric("🌿 Minimum PM2.5", f"{min_pm25:.1f} µg/m³",
                 delta="Cleanest reading")
    
    with stats_col4:
        total_cities = df["City"].nunique()
        st.metric("🏙 Cities Covered", total_cities)
    
    st.markdown("---")
    
    # City-wise Performance
    st.markdown("## 🏙 City-wise Air Quality Summary")
    
    # Calculate city averages
    city_stats = df.groupby('City').agg({
        'PM2.5': ['mean', 'max', 'min', 'count']
    }).round(2)
    
    # Flatten column names
    city_stats.columns = ['_'.join(col).strip() for col in city_stats.columns.values]
    city_stats = city_stats.reset_index()
    
    # Add AQI category
    city_stats['AQI_Category'] = city_stats['PM2.5_mean'].apply(aqi_category)
    
    # Display in a nice format
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 📊 City Rankings by Average PM2.5")
        city_stats_sorted = city_stats.sort_values('PM2.5_mean', ascending=False)
        
        # Create a styled table
        for idx, row in city_stats_sorted.iterrows():
            color = get_aqi_color(row['AQI_Category'])
            with st.container():
                cols = st.columns([3, 2, 2, 2, 2])
                with cols[0]:
                    st.markdown(f"**{row['City']}**")
                with cols[1]:
                    st.markdown(f"`{row['PM2.5_mean']:.1f}`")
                with cols[2]:
                    st.markdown(f"`{row['PM2.5_max']:.1f}`")
                with cols[3]:
                    st.markdown(f"`{row['PM2.5_min']:.1f}`")
                with cols[4]:
                    st.markdown(f"<span style='color:{color}; font-weight:bold;'>{row['AQI_Category']}</span>", 
                               unsafe_allow_html=True)
                st.markdown("---")
    
    with col2:
        st.markdown("### 📊 AQI Distribution")
        
        # Calculate AQI distribution
        aqi_dist = df['AQI Category'].value_counts()
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [get_aqi_color(cat) for cat in aqi_dist.index]
        wedges, texts, autotexts = ax.pie(aqi_dist.values, labels=aqi_dist.index, 
                                         autopct='%1.1f%%', colors=colors,
                                         startangle=90)
        
        # Improve label appearance
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        
        ax.set_title('Overall AQI Category Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        
        # AQI Legend
        st.markdown("### 🎨 AQI Color Guide")
        categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        for category in categories:
            color = get_aqi_color(category)
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 20px; background-color: {color}; 
                         border-radius: 3px; margin-right: 10px; border: 1px solid #ccc;"></div>
                <span style="font-weight: bold;">{category}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features Preview
    st.markdown("## 🚀 Dashboard Features Preview")
    
    feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
    
    with feature_col1:
        st.markdown("""
        ### 🔮 PM2.5 Prediction
        **Real-time ML predictions**
        
        - Input 6 parameters
        - Get instant PM2.5 predictions
        - View health advisories
        - Feature importance analysis
        """)
    
    with feature_col2:
        st.markdown("""
        ### 📊 EDA & Visualization
        **Interactive data exploration**
        
        - Filter by city & AQI
        - Scatter plots & histograms
        - Time series analysis
        - Correlation insights
        """)
    
    with feature_col3:
        st.markdown("""
        ### 🗺 India AQ Map
        **Geospatial analysis**
        
        - Interactive Folium maps
        - Color-coded cities
        - Top/Bottom rankings
        - Detailed popup info
        """)
    
    with feature_col4:
        st.markdown("""
        ### 🏙 City AQ Map
        **City-specific analysis**
        
        - Individual city focus
        - Multiple data points
        - Statistical overview
        - Detailed data view
        """)
    
    st.markdown("---")
    
    # Data Quality Info
    st.markdown("## 📊 Data Quality & Coverage")
    
    quality_col1, quality_col2, quality_col3 = st.columns(3)
    
    with quality_col1:
        # Data completeness
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = df.count().sum()
        completeness = (non_null_cells / total_cells) * 100
        st.metric("📈 Data Completeness", f"{completeness:.1f}%")
    
    with quality_col2:
        # Time range info
        date_cols = ['Date', 'date', 'timestamp', 'time', 'Time']
        date_col = None
        for col in date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                date_range = df[date_col].dropna()
                if len(date_range) > 0:
                    min_date = date_range.min().strftime('%Y-%m-%d')
                    max_date = date_range.max().strftime('%Y-%m-%d')
                    st.metric("📅 Time Range", f"{min_date} to {max_date}")
                else:
                    st.metric("📅 Time Range", "Not available")
            except:
                st.metric("📅 Time Range", "Not available")
        else:
            st.metric("📅 Time Range", "Not available")
    
    with quality_col3:
        # Parameter coverage
        parameters = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'NH3']
        available_params = [p for p in parameters if p in df.columns]
        st.metric("🔧 Parameters", f"{len(available_params)}/7")
    
    st.markdown("---")
    
    # Quick Access Buttons
    st.markdown("## 🚀 Quick Access")
    
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("🔮 Go to Predictor", use_container_width=True):
            st.session_state.page = "🔮 PM2.5 Prediction"
            st.rerun()
    
    with quick_col2:
        if st.button("📊 Explore Data", use_container_width=True):
            st.session_state.page = "📊 EDA & Visualisation"
            st.rerun()
    
    with quick_col3:
        if st.button("🗺 View India Map", use_container_width=True):
            st.session_state.page = "🗺 India AQ Map"
            st.rerun()
    
    with quick_col4:
        if st.button("🏙 City Analysis", use_container_width=True):
            st.session_state.page = "🏙 City AQ Map"
            st.rerun()
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <h3>🌍 Making Air Quality Data Accessible & Actionable</h3>
        <p>This dashboard is part of the Data analytics and Visualisation academic project focused on environmental data analytics.</p>
        <p>Developed using Python & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# PAGE 2 – PREDICTION
# ==================================================
elif page == "🔮 PM2.5 Prediction":
    # Title
    st.title("🔮 PM2.5 Concentration Predictor")
    st.markdown("Predict PM2.5 levels using machine learning based on 6 air quality parameters")
    
    # Main content
    tab1, tab2 = st.tabs(["📊 Predict", "ℹ️ About Model"])
    
    with tab1:
        # Add an information box
        with st.expander("ℹ️ Quick Guide", expanded=True):
            st.markdown("""
            **How to use:**
            1. Enter values for all 6 parameters in the input fields below
            2. Click the **'Predict PM2.5'** button
            3. View predictions, AQI category, and health advisories
            
            **Default values** are set to typical readings for reference.
            """)
        
        # Create input columns
        col1, col2 = st.columns(2)

        with col1:
            so2 = st.number_input("SO₂ (µg/m³)", 
                                 min_value=0.0, 
                                 max_value=500.0, 
                                 value=10.0,
                                 step=0.1,
                                 help="Sulfur Dioxide concentration")
            no2 = st.number_input("NO₂ (µg/m³)", 
                                 min_value=0.0, 
                                 max_value=500.0, 
                                 value=20.0,
                                 step=0.1,
                                 help="Nitrogen Dioxide concentration")
            co = st.number_input("CO (mg/m³)", 
                                min_value=0.0, 
                                max_value=10.0, 
                                value=1.0,
                                step=0.1,
                                help="Carbon Monoxide concentration")

        with col2:
            o3 = st.number_input("O₃ (µg/m³)", 
                                min_value=0.0, 
                                max_value=500.0, 
                                value=30.0,
                                step=0.1,
                                help="Ozone concentration")
            pm10 = st.number_input("PM10 (µg/m³)", 
                                  min_value=0.0, 
                                  max_value=1000.0, 
                                  value=50.0,
                                  step=0.1,
                                  help="PM10 concentration")
            nh3 = st.number_input("NH₃ (µg/m³)", 
                                 min_value=0.0, 
                                 max_value=500.0, 
                                 value=15.0,
                                 step=0.1,
                                 help="Ammonia concentration")
        
        # Add a predict button
        predict_btn = st.button("🔮 Predict PM2.5", type="primary", use_container_width=True)
        
        if predict_btn:
            if model is None:
                st.error("❌ Model not loaded. Please check the model file and try again.")
            else:
                try:
                    # Create input array with correct feature order and shape
                    X_input = np.array([so2, no2, co, o3, pm10, nh3]).reshape(1, -1)
                    
                    # Make prediction
                    pred = model.predict(X_input)[0]
                    
                    # Get AQI category
                    category = aqi_category(pred)
                    
                    # Results section
                    st.success(f"✅ **Prediction Successful!**")
                    
                    # Results in columns
                    results_col1, results_col2, results_col3 = st.columns(3)
                    
                    with results_col1:
                        st.markdown("### 📊 Prediction")
                        st.markdown(f"<h1 style='text-align: center; color: {get_aqi_color(category)};'>{pred:.1f} µg/m³</h1>", 
                                   unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align: center;'>Predicted PM2.5 Concentration</p>", 
                                   unsafe_allow_html=True)
                    
                    with results_col2:
                        st.markdown("### 🏷️ AQI Category")
                        st.markdown(f"<h1 style='text-align: center;'>{category}</h1>", 
                                   unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align: center;'>", unsafe_allow_html=True)
                        color_dot = "🟢" if category == "Good" else \
                                   "🟡" if category == "Satisfactory" else \
                                   "🟠" if category == "Moderate" else \
                                   "🔴" if category == "Poor" else \
                                   "💜" if category == "Very Poor" else "🟣"
                        st.markdown(f"<h2 style='text-align: center;'>{color_dot}</h2>", unsafe_allow_html=True)
                        st.markdown(f"</div>", unsafe_allow_html=True)
                    
                    with results_col3:
                        st.markdown("### 📈 Pollution Level")
                        progress_value = min(pred / 300, 1.0)
                        st.progress(float(progress_value), 
                                   text=f"PM2.5: {pred:.1f} µg/m³")
                        
                        # Simple gauge
                        gauge_color = get_aqi_color(category)
                        st.markdown(f"""
                        <div style="width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; margin: 10px 0;">
                            <div style="width: {progress_value*100}%; height: 100%; background-color: {gauge_color}; 
                                    border-radius: 10px;"></div>
                        </div>
                        <p style="text-align: center; font-size: 12px; color: #666;">
                            Low (0) ────────────────────────────────────────────────────── High (300+)
                        </p>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Health Advisory
                    st.markdown("### 📋 Health Advisory")
                    advisories = {
                        "Good": """
                        ✅ **Good Air Quality**
                        - Air quality is satisfactory
                        - No health impacts expected
                        - Ideal for outdoor activities
                        """,
                        "Satisfactory": """
                        ⚠️ **Satisfactory Air Quality**
                        - Air quality is acceptable
                        - Unusually sensitive people should consider limiting prolonged outdoor exertion
                        - Generally safe for most activities
                        """,
                        "Moderate": """
                        ⚠️ **Moderate Air Quality**
                        - Members of sensitive groups may experience health effects
                        - General public is not likely to be affected
                        - Consider reducing prolonged outdoor exertion if sensitive
                        """,
                        "Poor": """
                        ❗ **Poor Air Quality**
                        - Everyone may begin to experience health effects
                        - Members of sensitive groups may experience more serious health effects
                        - Reduce outdoor activities
                        - Sensitive groups should avoid outdoor exertion
                        """,
                        "Very Poor": """
                        🚨 **Very Poor Air Quality**
                        - Health alert: everyone may experience more serious health effects
                        - Avoid outdoor activities
                        - Sensitive groups should remain indoors
                        - Consider using air purifiers indoors
                        """,
                        "Severe": """
                        🆘 **Severe Air Quality**
                        - Health warning of emergency conditions
                        - The entire population is more likely to be affected
                        - Stay indoors with windows closed
                        - Use air purifiers
                        - Avoid all outdoor activities
                        """
                    }
                    
                    st.info(advisories.get(category, "No advisory available."))
                    
                    # Input Summary
                    with st.expander("📊 Input Summary", expanded=False):
                        input_data = pd.DataFrame({
                            'Parameter': ['SO₂', 'NO₂', 'CO', 'O₃', 'PM10', 'NH₃'],
                            'Value': [so2, no2, co, o3, pm10, nh3],
                            'Unit': ['µg/m³', 'µg/m³', 'mg/m³', 'µg/m³', 'µg/m³', 'µg/m³']
                        })
                        st.table(input_data)
                        
                    # Feature importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        with st.expander("🔍 Feature Importance Analysis", expanded=False):
                            features = ['SO₂', 'NO₂', 'CO', 'O₃', 'PM10', 'NH₃']
                            importance = model.feature_importances_
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.barh(features, importance, color='skyblue')
                            ax.set_xlabel('Importance Score', fontweight='bold')
                            ax.set_title('Feature Importance in PM2.5 Prediction', fontweight='bold')
                            
                            # Add value labels
                            for bar in bars:
                                width = bar.get_width()
                                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                                       f'{width:.3f}', ha='left', va='center')
                            
                            st.pyplot(fig)
                            
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("Please check if the input values are valid and within reasonable ranges.")
    
    with tab2:
        st.markdown("""
        ## ℹ️ About the Prediction Model
        
        ### 🎯 Model Overview
        This application uses a **Random Forest Regressor** to predict PM2.5 concentrations based on 6 input parameters.
        
        ### 🔧 Model Details
        - **Algorithm:** Random Forest Regressor
        - **Target Variable:** PM2.5 (µg/m³)
        - **Input Features:** 6 air quality parameters
        - **Training Data:** Multi-city air quality measurements
        
        ### 📊 Input Parameters
        1. **SO₂ (Sulfur Dioxide)** - From fossil fuel combustion
        2. **NO₂ (Nitrogen Dioxide)** - From vehicle emissions
        3. **CO (Carbon Monoxide)** - From incomplete combustion
        4. **O₃ (Ozone)** - Ground-level ozone (pollutant)
        5. **PM10** - Coarse particulate matter
        6. **NH₃ (Ammonia)** - From agricultural activities
        
        ### 🎯 Prediction Accuracy
        The model has been trained on historical data and provides predictions with reasonable accuracy for:
        - Urban air quality assessment
        - Pollution trend analysis
        - Health impact estimation
        
        ### ⚠️ Limitations
        - Predictions are estimates based on historical patterns
        - Actual measurements may vary
        - Model performance depends on data quality
        - Seasonal variations may affect accuracy
        
        ### 🔄 Model Updates
        The model can be retrained with new data to improve accuracy over time.
        """)

# ==================================================
# PAGE 3 – EDA & VISUALIZATION (Keep existing code)
# ==================================================
elif page == "📊 EDA & Visualisation":
    # Title
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Interactive visualization and analysis of air quality data")
    
    # Filter section
    st.subheader("🔍 Filter Data")
    
    f1, f2, f3 = st.columns(3)

    with f1:
        city_filter = st.selectbox(
            "City",
            ["All"] + sorted(df["City"].unique())
        )

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

    # Clean filtered data - remove NaN values
    filtered_df = filtered_df.dropna(subset=['PM2.5', 'PM10'], how='any')
    
    st.markdown(f"**📊 Showing {len(filtered_df)} records**")
    
    if len(filtered_df) > 0:
        # Data preview
        with st.expander("📋 View Filtered Data", expanded=False):
            st.dataframe(filtered_df.head(20))

        # Statistics
        st.subheader("📈 Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg PM2.5", f"{filtered_df['PM2.5'].mean():.2f}")
        with col2:
            st.metric("Max PM2.5", f"{filtered_df['PM2.5'].max():.2f}")
        with col3:
            st.metric("Min PM2.5", f"{filtered_df['PM2.5'].min():.2f}")
        with col4:
            st.metric("Std Dev", f"{filtered_df['PM2.5'].std():.2f}")

        # Visualization tabs
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Distributions", "📈 Correlations", "📅 Time Series"])
        
        with viz_tab1:
            # AQI Distribution
            st.subheader("AQI Category Distribution")
            aqi_counts = filtered_df["AQI Category"].value_counts()
            if len(aqi_counts) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Bar chart
                colors = [get_aqi_color(cat) for cat in aqi_counts.index]
                bars = ax1.bar(aqi_counts.index, aqi_counts.values, color=colors)
                ax1.set_ylabel('Count')
                ax1.set_title('AQI Category Distribution (Bar Chart)')
                ax1.tick_params(axis='x', rotation=45)
                
                # Add count labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
                
                # Pie chart
                wedges, texts, autotexts = ax2.pie(aqi_counts.values, labels=aqi_counts.index, 
                                                  autopct='%1.1f%%', colors=colors,
                                                  startangle=90)
                ax2.set_title('AQI Category Distribution (Pie Chart)')
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No data available for selected AQI categories.")

            # PM2.5 Distribution
            st.subheader("PM2.5 Distribution Histogram")
            if len(filtered_df) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                n_bins = min(30, len(filtered_df))
                ax.hist(filtered_df["PM2.5"], bins=n_bins, edgecolor='black', alpha=0.7, color='skyblue')
                ax.axvline(filtered_df["PM2.5"].mean(), color='red', linestyle='--', 
                          label=f'Mean: {filtered_df["PM2.5"].mean():.2f}')
                ax.set_xlabel('PM2.5 (µg/m³)')
                ax.set_ylabel('Frequency')
                ax.set_title('PM2.5 Distribution Histogram')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.warning("Insufficient data for histogram.")
        
        with viz_tab2:
            # Scatter Plot
            st.subheader("PM2.5 vs PM10 Scatter Plot")
            if len(filtered_df) > 1:
                valid_data = filtered_df[['PM10', 'PM2.5']].dropna()
                
                if len(valid_data) >= 2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(valid_data["PM10"], valid_data["PM2.5"], 
                                        alpha=0.6, c=valid_data["PM2.5"], cmap='viridis')
                    ax.set_xlabel('PM10 (µg/m³)')
                    ax.set_ylabel('PM2.5 (µg/m³)')
                    ax.set_title('PM2.5 vs PM10 Correlation')
                    
                    try:
                        if len(valid_data) >= 2:
                            if valid_data["PM10"].std() > 0 and valid_data["PM2.5"].std() > 0:
                                z = np.polyfit(valid_data["PM10"], valid_data["PM2.5"], 1)
                                p = np.poly1d(z)
                                ax.plot(valid_data["PM10"], p(valid_data["PM10"]), "r--", alpha=0.8)
                                
                                correlation = valid_data["PM10"].corr(valid_data["PM2.5"])
                                ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                                        transform=ax.transAxes, fontsize=12,
                                        verticalalignment='top', 
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    except:
                        pass
                    
                    plt.colorbar(scatter, label='PM2.5 (µg/m³)')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                else:
                    st.warning("Need at least 2 valid data points for scatter plot.")
            else:
                st.warning("Need at least 2 data points for scatter plot.")
        
        with viz_tab3:
            # Time series
            date_columns = ['Date', 'date', 'timestamp', 'time', 'Time']
            date_col = None
            for col in date_columns:
                if col in filtered_df.columns:
                    date_col = col
                    break
            
            if date_col and len(filtered_df) > 1:
                try:
                    filtered_df_copy = filtered_df.copy()
                    filtered_df_copy[date_col] = pd.to_datetime(filtered_df_copy[date_col], errors='coerce')
                    filtered_df_copy = filtered_df_copy.dropna(subset=[date_col])
                    filtered_df_copy = filtered_df_copy.sort_values(date_col)
                    
                    if len(filtered_df_copy) > 0:
                        st.subheader("PM2.5 Time Series")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(filtered_df_copy[date_col], filtered_df_copy['PM2.5'], 
                               marker='o', markersize=3, linewidth=1, color='royalblue')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('PM2.5 (µg/m³)')
                        ax.set_title('PM2.5 Over Time')
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        
                        # Add average line
                        avg_pm25 = filtered_df_copy['PM2.5'].mean()
                        ax.axhline(y=avg_pm25, color='red', linestyle='--', alpha=0.7, 
                                  label=f'Average: {avg_pm25:.1f} µg/m³')
                        ax.legend()
                        
                        st.pyplot(fig)
                except:
                    st.info("Could not create time series plot. Date format may be incompatible.")
            else:
                st.info("Time series data not available or insufficient data points.")
    
    else:
        st.warning("⚠️ No data matches the selected filters. Please adjust your filter criteria.")

# ==================================================
# PAGE 4 – INDIA MAP (Keep existing code)
# ==================================================
elif page == "🗺 India AQ Map":
    st.title("🗺 India Air Quality Map")
    st.markdown("Interactive map showing average PM2.5 levels across major Indian cities")
    
    # Calculate city averages
    city_pm = df.groupby("City")["PM2.5"].mean().reset_index()
    city_pm = city_pm.sort_values("PM2.5", ascending=False)
    
        # Summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Most Polluted Cities")
        top_cities = city_pm.head(5)
        for idx, (_, row) in enumerate(top_cities.iterrows(), 1):
            category = aqi_category(row["PM2.5"])
            color = get_aqi_color(category)
            st.markdown(f"""
            <div style="padding: 8px; margin: 5px 0; border-left: 4px solid {color}; background-color: #000000;">
                <b>{idx}. {row['City']}</b><br>
                <span style="color: #666;">{row['PM2.5']:.1f} µg/m³ • {category}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🌿 Least Polluted Cities")
        bottom_cities = city_pm.tail(5).iloc[::-1]
        for idx, (_, row) in enumerate(bottom_cities.iterrows(), 1):
            category = aqi_category(row["PM2.5"])
            color = get_aqi_color(category)
            st.markdown(f"""
            <div style="padding: 8px; margin: 5px 0; border-left: 4px solid {color}; background-color: #000000;">
                <b>{idx}. {row['City']}</b><br>
                <span style="color: #666;">{row['PM2.5']:.1f} µg/m³ • {category}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Create the map
    st.markdown("### 📍 Interactive Map")
    india_map = folium.Map(location=[22.5, 80.0], zoom_start=5, tiles='CartoDB positron')
    
    # Add markers for each city
    for _, row in city_pm.iterrows():
        city_name = row["City"]
        if city_name in CITY_COORDS:
            lat, lon = CITY_COORDS[city_name]
            pm_value = row["PM2.5"]
            category = aqi_category(pm_value)
            
            # Color coding based on AQI
            color_map = {
                "Good": "green",
                "Satisfactory": "lightgreen",
                "Moderate": "orange",
                "Poor": "red",
                "Very Poor": "purple",
                "Severe": "darkred"
            }
            color = color_map.get(category, "gray")
            
            # Size based on PM2.5 value
            radius = max(5, min(20, pm_value / 10))
            
            # Create popup content
            popup_html = f"""
            <div style="font-family: Arial; padding: 10px; min-width: 200px;">
                <h4 style="margin-bottom: 5px; color: {color};">{city_name}</h4>
                <hr style="margin: 5px 0;">
                <b>PM2.5:</b> {pm_value:.1f} µg/m³<br>
                <b>AQI Category:</b> {category}<br>
                <b>Status:</b> <span style="color: {color}; font-weight: bold;">● {category}</span>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{city_name}: {pm_value:.1f} µg/m³ ({category})",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=2
            ).add_to(india_map)
    
       # Add legend with color codes
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 220px; height: 260px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);">
        <b>AQI Categories</b><br>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 15px; height: 15px; background-color: #00E400; margin-right: 5px; border-radius: 50%;"></div>
            <span>Good (≤30 µg/m³)</span>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 15px; height: 15px; background-color: #FFFF00; margin-right: 5px; border-radius: 50%;"></div>
            <span>Satisfactory (31-60)</span>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 15px; height: 15px; background-color: #FF7E00; margin-right: 5px; border-radius: 50%;"></div>
            <span>Moderate (61-90)</span>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 15px; height: 15px; background-color: #FF0000; margin-right: 5px; border-radius: 50%;"></div>
            <span>Poor (91-120)</span>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 15px; height: 15px; background-color: #8F3F97; margin-right: 5px; border-radius: 50%;"></div>
            <span>Very Poor (121-250)</span>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 15px; height: 15px; background-color: #7E0023; margin-right: 5px; border-radius: 50%;"></div>
            <span>Severe  (>250)</span>
        </div>
        <hr style="margin: 10px 0;">
        <small>Marker size indicates pollution level</small>
    </div>
    '''
    india_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Display the map
    st_folium(india_map, width=1000, height=600, returned_objects=[])

# ==================================================
# PAGE 5 – CITY MAP (Keep existing code)
# ==================================================
elif page == "🏙 City AQ Map":
    st.title("🏙 City-Specific Air Quality Analysis")
    st.markdown("Detailed analysis and visualization for individual cities")
    
    # City selection
    city = st.selectbox("Select City", sorted(df["City"].unique()))
    
    # Filter data for selected city
    city_df = df[df["City"] == city]
    
    # Clean data
    city_df = city_df.dropna(subset=['PM2.5'])
    
    if len(city_df) > 0:
        # Calculate statistics
        avg_pm = city_df["PM2.5"].mean()
        max_pm = city_df["PM2.5"].max()
        min_pm = city_df["PM2.5"].min()
        std_pm = city_df["PM2.5"].std()
        category = aqi_category(avg_pm)
        
        # City statistics in nice format
        st.markdown(f"### 📊 {city} Air Quality Statistics")
        
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Average PM2.5", f"{avg_pm:.2f} µg/m³", category)
        
        with stats_col2:
            st.metric("Maximum PM2.5", f"{max_pm:.2f} µg/m³")
        
        with stats_col3:
            st.metric("Minimum PM2.5", f"{min_pm:.2f} µg/m³")
        
        with stats_col4:
            st.metric("Std Deviation", f"{std_pm:.2f} µg/m³")
        
        st.markdown("---")
        
        # Map and data side by side
        map_col, data_col = st.columns([2, 1])
        
        with map_col:
            st.markdown("### 📍 City Location")
            
            # Get city coordinates
            lat, lon = CITY_COORDS.get(city, [22.5, 80.0])
            
            # Create city map
            city_map = folium.Map(location=[lat, lon], zoom_start=12, tiles='CartoDB positron')
            
            # Color coding based on AQI
            color_map = {
                "Good": "green",
                "Satisfactory": "lightgreen",
                "Moderate": "orange",
                "Poor": "red",
                "Very Poor": "purple",
                "Severe": "darkred"
            }
            color = color_map.get(category, "gray")
            
            # Add main city marker
            popup_html = f"""
            <div style="font-family: Arial; padding: 10px; min-width: 200px;">
                <h3 style="margin-bottom: 5px; color: {color};">{city}</h3>
                <hr style="margin: 5px 0;">
                <b>Average PM2.5:</b> {avg_pm:.2f} µg/m³<br>
                <b>AQI Category:</b> {category}<br>
                <b>Data Points:</b> {len(city_df)}<br>
                <b>Range:</b> {min_pm:.1f} - {max_pm:.1f} µg/m³<br>
                <b>Std Dev:</b> {std_pm:.1f} µg/m³
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=25,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{city}: Avg PM2.5 = {avg_pm:.2f} µg/m³",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                weight=3
            ).add_to(city_map)
            
            # Display the map
            st_folium(city_map, width=700, height=400, returned_objects=[])
        
        with data_col:
            st.markdown("### 📈 Quick Insights")
            
            # AQI category info
            st.markdown(f"#### AQI: **{category}**")
            st.markdown(f"Color: <span style='color:{color}; font-weight:bold;'>● {color}</span>", 
                       unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Data points info
            st.markdown(f"**Data Points:** {len(city_df)}")
            
            # Calculate AQI distribution for this city
            city_aqi_dist = city_df['AQI Category'].value_counts()
            st.markdown("**AQI Distribution:**")
            for aqi_cat, count in city_aqi_dist.items():
                aqi_color = get_aqi_color(aqi_cat)
                st.markdown(f"- <span style='color:{aqi_color};'>{aqi_cat}:</span> {count} records", 
                           unsafe_allow_html=True)
            
            # Data quality info
            st.markdown("---")
            st.markdown("**Data Quality:**")
            completeness = (city_df.count().sum() / (city_df.shape[0] * city_df.shape[1])) * 100
            st.markdown(f"- Completeness: {completeness:.1f}%")
            
            # Date range if available
            date_cols = ['Date', 'date', 'timestamp', 'time', 'Time']
            date_col = None
            for col in date_cols:
                if col in city_df.columns:
                    date_col = col
                    break
            
            if date_col:
                try:
                    city_df[date_col] = pd.to_datetime(city_df[date_col], errors='coerce')
                    date_range = city_df[date_col].dropna()
                    if len(date_range) > 0:
                        min_date = date_range.min().strftime('%Y-%m-%d')
                        max_date = date_range.max().strftime('%Y-%m-%d')
                        st.markdown(f"- Time Range: {min_date} to {max_date}")
                except:
                    pass
        
        # Detailed data view
        st.markdown("---")
        with st.expander(f"📋 View Detailed {city} Data", expanded=False):
            st.dataframe(city_df.head(100))
            
            # Download option
            csv = city_df.to_csv(index=False)
            st.download_button(
                label="📥 Download City Data (CSV)",
                data=csv,
                file_name=f"{city}_air_quality_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.warning(f"No data available for {city}")

# --------------------------------------------------
# COMMON FOOTER FOR ALL PAGES
# --------------------------------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.markdown("""
    **📘 Course:** Data analytics and Visualisation – Air Quality Analysis & Prediction  
    **🎓 Academic Project** – Advanced Data Analytics  
    **🌍** Making environmental data accessible and actionable
    """)

with footer_col2:
    st.markdown("""
    **🛠 Built with:**  
    • Python 🐍  
    • Streamlit ⚡  
    • Scikit-learn 🤖  
    • Folium 🗺
    """)

with footer_col3:
    st.markdown("""
    **☁ Deployment:**  
    Streamlit Cloud  
    
    
    """)

# Add a simple back to top button
st.markdown("""
<div style="text-align: center; margin-top: 20px;">
    <a href="#top" style="text-decoration: none; color: #666;">
        ⬆ Back to Top
    </a>
</div>
""", unsafe_allow_html=True)
