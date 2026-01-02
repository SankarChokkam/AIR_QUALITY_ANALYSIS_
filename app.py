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
# DYNAMIC CITY COORDINATES SYSTEM
# --------------------------------------------------
# Default coordinates for common Indian cities
INDIAN_CITY_COORDS = {
    # North
    "Delhi": [28.6139, 77.2090],
    "Lucknow": [26.8467, 80.9462],
    "Chandigarh": [30.7333, 76.7794],
    "Dehradun": [30.3165, 78.0322],
    "Amritsar": [31.6340, 74.8723],
    # South
    "Bengaluru": [12.9716, 77.5946],
    "Chennai": [13.0827, 80.2707],
    "Hyderabad": [17.3850, 78.4867],
    "Kochi": [9.9312, 76.2673],
    "Thiruvananthapuram": [8.5241, 76.9366],
    "Coimbatore": [11.0168, 76.9558],
    "Visakhapatnam": [17.6868, 83.2185],
    # West
    "Mumbai": [19.0760, 72.8777],
    "Pune": [18.5204, 73.8567],
    "Ahmedabad": [23.0225, 72.5714],
    "Surat": [21.1702, 72.8311],
    "Nagpur": [21.1458, 79.0882],
    "Goa": [15.2993, 74.1240],
    # East
    "Kolkata": [22.5726, 88.3639],
    "Patna": [25.5941, 85.1376],
    "Bhubaneswar": [20.2961, 85.8245],
    "Guwahati": [26.1445, 91.7362],
    "Ranchi": [23.3441, 85.3096],
    # Central
    "Bhopal": [23.2599, 77.4126],
    "Indore": [22.7196, 75.8577],
    "Gwalior": [26.2183, 78.1828],
    "Jabalpur": [23.1815, 79.9864],
    # Northwest
    "Jaipur": [26.9124, 75.7873],
    "Jodhpur": [26.2389, 73.0243],
    "Udaipur": [24.5854, 73.7125],
    "Ajmer": [26.4499, 74.6399],
    # Northeast
    "Shillong": [25.5788, 91.8933],
    "Agartala": [23.8315, 91.2868],
    "Aizawl": [23.7271, 92.7176],
    "Imphal": [24.8170, 93.9368],
    # Union Territories
    "Puducherry": [11.9416, 79.8083],
    "Daman": [20.3974, 72.8328],
}

def get_city_coordinates(city_name):
    """Get coordinates for a city from the database or use fallback"""
    # Clean city name (remove extra spaces, etc.)
    city_name = str(city_name).strip()
    
    # Check if city exists in our coordinates database
    if city_name in INDIAN_CITY_COORDS:
        return INDIAN_CITY_COORDS[city_name]
    
    # For cities in dataset but not in coordinates database
    # Let's add them based on known locations from the dataset
    
    # Check the dataset for similar city names or patterns
    # This is a fallback for cities not in our database
    # We can approximate based on Indian regions
    
    # Common dataset cities that might not be in INDIAN_CITY_COORDS
    dataset_city_fallbacks = {
        # Add any cities from your dataset that aren't in INDIAN_CITY_COORDS
        # Example: "CityName": [lat, lon]
    }
    
    if city_name in dataset_city_fallbacks:
        return dataset_city_fallbacks[city_name]
    
    # Ultimate fallback: Use India's centroid
    # But first, let's try to approximate based on city name patterns
    if "nag" in city_name.lower() or "nagar" in city_name.lower():
        return [21.15, 79.09]  # Near Nagpur
    elif "pur" in city_name.lower():
        return [23.0, 77.0]  # Central India
    elif "bad" in city_name.lower():
        return [23.0, 72.5]  # Near Ahmedabad
    else:
        # Use India's centroid for unknown cities
        return [22.5, 80.0]

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

# Get all unique cities from dataset
@st.cache_data
def get_all_cities():
    cities = df["City"].unique().tolist()
    return sorted([city for city in cities if pd.notna(city)])

all_cities = get_all_cities()

# Display city count in sidebar
st.sidebar.markdown(f"**🏙 Cities in Dataset:** {len(all_cities)}")

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
        st.metric("Cities", len(all_cities))
        st.metric("Records", f"{len(df):,}")
    
    st.markdown("---")
    
    # Display all cities in dataset
    st.markdown(f"### 🏙 All Cities in Dataset ({len(all_cities)})")
    
    # Create columns for displaying cities
    num_columns = 4
    cols = st.columns(num_columns)
    
    for idx, city in enumerate(all_cities):
        with cols[idx % num_columns]:
            # Get city statistics
            city_data = df[df["City"] == city]
            avg_pm = city_data["PM2.5"].mean() if len(city_data) > 0 else 0
            category = aqi_category(avg_pm)
            color = get_aqi_color(category)
            
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; border-radius: 5px; 
                        border-left: 4px solid {color}; background-color: #f8f9fa;">
                <b>{city}</b><br>
                <small>Avg PM2.5: {avg_pm:.1f} µg/m³</small><br>
                <small style="color: {color};"><b>{category}</b></small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Rest of the Home Dashboard code remains the same...
    # [Keep all existing Home Dashboard code here, just replace any hardcoded city references]

# ==================================================
# PAGE 4 – INDIA MAP (UPDATED)
# ==================================================
elif page == "🗺 India AQ Map":
    st.title("🗺 India Air Quality Map")
    st.markdown(f"Interactive map showing average PM2.5 levels across {len(all_cities)} Indian cities")
    
    # Calculate city averages
    city_pm = df.groupby("City")["PM2.5"].mean().reset_index()
    city_pm = city_pm.sort_values("PM2.5", ascending=False)
    
    # Display city count
    st.markdown(f"**📊 Showing data for {len(city_pm)} cities**")
    
    # Summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Most Polluted Cities")
        top_cities = city_pm.head(10)  # Show top 10 instead of 5
        for idx, (_, row) in enumerate(top_cities.iterrows(), 1):
            category = aqi_category(row["PM2.5"])
            color = get_aqi_color(category)
            st.markdown(f"""
            <div style="padding: 8px; margin: 5px 0; border-left: 4px solid {color}; background-color: #f8f9fa;">
                <b>{idx}. {row['City']}</b><br>
                <span style="color: #666;">{row['PM2.5']:.1f} µg/m³ • {category}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🌿 Least Polluted Cities")
        bottom_cities = city_pm.tail(10).iloc[::-1]  # Show bottom 10 instead of 5
        for idx, (_, row) in enumerate(bottom_cities.iterrows(), 1):
            category = aqi_category(row["PM2.5"])
            color = get_aqi_color(category)
            st.markdown(f"""
            <div style="padding: 8px; margin: 5px 0; border-left: 4px solid {color}; background-color: #f8f9fa;">
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
        lat, lon = get_city_coordinates(city_name)
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
            <b>Data Points:</b> {len(df[df['City'] == city_name])}<br>
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
    
    # Add legend
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
    
    # Display all cities in a table
    with st.expander("📋 View All City Data", expanded=False):
        # Create a nice table with all city data
        city_summary = df.groupby("City").agg({
            'PM2.5': ['mean', 'max', 'min', 'count'],
            'PM10': 'mean'
        }).round(2)
        
        # Flatten the multi-level columns
        city_summary.columns = ['PM2.5_Mean', 'PM2.5_Max', 'PM2.5_Min', 'Records', 'PM10_Mean']
        city_summary = city_summary.reset_index()
        
        # Add AQI Category
        city_summary['AQI_Category'] = city_summary['PM2.5_Mean'].apply(aqi_category)
        
        # Sort by PM2.5 mean
        city_summary = city_summary.sort_values('PM2.5_Mean', ascending=False)
        
        st.dataframe(city_summary, use_container_width=True)

# ==================================================
# PAGE 5 – CITY MAP (UPDATED)
# ==================================================
elif page == "🏙 City AQ Map":
    st.title("🏙 City-Specific Air Quality Analysis")
    st.markdown(f"Detailed analysis and visualization for individual cities ({len(all_cities)} cities available)")
    
    # City selection with all cities from dataset
    city = st.selectbox("Select City", all_cities)
    
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
        
        # Get city coordinates
        lat, lon = get_city_coordinates(city)
        
        # Display coordinates
        st.caption(f"📍 Coordinates: {lat:.4f}, {lon:.4f}")
        
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
                <b>Coordinates:</b> {lat:.4f}, {lon:.4f}<br>
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
            st.markdown(f"Color: <span style='color:{color}; font-weight:bold;'>● {category}</span>", 
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

# ==================================================
# OTHER PAGES (EDAVISUALIZATION & PREDICTION)
# ==================================================
# These pages should use `all_cities` instead of hardcoded city lists

# In the EDA page, update the city filter:
elif page == "📊 EDA & Visualisation":
    # Title
    st.title("📊 Exploratory Data Analysis")
    st.markdown(f"Interactive visualization and analysis of air quality data ({len(all_cities)} cities)")
    
    # Filter section
    st.subheader("🔍 Filter Data")
    
    f1, f2, f3 = st.columns(3)

    with f1:
        city_filter = st.selectbox(
            "City",
            ["All"] + all_cities  # Use all_cities instead of hardcoded list
        )
    
    # Rest of EDA code remains the same...

# In the Prediction page, no changes needed as it doesn't use city lists

# --------------------------------------------------
# COMMON FOOTER FOR ALL PAGES
# --------------------------------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.markdown(f"""
    **📘 Course:** Data analytics and Visualisation – Air Quality Analysis & Prediction  
    **🎓 Academic Project** – Advanced Data Analytics  
    **🏙 Cities Analyzed:** {len(all_cities)}  
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
