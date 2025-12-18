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
# SIDEBAR – NAVIGATION
# --------------------------------------------------
st.sidebar.title("📊 Dashboard Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🔮 PM2.5 Prediction",
        "📊 EDA & Visualisation",
        "🗺 India AQ Map",
        "🏙 City AQ Map"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("CMP7005 – Air Quality Analysis")

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
# KPI METRICS (GLOBAL)
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

df["AQI Category"] = df["PM2.5"].apply(aqi_category)

# ==================================================
# PAGE 1 – PREDICTION (CORRECTED)
# ==================================================
if page == "🔮 PM2.5 Prediction":
    st.subheader("Predict PM2.5 Concentration")
    
    # Add an information box
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        This model predicts PM2.5 concentration based on 6 air quality parameters:
        - SO₂ (Sulfur Dioxide)
        - NO₂ (Nitrogen Dioxide) 
        - CO (Carbon Monoxide)
        - O₃ (Ozone)
        - PM10 (Particulate Matter 10)
        - NH₃ (Ammonia)
        
        **Model Type:** Random Forest Regressor
        **Target:** PM2.5 (µg/m³)
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
                
                # Display results
                st.success(f"✅ **Predicted PM2.5: {pred:.2f} µg/m³**")
                
                # Get AQI category
                category = aqi_category(pred)
                
                # Create columns for better layout
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("PM2.5 Prediction", f"{pred:.2f} µg/m³")
                
                with col_b:
                    st.metric("AQI Category", category)
                
                with col_c:
                    # Simple AQI status indicator
                    aqi_colors = {
                        "Good": "🟢",
                        "Satisfactory": "🟡", 
                        "Moderate": "🟠",
                        "Poor": "🔴",
                        "Very Poor": "💜",
                        "Severe": "🟣"
                    }
                    st.markdown(f"**Status:** {aqi_colors.get(category, '⚫')} {category}")
                
                # Progress bar for visual representation
                st.subheader("Pollution Level Indicator")
                
                # Normalize for progress bar (cap at 300 for display)
                progress_value = min(pred / 300, 1.0)  # Cap at 300 µg/m³ for visualization
                st.progress(float(progress_value), 
                           text=f"PM2.5: {pred:.1f} µg/m³ | {category}")
                
                # Add color-coded progress bar
                colors = {
                    "Good": "#00E400",
                    "Satisfactory": "#FFFF00",
                    "Moderate": "#FF7E00",
                    "Poor": "#FF0000",
                    "Very Poor": "#8F3F97",
                    "Severe": "#7E0023"
                }
                
                # Create a custom progress bar with color
                st.markdown(
                    f"""
                    <div style="width: 100%; background-color: #f0f0f0; border-radius: 10px; margin: 10px 0;">
                        <div style="width: {progress_value*100}%; background-color: {colors.get(category, '#808080')}; 
                                border-radius: 10px; text-align: center; padding: 5px; color: white;">
                            {pred:.1f} µg/m³
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Add health advisory based on AQI category
                st.subheader("📋 Health Advisory")
                advisories = {
                    "Good": "✅ **Air quality is satisfactory.** No health impacts expected.",
                    "Satisfactory": "⚠️ **Air quality is acceptable.** Unusually sensitive people should consider limiting prolonged outdoor exertion.",
                    "Moderate": "⚠️ **Members of sensitive groups may experience health effects.** General public is not likely to be affected.",
                    "Poor": "❗ **Everyone may begin to experience health effects;** members of sensitive groups may experience more serious health effects.",
                    "Very Poor": "🚨 **Health alert:** everyone may experience more serious health effects.",
                    "Severe": "🆘 **Health warning of emergency conditions.** The entire population is more likely to be affected."
                }
                
                st.info(advisories.get(category, "No advisory available."))
                
                # Show input summary
                with st.expander("📊 Input Summary"):
                    input_data = pd.DataFrame({
                        'Parameter': ['SO₂', 'NO₂', 'CO', 'O₃', 'PM10', 'NH₃'],
                        'Value': [so2, no2, co, o3, pm10, nh3],
                        'Unit': ['µg/m³', 'µg/m³', 'mg/m³', 'µg/m³', 'µg/m³', 'µg/m³']
                    })
                    st.table(input_data)
                    
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    with st.expander("🔍 Feature Importance"):
                        features = ['SO₂', 'NO₂', 'CO', 'O₃', 'PM10', 'NH₃']
                        importance = model.feature_importances_
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(features, importance)
                        ax.set_xlabel('Importance')
                        ax.set_title('Feature Importance in PM2.5 Prediction')
                        st.pyplot(fig)
                        
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please check if the input values are valid and within reasonable ranges.")
                
                # Debug information
                if st.checkbox("Show debug info"):
                    st.write(f"Model type: {type(model)}")
                    if hasattr(model, 'n_features_in_'):
                        st.write(f"Model expects {model.n_features_in_} features")
                    st.write(f"Input shape: {X_input.shape if 'X_input' in locals() else 'Not created'}")
                    st.write(f"Input values: {[so2, no2, co, o3, pm10, nh3]}")

# ==================================================
# PAGE 2 – FILTERED EDA
# ==================================================
elif page == "📊 EDA & Visualisation":
    st.subheader("Exploratory Data Analysis (Filter-Driven)")

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

    st.markdown(f"**Showing {len(filtered_df)} records**")
    st.dataframe(filtered_df.head(20))

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Filtered Avg PM2.5", f"{filtered_df['PM2.5'].mean():.2f}")
    with col2:
        st.metric("Filtered Max PM2.5", f"{filtered_df['PM2.5'].max():.2f}")
    with col3:
        st.metric("Filtered Min PM2.5", f"{filtered_df['PM2.5'].min():.2f}")
    with col4:
        st.metric("Std Dev", f"{filtered_df['PM2.5'].std():.2f}")

    st.subheader("AQI Category Distribution")
    aqi_counts = filtered_df["AQI Category"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97', '#7E0023']
    ax.bar(aqi_counts.index, aqi_counts.values, color=colors[:len(aqi_counts)])
    ax.set_ylabel('Count')
    ax.set_title('AQI Category Distribution')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("PM2.5 Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(filtered_df["PM2.5"], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(filtered_df["PM2.5"].mean(), color='red', linestyle='--', label=f'Mean: {filtered_df["PM2.5"].mean():.2f}')
    ax.set_xlabel('PM2.5 (µg/m³)')
    ax.set_ylabel('Frequency')
    ax.set_title('PM2.5 Distribution Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("PM2.5 vs PM10 Scatter Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(filtered_df["PM10"], filtered_df["PM2.5"], 
                        alpha=0.6, c=filtered_df["PM2.5"], cmap='viridis')
    ax.set_xlabel('PM10 (µg/m³)')
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.set_title('PM2.5 vs PM10 Correlation')
    
    # Add correlation line if there's data
    if len(filtered_df) > 1:
        z = np.polyfit(filtered_df["PM10"], filtered_df["PM2.5"], 1)
        p = np.poly1d(z)
        ax.plot(filtered_df["PM10"], p(filtered_df["PM10"]), "r--", alpha=0.8)
        
        # Calculate correlation
        correlation = filtered_df["PM10"].corr(filtered_df["PM2.5"])
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.colorbar(scatter, label='PM2.5 (µg/m³)')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Time series if date column exists
    date_columns = ['Date', 'date', 'timestamp']
    date_col = None
    for col in date_columns:
        if col in filtered_df.columns:
            date_col = col
            break
    
    if date_col:
        try:
            filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])
            filtered_df = filtered_df.sort_values(date_col)
            
            st.subheader("PM2.5 Time Series")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(filtered_df[date_col], filtered_df['PM2.5'], marker='o', markersize=3, linewidth=1)
            ax.set_xlabel('Date')
            ax.set_ylabel('PM2.5 (µg/m³)')
            ax.set_title('PM2.5 Over Time')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except:
            pass

# ==================================================
# PAGE 3 – INDIA MAP
# ==================================================
elif page == "🗺 India AQ Map":
    st.subheader("India Air Quality Map (Average PM2.5)")
    
    # Calculate city averages
    city_pm = df.groupby("City")["PM2.5"].mean().reset_index()
    city_pm = city_pm.sort_values("PM2.5", ascending=False)
    
    # Display top and bottom cities
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏆 Most Polluted Cities**")
        top_cities = city_pm.head(5)
        for i, row in top_cities.iterrows():
            category = aqi_category(row["PM2.5"])
            st.markdown(f"{i+1}. {row['City']}: {row['PM2.5']:.1f} µg/m³ ({category})")
    
    with col2:
        st.markdown("**🌿 Least Polluted Cities**")
        bottom_cities = city_pm.tail(5).iloc[::-1]
        for i, row in bottom_cities.iterrows():
            category = aqi_category(row["PM2.5"])
            st.markdown(f"{i+1}. {row['City']}: {row['PM2.5']:.1f} µg/m³ ({category})")
    
    # Create the map
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
            <div style="font-family: Arial; padding: 10px;">
                <h4 style="margin-bottom: 5px;">{city_name}</h4>
                <hr style="margin: 5px 0;">
                <b>PM2.5:</b> {pm_value:.1f} µg/m³<br>
                <b>AQI:</b> {category}<br>
                <b>Status:</b> <span style="color: {color};">● {category}</span>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{city_name}: {pm_value:.1f} µg/m³",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=2
            ).add_to(india_map)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);">
        <b>AQI Categories</b><br>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 15px; height: 15px; background-color: green; margin-right: 5px; border-radius: 50%;"></div>
            <span>Good (≤30)</span>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 15px; height: 15px; background-color: lightgreen; margin-right: 5px; border-radius: 50%;"></div>
            <span>Satisfactory (31-60)</span>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 15px; height: 15px; background-color: orange; margin-right: 5px; border-radius: 50%;"></div>
            <span>Moderate (61-90)</span>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 15px; height: 15px; background-color: red; margin-right: 5px; border-radius: 50%;"></div>
            <span>Poor (91-120)</span>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 15px; height: 15px; background-color: purple; margin-right: 5px; border-radius: 50%;"></div>
            <span>Very Poor (121-250)</span>
        </div>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 15px; height: 15px; background-color: darkred; margin-right: 5px; border-radius: 50%;"></div>
            <span>Severe (>250)</span>
        </div>
    </div>
    '''
    
    india_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Display the map
    st_folium(india_map, width=1000, height=600, returned_objects=[])

# ==================================================
# PAGE 4 – CITY MAP
# ==================================================
elif page == "🏙 City AQ Map":
    st.subheader("City-Specific Air Quality Map")
    
    # City selection
    city = st.selectbox("Select City", sorted(df["City"].unique()))
    
    # Filter data for selected city
    city_df = df[df["City"] == city]
    
    # Calculate statistics
    avg_pm = city_df["PM2.5"].mean()
    max_pm = city_df["PM2.5"].max()
    min_pm = city_df["PM2.5"].min()
    category = aqi_category(avg_pm)
    
    # Display city statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average PM2.5", f"{avg_pm:.2f} µg/m³")
    with col2:
        st.metric("Maximum PM2.5", f"{max_pm:.2f} µg/m³")
    with col3:
        st.metric("Minimum PM2.5", f"{min_pm:.2f} µg/m³")
    with col4:
        st.metric("AQI Category", category)
    
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
        <h3 style="margin-bottom: 5px;">{city}</h3>
        <hr style="margin: 5px 0;">
        <b>Average PM2.5:</b> {avg_pm:.2f} µg/m³<br>
        <b>AQI Category:</b> {category}<br>
        <b>Data Points:</b> {len(city_df)}<br>
        <b>Range:</b> {min_pm:.1f} - {max_pm:.1f} µg/m³
    </div>
    """
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=20,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{city}: Average PM2.5 = {avg_pm:.2f} µg/m³",
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        weight=3
    ).add_to(city_map)
    
    # If there are multiple data points with coordinates, add them
    coord_cols = ['lat', 'latitude', 'Latitude', 'LAT']
    lat_col = None
    lon_col = None
    
    for col in coord_cols:
        if col in city_df.columns:
            lat_col = col
            break
    
    lon_options = ['lon', 'longitude', 'Longitude', 'LON', 'lng']
    for col in lon_options:
        if col in city_df.columns:
            lon_col = col
            break
    
    if lat_col and lon_col and not city_df[lat_col].isnull().all():
        # Add individual data points
        for _, row in city_df.iterrows():
            if pd.notnull(row[lat_col]) and pd.notnull(row[lon_col]):
                point_color = "blue"
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=5,
                    tooltip=f"PM2.5: {row['PM2.5']:.1f} µg/m³",
                    color=point_color,
                    fill=True,
                    fill_color=point_color,
                    fill_opacity=0.6,
                    weight=1
                ).add_to(city_map)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 220px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);">
        <b>{city} Air Quality</b><br>
        <div style="margin-top: 10px;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: {color}; margin-right: 10px; border-radius: 50%; border: 2px solid black;"></div>
                <div>
                    <b>Average PM2.5:</b> {avg_pm:.2f} µg/m³<br>
                    <b>Category:</b> {category}
                </div>
            </div>
        </div>
        <hr style="margin: 10px 0;">
        <div style="display: flex; align-items: center;">
            <div style="width: 10px; height: 10px; background-color: blue; margin-right: 10px; border-radius: 50%;"></div>
            <span>Individual measurements</span>
        </div>
    </div>
    '''
    
    city_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Display the map
    st_folium(city_map, width=900, height=600, returned_objects=[])
    
    # Show city data table
    with st.expander(f"📋 View {city} Data"):
        st.dataframe(city_df.head(50))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("📘 Course: CMP7005 – Air Quality Analysis")
st.markdown("👨‍💻 Developed for Academic Project")
st.markdown("☁ Deployed on Streamlit Cloud")
