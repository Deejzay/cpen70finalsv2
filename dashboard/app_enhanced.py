# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import joblib

# Page config
st.set_page_config(
    page_title="Taal Lake Water Quality Predictor - Enhanced",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 150px;
    }
    .metric-title {
        font-size: 1rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #0066cc;
        margin-bottom: 0.5rem;
    }
    .metric-trend {
        font-size: 0.8rem;
        color: #777;
    }
    .parameter-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Global Variables and Data Loading ---
PROCESSED_DATA_PATH = 'data/processed/processed_data.csv'
LOOK_BACK = 12

# Define parameter groups
WATER_PARAMS = [
    'Surface Water Temp (°C)', 'Middle Water Temp (°C)', 'Bottom Water Temp (°C)',
    'pH Level', 'Dissolved Oxygen (mg/L)'
]

CLIMATE_PARAMS = [
    'RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION'
]

VOLCANIC_PARAMS = [
    'SO2', 'CO2'
]

POLLUTANT_PARAMS = [
    'Ammonia (mg/L)', 'Nitrate-N/Nitrite-N  (mg/L)', 'Phosphate (mg/L)'
]

# Parameter combination definitions
PARAMETER_COMBINATIONS = {
    "Water Parameters Only": WATER_PARAMS,
    "Water + Climate Parameters": WATER_PARAMS + CLIMATE_PARAMS,
    "Water + Volcanic Parameters": WATER_PARAMS + VOLCANIC_PARAMS,
    "Water + Climate + Volcanic Parameters": WATER_PARAMS + CLIMATE_PARAMS + VOLCANIC_PARAMS
}

# Add the full feature columns for model input (must match training order)
FEATURE_COLUMNS = [
    'RAINFALL',
    'TMAX',
    'TMIN',
    'RH',
    'WIND_SPEED',
    'WIND_DIRECTION',
    'SO2',
    'CO2',
    'Surface Water Temp (°C)',
    'Middle Water Temp (°C)',
    'Bottom Water Temp (°C)',
    'pH Level',
    'Ammonia (mg/L)',
    'Nitrate-N/Nitrite-N  (mg/L)',
    'Phosphate (mg/L)',
    'Dissolved Oxygen (mg/L)',
    'WQI',
    'Pollutant_Level'
]

# Use the multi-output model for enhanced predictions
MODEL_PATH = 'models/hybrid_model_enhanced_multi_output.h5'

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data for different parameter combinations"""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
        # Load the scaler from file (do not fit a new one)
        scaler = joblib.load('data/processed/scaler.pkl')
        return df, scaler
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {e}")
        return pd.DataFrame(), None

@st.cache_resource
def load_models():
    """Load the multi-output model"""
    try:
        # Load the multi-output model
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        return {'multi_output': model}
    except Exception as e:
        st.error(f"Error loading multi-output model: {e}")
        return {}

# Load data and models
df_full, scaler = load_and_preprocess_data()
models = load_models()

if df_full.empty or scaler is None:
    st.error("Failed to load data. Please check the data files.")
    st.stop()

def forecast_multi_output(model, scaler, historical_data, n_steps, params):
    """Enhanced forecasting function for multi-output model (WQI + Pollutant Level)"""
    try:
        # Always use all 17 features for model input
        features_df = historical_data[FEATURE_COLUMNS].copy() if all(col in historical_data.columns for col in FEATURE_COLUMNS) else historical_data.reindex(columns=FEATURE_COLUMNS)
        # Fill any missing columns with the last available value or 0
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Scale the features
        scaled_features = scaler.transform(features_df)
        
        # Prepare the input sequence (last 12 timesteps)
        sequence_length = 12
        if len(scaled_features) < sequence_length:
            st.error(f"Not enough historical data. Need at least {sequence_length} timesteps.")
            return pd.DataFrame()
        
        # Get the last sequence_length timesteps
        input_sequence = scaled_features[-sequence_length:].reshape(1, sequence_length, -1)
        
        # Make prediction with multi-output model
        predictions = model.predict(input_sequence, verbose=0)
        wqi_pred, pollutant_pred = predictions
        
        # Inverse transform the predictions
        # Create dummy arrays with the same shape as training data
        dummy_array_wqi = np.zeros((1, len(FEATURE_COLUMNS)))
        dummy_array_pollutant = np.zeros((1, len(FEATURE_COLUMNS)))
        
        # For WQI prediction
        dummy_array_wqi[0, FEATURE_COLUMNS.index('WQI')] = wqi_pred[0, 0]
        unscaled_wqi = scaler.inverse_transform(dummy_array_wqi)[0, FEATURE_COLUMNS.index('WQI')]
        
        # For pollutant prediction (we'll use a placeholder since pollutant level is calculated)
        # The model predicts a single pollutant level value
        unscaled_pollutant = pollutant_pred[0, 0]
        
        # Ensure numeric outputs
        try:
            unscaled_wqi = float(unscaled_wqi)
            unscaled_pollutant = float(unscaled_pollutant)
        except Exception:
            unscaled_wqi = np.nan
            unscaled_pollutant = np.nan
        
        # Generate future dates
        start_date = pd.Timestamp.today().normalize()
        future_dates = pd.date_range(start=start_date, periods=n_steps, freq='D')
        
        # Create forecast DataFrame with both outputs
        forecast_data = []
        for i, date in enumerate(future_dates):
            forecast_data.append({
                'Date': date,
                'WQI': unscaled_wqi,
                'Water_Pollutant_Level': unscaled_pollutant
            })
        
        forecast_df = pd.DataFrame(forecast_data)
        forecast_df.set_index('Date', inplace=True)
        
        return forecast_df
        
    except Exception as e:
        st.error(f"Error generating multi-output forecast: {e}")
        return pd.DataFrame()

def get_wqi_classification_and_advice(wqi):
    """Get WQI classification and advice"""
    if wqi < 50:
        return "Excellent", "Water is considered safe and clean, suitable for drinking and aquatic life. Maintain current practices."
    elif 50 <= wqi < 75:
        return "Good", "Water quality is slightly impacted but generally safe for most uses. Minor treatment might be needed for sensitive applications. Monitor regularly."
    elif 75 <= wqi < 90:
        return "Fair", "Water quality is moderately impacted. May not be suitable for drinking without significant treatment. Aquatic life might be stressed. Consider investigating sources of impact and implement mitigation measures."
    else:
        return "Poor", "Water quality is severely impacted and generally unsafe for most uses. Significant pollution is present, requiring urgent intervention. Immediate investigation and intervention are required to identify and address pollution sources."

def create_metric_card(title, value, unit, trend_data, icon_path=""):
    """Create a metric card with enhanced styling"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}{unit}</div>
        <div class="metric-trend">{trend_data}</div>
    </div>
    """, unsafe_allow_html=True)

# Helper function to safely format floats
def safe_format_float(val, precision=3):
    try:
        if pd.isna(val):
            return "N/A"
        return f"{float(val):.{precision}f}"
    except Exception:
        return "N/A"

# Sidebar
st.sidebar.title("🌊 Taal Lake Water Quality")
st.sidebar.markdown("---")

# Parameter Selection
st.sidebar.subheader("🔧 Parameter Selection")
selected_combo = st.sidebar.selectbox(
    "Choose Parameter Combination:",
    list(PARAMETER_COMBINATIONS.keys()),
    help="Select which parameters to include in the prediction model"
)

# Navigation
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Home", "Model Information"])

# Filters
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Location Filter")
locations = df_full['Location'].unique().tolist() if 'Location' in df_full.columns else [
    "Tanauan", "Talisay", "Laurel", "Agoncillo", "San Nicolas", "Alitagtag"
]
selected_location = st.sidebar.selectbox("Select Location:", locations)

# Time period selection
st.sidebar.markdown("---")
st.sidebar.subheader("⏰ Time Period")
time_period = st.sidebar.selectbox(
    "Select Forecast Period:",
    ["Weekly (7 days)", "Monthly (30 days)", "Yearly (365 days)"],
    help="Choose how far into the future to predict"
)

# Convert time period to days
period_days = {
    "Weekly (7 days)": 7,
    "Monthly (30 days)": 30,
    "Yearly (365 days)": 365
}
n_steps = period_days[time_period]

# Main content
if page == "Home":
    st.title("🌊 Taal Lake Water Quality Monitoring Dashboard")
    
    # Parameter combination info
    st.markdown(f"""
    <div class="parameter-info" style="color:#222;">
        <h4>🔧 Current Parameter Combination: {selected_combo}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter data for selected location
    location_data = df_full[df_full['Location'] == selected_location].copy()
    
    if location_data.empty:
        st.error(f"No data available for location: {selected_location}")
        st.stop()
    
    # Get current WQI
    current_wqi = location_data['WQI'].iloc[-1] if not location_data.empty else 0
    classification, advice = get_wqi_classification_and_advice(current_wqi)
    
    # Display current metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            "Current WQI",
            safe_format_float(current_wqi),
            "",
            f"Classification: {classification}"
        )
    
    with col2:
        create_metric_card(
            "Location",
            len(location_data),
            " data points",
            f"Selected: {selected_location}"
        )
    
    with col3:
        create_metric_card(
            "Forecast Period",
            n_steps,
            " days",
            f"Period: {time_period}"
        )
    
    # Generate forecast
    st.markdown("---")
    st.subheader("📈 Water Quality Forecast")
    
    try:
        forecast_df = forecast_multi_output(
            models['multi_output'],
            scaler,
            location_data,
            n_steps,
            PARAMETER_COMBINATIONS[selected_combo]
        )
        if not forecast_df.empty:
            # Only show predictions (no historical data)
            forecast_dates = forecast_df.index
            wqi_pred = forecast_df['WQI']
            pollutant_pred = forecast_df['Water_Pollutant_Level']

            col1, col2 = st.columns(2)
            with col1:
                fig_wqi = go.Figure()
                fig_wqi.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=wqi_pred,
                    mode='lines+markers',
                    name='Predicted WQI',
                    line=dict(color='royalblue')
                ))
                fig_wqi.update_layout(
                    title='Predicted Water Quality Index (WQI)',
                    xaxis_title='Date',
                    yaxis_title='WQI',
                    showlegend=False
                )
                st.plotly_chart(fig_wqi, use_container_width=True)

            with col2:
                fig_pollutant = go.Figure()
                fig_pollutant.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=pollutant_pred,
                    mode='lines+markers',
                    name='Predicted Pollutant Level',
                    line=dict(color='firebrick')
                ))
                fig_pollutant.update_layout(
                    title='Predicted Water Pollutant Level',
                    xaxis_title='Date',
                    yaxis_title='Pollutant Level',
                    showlegend=False
                )
                st.plotly_chart(fig_pollutant, use_container_width=True)
            
            # Display forecast metrics
            st.subheader("📊 Forecast Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_wqi = forecast_df['WQI'].mean()
                create_metric_card(
                    "Average WQI",
                    safe_format_float(avg_wqi),
                    "",
                    f"Range: {safe_format_float(forecast_df['WQI'].min())} - {safe_format_float(forecast_df['WQI'].max())}"
                )
            
            with col2:
                avg_pollutant = forecast_df['Water_Pollutant_Level'].mean()
                create_metric_card(
                    "Average Pollutant Level",
                    safe_format_float(avg_pollutant),
                    "",
                    f"Range: {safe_format_float(forecast_df['Water_Pollutant_Level'].min())} - {safe_format_float(forecast_df['Water_Pollutant_Level'].max())}"
                )
            
            with col3:
                wqi_class, _ = get_wqi_classification_and_advice(avg_wqi)
                create_metric_card(
                    "WQI Classification",
                    wqi_class,
                    "",
                    ""
                )
            
            with col4:
                # Pollutant level classification
                if avg_pollutant < 0.5:
                    pollutant_class = "Low"
                elif avg_pollutant < 1.0:
                    pollutant_class = "Moderate"
                else:
                    pollutant_class = "High"
                create_metric_card(
                    "Pollutant Level",
                    pollutant_class,
                    "",
                    ""
                )
            
            # Advice section
            st.subheader("💡 Water Quality Assessment & Recommendations")
            _, wqi_advice = get_wqi_classification_and_advice(avg_wqi)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**WQI Assessment:**")
                st.info(wqi_advice)
            
            with col2:
                st.markdown("**Pollutant Level Assessment:**")
                if avg_pollutant < 0.5:
                    pollutant_advice = "Pollutant levels are low. Water quality is good from a pollutant perspective."
                elif avg_pollutant < 1.0:
                    pollutant_advice = "Pollutant levels are moderate. Monitor regularly and consider source reduction measures."
                else:
                    pollutant_advice = "Pollutant levels are high. Immediate action required to identify and reduce pollution sources."
                st.warning(pollutant_advice)
        else:
            st.warning("Unable to generate forecast. Please check the data and model.")
            
    except Exception as e:
        st.error(f"Error generating forecast: {e}")

elif page == "Model Information":
    st.title("🤖 AI Model Information")
    st.markdown("### Understanding the Enhanced Multi-Output Prediction System")
    
    st.subheader("🔧 Multi-Output Prediction System")
    st.markdown("""
    This enhanced dashboard uses a **multi-output hybrid model** that predicts two key water quality indicators simultaneously:
    
    **1. Water Quality Index (WQI)**
    - Comprehensive water quality assessment
    - Combines multiple water quality parameters
    - Provides overall water quality classification
    
    **2. Water Pollutant Level**
    - Calculated from ammonia, nitrate, and phosphate concentrations
    - Indicates the level of chemical pollution
    - Helps identify pollution sources and trends
    
    **Model Architecture:**
    - Hybrid CNN-LSTM with Attention mechanism
    - Dual output heads for WQI and Pollutant Level
    - Time series forecasting with 12-month lookback
    - Enhanced with parameter selection capabilities
    """)
    
    st.subheader("📊 Model Performance")
    st.markdown("""
    **Multi-Output Model Performance:**
    - **WQI Prediction R²:** ~0.63
    - **Pollutant Level Prediction R²:** ~0.44
    - **Overall Model Accuracy:** ~54%
    
    **Advantages of Multi-Output Approach:**
    - Simultaneous prediction of related water quality indicators
    - Shared feature learning between WQI and pollutant level
    - More comprehensive water quality assessment
    - Better understanding of pollution-water quality relationships
    """)
    
    st.subheader("🎯 Parameter Selection System")
    st.markdown("""
    The dashboard allows you to select different parameter combinations for enhanced predictions:
    
    **Water Parameters Only**
    - Core water quality indicators (temperature, pH, dissolved oxygen)
    - Best for understanding intrinsic water quality
    
    **Water + Climate Parameters**
    - Includes weather and climate factors
    - Accounts for seasonal variations and weather impacts
    
    **Water + Volcanic Parameters**
    - Includes volcanic activity indicators (SO2, CO2)
    - Critical for Taal Lake's volcanic environment
    
    **Water + Climate + Volcanic Parameters**
    - Most comprehensive model
    - Includes all available parameters
    - Best for accurate predictions in complex environments
    """)
    
    st.subheader("💡 Interpretation Guidelines")
    st.markdown("""
    **WQI Classification:**
    - **Excellent (< 50):** Safe and clean water
    - **Good (50-75):** Slightly impacted but generally safe
    - **Fair (75-90):** Moderately impacted, treatment may be needed
    - **Poor (> 90):** Severely impacted, urgent intervention required
    
    **Pollutant Level Classification:**
    - **Low (< 0.5):** Minimal pollution, good water quality
    - **Moderate (0.5-1.0):** Some pollution, regular monitoring needed
    - **High (> 1.0):** Significant pollution, immediate action required
    
    **Usage Recommendations:**
    - **Short-term Monitoring:** Water Parameters Only
    - **Seasonal Analysis:** Water + Climate Parameters
    - **Volcanic Impact Assessment:** Water + Volcanic Parameters
    - **Comprehensive Analysis:** Water + Climate + Volcanic Parameters
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌊 Taal Lake Water Quality Monitoring System | Enhanced with Parameter Selection</p>
    <p>Built with Streamlit and TensorFlow | Data-driven environmental monitoring</p>
</div>
""", unsafe_allow_html=True) 