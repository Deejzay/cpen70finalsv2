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

# Page config (moved up)
st.set_page_config(
    page_title="Taal Lake Water Quality Predictor",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS for the metric cards and overall aesthetic
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
        background-color: #f0f2f6; /* Light gray background for cards */
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05); /* Subtle shadow */
        display: flex;
        flex-direction: column;
        justify-content: space-between;\n        min-height: 150px; /* Ensure consistent card height */
    }
    .metric-title {
        font-size: 1rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #0066cc; /* Primary color for values */
        margin-bottom: 0.5rem;
    }
    .metric-trend {
        font-size: 0.8rem;
        color: #777;
    }
    </style>
    """, unsafe_allow_html=True)

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# --- Global Variables and Data Loading ---
PROCESSED_DATA_PATH = 'data/processed/processed_data.csv'
MODEL_PATH = 'models/hybrid_model_no_early_stopping.h5'
LOOK_BACK = 12 # This should match the look_back used during model training

# Define the 17 features in the exact order they appear in your processed data after numeric selection
# This order is CRUCIAL and derived from prepare_model_data.py's logic after dropping 'Location' and selecting dtypes.
# It's assumed to be alphabetical after initial processing, but please verify if your actual data order differs.
FEATURE_COLUMNS = [
    'RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION', 'SO2', 'CO2',
    'Surface Water Temp (°C)', 'Middle Water Temp (°C)', 'Bottom Water Temp (°C)',
    'pH Level', 'Ammonia (mg/L)', 'Nitrate-N/Nitrite-N  (mg/L)', 'Phosphate (mg/L)',
    'Dissolved Oxygen (mg/L)', 'WQI'
]
TARGET_COLUMN = 'WQI'

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure the model file exists at {MODEL_PATH}.")
        return None

@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
        # Ensure 'Location' column exists before dropping
        if 'Location' not in df.columns:
            st.warning(f"'Location' column not found in {PROCESSED_DATA_PATH}. Some dashboard features may be affected.")

        # Drop WQI and Location for feature scaling
        features_df = df[FEATURE_COLUMNS].copy()

        # Initialize and fit scaler on ALL historic feature data
        # In a real scenario, this scaler should be saved from training and loaded here.
        scaler = MinMaxScaler()
        scaler.fit(features_df) # Fit on the entire feature set for consistent scaling
        
        return df, scaler
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {e}. Please check {PROCESSED_DATA_PATH}.")
        return pd.DataFrame(), None

model = load_model()
df_full, scaler = load_and_preprocess_data()

if model is None or df_full.empty or scaler is None:
    st.stop() # Stop if model or data loading failed


# --- Forecasting Function ---
def forecast_wqi(model, scaler, historical_data, n_steps, feature_columns, target_column):
    # historical_data is a DataFrame for a specific location, with datetime index
    # It should contain ALL feature_columns and the target_column
    
    # Ensure data is sorted by index (time)
    historical_data = historical_data.sort_index()
    
    # Extract features and target from the full historical data
    features_only_df = historical_data[feature_columns]
    wqi_history = historical_data[target_column]
    
    # Get the last LOOK_BACK features for initial prediction
    last_sequence_features = features_only_df.tail(LOOK_BACK)
    if len(last_sequence_features) < LOOK_BACK:
        st.warning(f"Not enough historical data ({len(last_sequence_features)} points) for {LOOK_BACK}-step look-back. Cannot forecast.")
        return pd.Series() # Return empty Series

    # Scale the last sequence of features
    last_sequence_scaled = scaler.transform(last_sequence_features)
    
    # Get the current date from the system
    current_date = datetime.now()
    forecast_dates = pd.date_range(start=current_date + timedelta(days=1), periods=n_steps, freq='D')

    forecasted_wqi = []
    current_sequence = last_sequence_scaled.copy()
    
    for _ in range(n_steps):
        # Reshape for model input (1, LOOK_BACK, num_features)
        input_for_prediction = current_sequence.reshape(1, LOOK_BACK, len(feature_columns))
        
        # Predict the next WQI value
        predicted_wqi_scaled = model.predict(input_for_prediction, verbose=0)[0][0]
        
        # The model predicts scaled WQI. We need to inverse transform it.
        # This requires creating a dummy array with only the WQI value to inverse transform it.
        
        # To inverse transform only the WQI, we need a vector of zeros except at the WQI position.
        # Assuming WQI is the LAST column of the scaled data (after FEATURE_COLUMNS) for inverse scaling.
        all_numeric_cols = FEATURE_COLUMNS # Now FEATURE_COLUMNS contains WQI
        dummy_row_scaled = np.zeros(len(all_numeric_cols))
        wqi_idx_in_all_numeric = all_numeric_cols.index(TARGET_COLUMN)
        dummy_row_scaled[wqi_idx_in_all_numeric] = predicted_wqi_scaled
        
        predicted_wqi = scaler.inverse_transform(dummy_row_scaled.reshape(1, -1))[0, wqi_idx_in_all_numeric]
        forecasted_wqi.append(predicted_wqi)

        # Prepare for next prediction: Shift the window
        # Remove the oldest timestep, add the new (predicted) timestep
        # For autoregressive forecasting, the new input sequence should reflect the predicted state.
        # However, your model predicts only WQI. The other 16 features are not predicted.
        
        # Create a new row of scaled features for the next timestep
        # Assume other features stay constant as the last historical point
        new_feature_row_scaled = current_sequence[-1, :].copy()
        new_feature_row_scaled[wqi_idx_in_all_numeric] = predicted_wqi_scaled # Update WQI in the new sequence
        
        # Update current_sequence by dropping the oldest and adding the new scaled feature row
        current_sequence = np.vstack([current_sequence[1:], new_feature_row_scaled.reshape(1, -1)])

    return pd.Series(forecasted_wqi, index=forecast_dates)

def get_wqi_classification_and_advice(wqi):
    if wqi < 50:
        return "Excellent", "Water is considered safe and clean, suitable for drinking and aquatic life. Maintain current practices."
    elif 50 <= wqi < 75:
        return "Good", "Water quality is slightly impacted but generally safe for most uses. Minor treatment might be needed for sensitive applications. Monitor regularly."
    elif 75 <= wqi < 90:
        return "Fair", "Water quality is moderately impacted. May not be suitable for drinking without significant treatment. Aquatic life might be stressed. Consider investigating sources of impact and implement mitigation measures."
    else: # WQI >= 90
        return "Poor", "Water quality is severely impacted and generally unsafe for most uses. Significant pollution is present, requiring urgent intervention. Immediate investigation and intervention are required to identify and address pollution sources."

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "General Water Quality Guidelines", "AI Model Information"])

# Filters in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

locations = df_full['Location'].unique().tolist() if 'Location' in df_full.columns else [
    "Tanauan, Batangas", "Talisay, Batangas", "Laurel, Batangas",
    "Agoncillo, Batangas", "San Nicolas, Batangas", "Alitagtag, Batangas",
    "Balete, Batangas", "Mataasnakahoy, Batangas", "Cuenca, Batangas",
    "Lipa City, Batangas", "Malvar, Batangas", "Sto. Tomas, Batangas"
]
selected_location = st.sidebar.selectbox("Select Location", locations)

intervals = ["Weekly", "Monthly", "Yearly"]
selected_interval = st.sidebar.selectbox("Select Interval", intervals)


# --- Helper for Metric Cards (Mimicking Ubidots style) ---
def create_metric_card(title, value, unit, trend_data, icon_path=""):
    st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
    
    # Header row for icon and title
    col_icon, col_title = st.columns([0.2, 0.8])
    with col_icon:
        st.write("📈") # Using an emoji for now
    with col_title:
        st.markdown(f"<p class='metric-title'>{title}</p>", unsafe_allow_html=True)
    
    # Value and trend row
    col_val, col_chart = st.columns([0.4, 0.6])
    with col_val:
        st.markdown(f"<h3 class='metric-value'>{value:.2f} {unit}</h3>", unsafe_allow_html=True)
    with col_chart:
        # Sparkline chart
        fig = go.Figure(data=go.Scatter(y=trend_data, mode='lines', line=dict(width=2, color='#0066cc')))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=60,
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("</div>", unsafe_allow_html=True)


# Home Page
if page == "Home":
    st.title("🌊 Taal Lake Water Quality Monitoring Dashboard")

    st.markdown("""
    Welcome to the Taal Lake Water Quality Prediction Dashboard. This tool leverages advanced machine learning
    (Hybrid CNN-LSTM) to provide insights and predictions on water quality parameters in Taal Lake,
    helping with environmental monitoring and decision-making.
    """)

    st.subheader(f"Current Water Quality Overview for {selected_location}")
    
    # Filter data for selected location
    df_location = df_full[df_full['Location'] == selected_location].copy() if 'Location' in df_full.columns else df_full.copy()
    
    if not df_location.empty:
        # Get latest data for metric cards
        latest_data = df_location.iloc[-1]
        
        # Prepare trend data (last 7 points for sparkline)
        lookback_trend_days = 7 # Adjust as needed
        
        # Use actual features for metric cards now (all 17 features that exist in the dataframe)
        metric_params = {
            "pH": {"col": 'pH Level', "unit": ""},
            "Temperature": {"col": 'Surface Water Temp (°C)', "unit": "°C"},
            "Dissolved Oxygen": {"col": 'Dissolved Oxygen (mg/L)', "unit": "mg/L"},
            "Ammonia": {"col": 'Ammonia (mg/L)', "unit": "mg/L"},
            "Nitrate-N/Nitrite-N": {"col": 'Nitrate-N/Nitrite-N  (mg/L)', "unit": "mg/L"},
            "Phosphate": {"col": 'Phosphate (mg/L)', "unit": "mg/L"},
            "Wind Speed": {"col": 'WIND_SPEED', "unit": "m/s"},
            "Rainfall": {"col": 'RAINFALL', "unit": "mm"},
            "SO2": {"col": 'SO2', "unit": "ppb"},
            "CO2": {"col": 'CO2', "unit": "ppm"},
            "Middle Water Temp": {"col": 'Middle Water Temp (°C)', "unit": "°C"},
            "Bottom Water Temp": {"col": 'Bottom Water Temp (°C)', "unit": "°C"},
        }

        cols = st.columns(2)
        card_index = 0
        for title, info in metric_params.items():
            col_name = info["col"]
            if col_name in df_location.columns:
                current_value = latest_data[col_name]
                trend_data = df_location[col_name].tail(lookback_trend_days).values
                
                with cols[card_index % 2]:
                    create_metric_card(
                        title,
                        current_value, # Display the scaled value for now to avoid complex inverse_transform for individual features
                        info["unit"],
                        trend_data
                    )
                card_index += 1
            else:
                st.warning(f"Metric column '{col_name}' not found in data for {selected_location}.")

    else:
        st.warning(f"No data available for {selected_location}. Please select another location.")
    
    st.subheader(f"Water Quality Forecast for {selected_location} ({selected_interval})")

    if not model or df_full.empty or scaler is None:
        st.warning("Model or data not loaded. Cannot generate forecasts.")
    else:
        # Filter data for selected location and ensure it has enough history
        df_location = df_full[df_full['Location'] == selected_location].copy() if 'Location' in df_full.columns else df_full.copy()
        df_location = df_location.sort_index()

        if len(df_location) < LOOK_BACK:
            st.warning(f"Not enough historical data ({len(df_location)} points) for {LOOK_BACK}-step look-back for {selected_location}. Cannot forecast.")
        else:
            st.subheader("WQI Forecast")
            
            # Dynamically set n_steps based on selected_interval
            n_steps = 0
            if selected_interval == "Weekly":
                n_steps = 7
            elif selected_interval == "Monthly":
                n_steps = 30
            elif selected_interval == "Yearly":
                n_steps = 365

            if n_steps > 0:
                # Forecast WQI
                forecast_wqi_series = forecast_wqi(model, scaler, df_location, n_steps, FEATURE_COLUMNS, TARGET_COLUMN)

                if not forecast_wqi_series.empty:
                    # Create a DataFrame for plotting
                    forecast_df = pd.DataFrame({
                        'Date': forecast_wqi_series.index,
                        'WQI': forecast_wqi_series.values,
                        'Type': 'Forecast'
                    })

                    # Visualize the forecast
                    fig_forecast = px.line(
                        forecast_df,
                        x='Date',
                        y='WQI',
                        title=f'{n_steps}-Day WQI Forecast for {selected_location}',
                        labels={'WQI': 'Water Quality Index'},
                        color='Type',
                        color_discrete_map={'Forecast': 'blue'}
                    )
                    fig_forecast.update_traces(mode='lines')
                    fig_forecast.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    # Display prescriptive analytics
                    avg_forecast_wqi = forecast_wqi_series.mean()
                    classification, advice = get_wqi_classification_and_advice(avg_forecast_wqi)
                    st.subheader("Prescriptive Analytics")
                    st.info(f"**Forecasted Water Quality Status:** {classification}\n\n**Recommendation:** {advice}")
                else:
                    st.warning(f"Not enough historical data to generate a {n_steps}-day forecast for {selected_location}.")
            else:
                st.warning("Please select a valid forecast interval (Weekly, Monthly, or Yearly).")

# Analysis Page
elif page == "General Water Quality Guidelines":
    st.title("General Water Quality Guidelines")
    
    st.subheader("Prescriptive Analytics - General Water Quality Guidelines")
    st.markdown("""
    Based on general water quality guidelines, here's what different WQI ranges typically indicate:
    - **Excellent (WQI < 50):** Water is considered safe and clean, suitable for drinking and aquatic life.
    - **Good (50 ≤ WQI < 75):** Water quality is slightly impacted but generally safe for most uses. Minor treatment might be needed for sensitive applications.
    - **Fair (75 ≤ WQI < 90):** Water quality is moderately impacted. May not be suitable for drinking without significant treatment. Aquatic life might be stressed.
    - **Poor (WQI ≥ 90):** Water quality is severely impacted and generally unsafe for most uses. Significant pollution is present, requiring urgent intervention.

    **Note:** These are general guidelines. Specific local standards and the intended use of the water should always be considered.
    """)

elif page == "AI Model Information":
    st.title("AI Model Information")
    st.markdown("**Developed by:** David Tadeo, Andrew Valera, Jeamuel Pugne, Christian Austero, and Railey Decillo")
    
    st.subheader("Overall Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MSE", "0.0026")
    with col2:
        st.metric("RMSE", "0.0509")
    with col3:
        st.metric("MAE", "0.0277")
    with col4:
        st.metric("R²", "0.7835")

    st.subheader("Model Performance Comparison")
    
    models = ['Hybrid', 'LSTM', 'CNN']
    mse_values = [0.0026, 0.0038, 0.0039]
    rmse_values = [0.0509, 0.0614, 0.0624]
    mae_values = [0.0277, 0.0358, 0.0324]
    r2_values = [0.7835, 0.6844, 0.6654]
    
    metrics_df = pd.DataFrame({
        'Model': models,
        'MSE': mse_values,
        'RMSE': rmse_values,
        'MAE': mae_values,
        'R²': r2_values
    }).set_index('Model')
    
    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['R²']).highlight_min(axis=0, subset=['MSE', 'RMSE', 'MAE']))

    fig_mse = px.bar(metrics_df, x=metrics_df.index, y='MSE', title='Model Performance: Mean Squared Error')
    st.plotly_chart(fig_mse, use_container_width=True)

    fig_r2 = px.bar(metrics_df, x=metrics_df.index, y='R²', title='Model Performance: R-squared')
    st.plotly_chart(fig_r2, use_container_width=True)