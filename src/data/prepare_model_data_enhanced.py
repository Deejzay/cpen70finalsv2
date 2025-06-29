import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Load the processed data
data_path = 'data/processed/processed_data.csv'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Define weights for WQI calculation
wqi_weights = {
    'pH Level': 0.15,
    'Dissolved Oxygen (mg/L)': 0.25,
    'Nitrate-N/Nitrite-N  (mg/L)': 0.10,
    'Ammonia (mg/L)': 0.15,
    'Phosphate (mg/L)': 0.10,
    'Surface Water Temp (°C)': 0.05,
    'Middle Water Temp (°C)': 0.05,
    'Bottom Water Temp (°C)': 0.05,
}

# Define pollutant parameters and their thresholds
pollutant_params = {
    'Ammonia (mg/L)': {'threshold': 0.5, 'weight': 0.4},  # High weight due to toxicity
    'Nitrate-N/Nitrite-N  (mg/L)': {'threshold': 10.0, 'weight': 0.35},  # EPA standard
    'Phosphate (mg/L)': {'threshold': 0.1, 'weight': 0.25}  # Eutrophication indicator
}

# Calculate WQI
print('DataFrame shape before imputation:', df.shape)
print('NaN counts before imputation:')
print(df.isna().sum())

for col in wqi_weights.keys():
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Apply forward fill, then backward fill to handle missing values in WQI columns
wqi_cols = list(wqi_weights.keys())
df[wqi_cols] = df[wqi_cols].ffill().bfill()

print('DataFrame shape after imputation:', df.shape)
print('NaN counts after imputation:')
print(df.isna().sum())

# Impute missing values in non-WQI columns
non_wqi_cols = [col for col in df.columns if col not in wqi_cols]
df[non_wqi_cols] = df[non_wqi_cols].fillna(method='ffill').fillna(method='bfill')

print('DataFrame shape after imputing non-WQI columns:', df.shape)
print('NaN counts after imputing non-WQI columns:')
print(df.isna().sum())

def calculate_wqi(df, weights):
    """Calculate Water Quality Index"""
    weighted_values = df[weights.keys()].apply(lambda x: x * weights[x.name], axis=0)
    wqi = weighted_values.sum(axis=1)
    return wqi

def calculate_pollutant_level(df, pollutant_params):
    """
    Calculate Pollutant Level based on ammonia, nitrate, and phosphate concentrations.
    Returns a score from 0-100 where:
    0-25: Low pollution
    26-50: Moderate pollution  
    51-75: High pollution
    76-100: Very high pollution
    """
    pollutant_scores = []
    
    for _, row in df.iterrows():
        total_score = 0
        
        for pollutant, params in pollutant_params.items():
            concentration = row[pollutant]
            threshold = params['threshold']
            weight = params['weight']
            
            # Calculate individual pollutant score (0-100)
            if concentration <= threshold:
                # Below threshold: score based on how close to threshold
                score = (concentration / threshold) * 25  # 0-25 range
            else:
                # Above threshold: score based on how much over threshold
                excess_ratio = concentration / threshold
                if excess_ratio <= 2:
                    score = 25 + (excess_ratio - 1) * 25  # 25-50 range
                elif excess_ratio <= 5:
                    score = 50 + (excess_ratio - 2) * 8.33  # 50-75 range
                else:
                    score = 75 + min((excess_ratio - 5) * 5, 25)  # 75-100 range
            
            # Apply weight and add to total
            total_score += score * weight
        
        pollutant_scores.append(total_score)
    
    return np.array(pollutant_scores)

# Calculate both WQI and Pollutant Level
df['WQI'] = calculate_wqi(df, wqi_weights)
df['Pollutant_Level'] = calculate_pollutant_level(df, pollutant_params)

# Handle missing values (impute or drop)
df = df.dropna()

# Normalize features (fit scaler on all 18 features except 'Date' and 'Location')
feature_cols = [
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
print('\nColumns used for scaling:', feature_cols)
print('Are all columns present in df?', all([col in df.columns for col in feature_cols]))
print('First 3 rows of features to be scaled:')
print(df[feature_cols].head(3))
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
joblib.dump(scaler, 'data/processed/scaler.pkl')

# Save the enhanced processed DataFrame
df.to_csv('data/processed/processed_data_enhanced.csv', index=True)

# Create sequences for multi-target prediction
def create_multi_target_sequences(data, target_cols, look_back=12):
    """
    Create sequences for multiple target variables
    """
    # Drop the 'Location' column if it exists
    if 'Location' in data.columns:
        data = data.drop(columns=['Location'])
        
    X, y_dict = [], {col: [] for col in target_cols}
    
    for i in range(len(data) - look_back):
        X.append(data.iloc[i:(i + look_back)].values)
        for col in target_cols:
            y_dict[col].append(data.iloc[i + look_back][col])
    
    return np.array(X), {col: np.array(y_dict[col]) for col in target_cols}

# Create sequences for both WQI and Pollutant Level prediction
target_columns = ['WQI', 'Pollutant_Level']
X, y_dict = create_multi_target_sequences(df, target_columns)

# Split data into train/test sets (80/20 split)
# First split X, then split each target separately
X_train, X_test, train_indices, test_indices = train_test_split(
    X, range(len(X)), test_size=0.2, random_state=42
)

# Split targets using the same indices
y_train_dict = {}
y_test_dict = {}
for target in target_columns:
    y_train_dict[target] = y_dict[target][train_indices]
    y_test_dict[target] = y_dict[target][test_indices]

# Save the enhanced training data
np.save('data/processed/X_train_enhanced.npy', X_train.astype(np.float32))
np.save('data/processed/X_test_enhanced.npy', X_test.astype(np.float32))

# Save individual target arrays
for target in target_columns:
    np.save(f'data/processed/y_train_{target.lower()}.npy', 
            y_train_dict[target].astype(np.float32))
    np.save(f'data/processed/y_test_{target.lower()}.npy', 
            y_test_dict[target].astype(np.float32))

# Save combined target arrays for backward compatibility
np.save('data/processed/y_train.npy', y_train_dict['WQI'].astype(np.float32))
np.save('data/processed/y_test.npy', y_test_dict['WQI'].astype(np.float32))

# Print summary statistics
print('\n=== Enhanced Data Preparation Summary ===')
print(f'Total samples: {len(df)}')
print(f'Features: {len(df.columns) - 2}')  # Excluding WQI and Pollutant_Level
print(f'Targets: {len(target_columns)}')

print('\nWQI Statistics:')
print(f'Mean: {df["WQI"].mean():.3f}')
print(f'Std: {df["WQI"].std():.3f}')
print(f'Min: {df["WQI"].min():.3f}')
print(f'Max: {df["WQI"].max():.3f}')

print('\nPollutant Level Statistics:')
print(f'Mean: {df["Pollutant_Level"].mean():.3f}')
print(f'Std: {df["Pollutant_Level"].std():.3f}')
print(f'Min: {df["Pollutant_Level"].min():.3f}')
print(f'Max: {df["Pollutant_Level"].max():.3f}')

print('\nPollutant Level Classification:')
pollutant_levels = df['Pollutant_Level']
low_pollution = len(pollutant_levels[pollutant_levels <= 25])
moderate_pollution = len(pollutant_levels[(pollutant_levels > 25) & (pollutant_levels <= 50)])
high_pollution = len(pollutant_levels[(pollutant_levels > 50) & (pollutant_levels <= 75)])
very_high_pollution = len(pollutant_levels[pollutant_levels > 75])

print(f'Low pollution (0-25): {low_pollution} samples ({low_pollution/len(df)*100:.1f}%)')
print(f'Moderate pollution (26-50): {moderate_pollution} samples ({moderate_pollution/len(df)*100:.1f}%)')
print(f'High pollution (51-75): {high_pollution} samples ({high_pollution/len(df)*100:.1f}%)')
print(f'Very high pollution (76-100): {very_high_pollution} samples ({very_high_pollution/len(df)*100:.1f}%)')

print("\nEnhanced data preparation completed. Multi-target training sets saved in data/processed/.") 