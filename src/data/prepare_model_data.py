import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the processed data
data_path = 'data/processed/processed_data.csv'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Define weights for WQI calculation
weights = {
    'pH Level': 0.15,
    'Dissolved Oxygen (mg/L)': 0.25,
    'Nitrate-N/Nitrite-N  (mg/L)': 0.10,
    'Ammonia (mg/L)': 0.15,
    'Phosphate (mg/L)': 0.10,
    'Surface Water Temp (°C)': 0.05,
    'Middle Water Temp (°C)': 0.05,
    'Bottom Water Temp (°C)': 0.05,
}

# Calculate WQI
# Ensure all columns used in WQI are numeric
print('DataFrame shape before imputation:', df.shape)
print('NaN counts before imputation:')
print(df.isna().sum())

for col in weights.keys():
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Apply forward fill, then backward fill to handle missing values in WQI columns
wqi_cols = list(weights.keys())
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
    weighted_values = df[weights.keys()].apply(lambda x: x * weights[x.name], axis=0)
    wqi = weighted_values.sum(axis=1)
    return wqi

df['WQI'] = calculate_wqi(df, weights)

# Handle missing values (impute or drop)
df = df.dropna()

# Normalize features
scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save the processed DataFrame to CSV, including the WQI, normalized features, and Location
df.to_csv('data/processed/processed_data.csv', index=True)

# Create sequences for time series models (e.g., LSTM)
def create_sequences(data, target_col, look_back=12):
    # Drop the 'Location' column if it exists, as it's not a numerical feature for model input
    if 'Location' in data.columns:
        data = data.drop(columns=['Location'])
        
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data.iloc[i:(i + look_back)].values)
        y.append(data.iloc[i + look_back][target_col])
    return np.array(X), np.array(y)

# Create sequences for WQI prediction
X, y = create_sequences(df, 'WQI')

# Split data into train/test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Before saving, ensure arrays are float32
np.save('data/processed/X_train.npy', X_train.astype(np.float32))
np.save('data/processed/X_test.npy', X_test.astype(np.float32))
np.save('data/processed/y_train.npy', y_train.astype(np.float32))
np.save('data/processed/y_test.npy', y_test.astype(np.float32))

# After imputation, print non-null counts for all columns
print('Non-null counts for all columns after imputation:')
print(df.count())

print("Data preparation completed. Train/test sets saved in data/processed/.") 