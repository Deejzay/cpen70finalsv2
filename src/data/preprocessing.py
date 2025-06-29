import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

def load_climate_data(file_path):
    """
    Load and preprocess climate data from CSV file
    """
    df = pd.read_csv(file_path)
    
    # Handle missing values (-999)
    df = df.replace(-999, np.nan)
    
    # Create datetime index
    df['Date'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
    df = df.set_index('Date')
    
    # Drop original year and month columns
    df = df.drop(['YEAR', 'MONTH'], axis=1)
    
    return df

def load_volcanic_data(so2_file, co2_file):
    """
    Load and preprocess volcanic activity data from Excel files
    """
    # Load SO2 data (header in second row)
    so2_df = pd.read_excel(so2_file, header=1)
    so2_df = so2_df.rename(columns={so2_df.columns[0]: 'Date', so2_df.columns[1]: 'SO2'})
    so2_df['Date'] = pd.to_datetime(so2_df['Date'], dayfirst=True, errors='coerce')
    so2_df = so2_df[['Date', 'SO2']]
    
    # Load CO2 data (header in second row)
    co2_df = pd.read_excel(co2_file, header=1)
    co2_df = co2_df.rename(columns={co2_df.columns[0]: 'Date', co2_df.columns[1]: 'CO2'})
    co2_df['Date'] = pd.to_datetime(co2_df['Date'], dayfirst=True, errors='coerce')
    co2_df = co2_df[['Date', 'CO2']]
    
    # Merge volcanic data on Date
    volcanic_df = pd.merge(so2_df, co2_df, on='Date', how='outer')
    volcanic_df = volcanic_df.set_index('Date')
    
    return volcanic_df

def load_water_parameters(file_path):
    """
    Load and preprocess water parameters data from Excel file
    """
    df = pd.read_excel(file_path)
    
    # Preserve location information if it exists
    location_col = None
    for col in df.columns:
        if 'location' in col.lower() or 'site' in col.lower():
            location_col = col
            break
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m', errors='coerce')
        df = df.set_index('Date')
    else:
        # Fallback: try to construct date from Year/Month columns if present
        year_cols = [col for col in df.columns if 'year' in col.lower()]
        month_cols = [col for col in df.columns if 'month' in col.lower()]
        if year_cols and month_cols:
            df['Date'] = pd.to_datetime(df[year_cols[0]].astype(str) + '-' + df[month_cols[0]].astype(str) + '-01', errors='coerce')
            df = df.set_index('Date')
    
    # If location column exists, rename it to 'Location' for consistency
    if location_col:
        df = df.rename(columns={location_col: 'Location'})
    
    return df

def preprocess_data(climate_df, volcanic_df, water_df):
    """
    Preprocess and merge all datasets
    """
    # Handle missing values using KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    
    # Preprocess climate data
    climate_cols = climate_df.select_dtypes(include=['float64', 'int64']).columns
    climate_df[climate_cols] = imputer.fit_transform(climate_df[climate_cols])
    
    # Preprocess volcanic data
    volcanic_cols = volcanic_df.select_dtypes(include=['float64', 'int64']).columns
    volcanic_df[volcanic_cols] = imputer.fit_transform(volcanic_df[volcanic_cols])
    
    # Preprocess water parameters
    water_cols = water_df.select_dtypes(include=['float64', 'int64']).columns
    water_df[water_cols] = imputer.fit_transform(water_df[water_cols])
    
    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler()
    climate_df[climate_cols] = scaler.fit_transform(climate_df[climate_cols])
    volcanic_df[volcanic_cols] = scaler.fit_transform(volcanic_df[volcanic_cols])
    water_df[water_cols] = scaler.fit_transform(water_df[water_cols])
    
    # Reset index to make Date a column for merging
    climate_df = climate_df.reset_index()
    volcanic_df = volcanic_df.reset_index()
    water_df = water_df.reset_index()
    
    # Merge all datasets on Date
    merged_df = pd.merge(climate_df, volcanic_df, on='Date', how='outer')
    final_df = pd.merge(merged_df, water_df, on='Date', how='outer')
    
    # Sort by Date
    final_df = final_df.sort_values('Date')
    
    # Set Date as index
    final_df = final_df.set_index('Date')
    
    return final_df, scaler

def main():
    """
    Main function to run the preprocessing pipeline
    """
    # Load datasets
    climate_df = load_climate_data('raw_data/Ambulong-Monthly-Data.csv')
    volcanic_df = load_volcanic_data('raw_data/TV_SO2_Flux_2020-2024.xlsx', 
                                   'raw_data/TV_CO2_Flux_2013-2019.xlsx')
    water_df = load_water_parameters('raw_data/Water-Parameters_2013-2025-CpE-copy.xlsx')
    
    # Print info about loaded datasets
    print("\nClimate Data Shape:", climate_df.shape)
    print("Volcanic Data Shape:", volcanic_df.shape)
    print("Water Parameters Shape:", water_df.shape)
    
    # Preprocess and merge data
    final_df, scaler = preprocess_data(climate_df, volcanic_df, water_df)
    
    # Print info about final dataset
    print("\nFinal Dataset Shape:", final_df.shape)
    print("\nColumns in final dataset:", final_df.columns.tolist())
    
    # Save processed data
    final_df.to_csv('data/processed/processed_data.csv')
    
    return final_df, scaler

if __name__ == "__main__":
    final_df, scaler = main() 