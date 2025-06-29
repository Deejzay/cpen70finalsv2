import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")

# Load the processed data
data_path = os.path.join('data', 'processed', 'processed_data.csv')
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Filter only numeric columns for EDA
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Summary statistics
print("Summary Statistics:")
print(numeric_df.describe())

# Missing value analysis
print("\nMissing Values:")
print(numeric_df.isnull().sum())

# Correlation matrix
correlation_matrix = numeric_df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Ensure output directory exists
output_dir = os.path.join('data', 'processed')
os.makedirs(output_dir, exist_ok=True)

def sanitize_filename(name):
    # Replace or remove problematic characters for Windows filenames
    return (
        name.replace(' ', '_')
            .replace('/', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('°', 'deg')
            .replace('<', '')
            .replace('>', '')
            .replace(':', '')
            .replace('?', '')
            .replace('"', '')
            .replace('|', '')
    )

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()

# Time series plots for key variables
key_vars = ['RAINFALL', 'TMAX', 'TMIN', 'SO2', 'CO2', 'Surface Water Temp (°C)', 'pH Level', 'Dissolved Oxygen (mg/L)']
for var in key_vars:
    if var in numeric_df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(numeric_df.index, numeric_df[var], label=var)
        plt.title(f'Time Series Plot for {var}')
        plt.xlabel('Date')
        plt.ylabel(var)
        plt.legend()
        plt.tight_layout()
        fname = sanitize_filename(f'{var}_timeseries.png')
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

# Distribution plots for each variable
for var in numeric_df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(numeric_df[var].dropna(), kde=True)
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.tight_layout()
    fname = sanitize_filename(f'{var}_distribution.png')
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

print("EDA completed. Plots saved in data/processed/.") 