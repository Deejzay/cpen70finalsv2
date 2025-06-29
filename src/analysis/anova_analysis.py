import pandas as pd
import numpy as np
from scipy import stats

def perform_one_way_anova_by_location(data_path, parameter_name):
    """
    Performs a One-Way ANOVA test for a specified water quality parameter
    across different locations.

    Args:
        data_path (str): Path to the processed CSV data file (e.g., 'data/processed/processed_data.csv').
        parameter_name (str): The name of the water quality parameter to test (e.g., 'pH Level', 'Dissolved Oxygen (mg/L)').
    """
    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please ensure 'processed_data.csv' exists.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if 'Location' not in df.columns:
        print("Error: 'Location' column not found in the DataFrame. Cannot perform ANOVA by location.")
        return

    if parameter_name not in df.columns:
        print(f"Error: Parameter '{parameter_name}' not found in the DataFrame.")
        print(f"Available parameters are: {df.columns.tolist()}")
        return

    unique_locations = df['Location'].unique()

    if len(unique_locations) < 2:
        print("One-Way ANOVA requires at least two unique locations to compare.")
        return

    data_groups = []
    group_names = []
    
    print(f"Preparing data for One-Way ANOVA on '{parameter_name}' across locations...")
    for location in unique_locations:
        # Filter data for the current location and the specified parameter
        location_data = df[df['Location'] == location][parameter_name].dropna()
        if not location_data.empty:
            data_groups.append(location_data.values)
            group_names.append(location)
            print(f"  - '{location}': {len(location_data)} data points, Mean: {np.mean(location_data):.4f}")
        else:
            print(f"  - Warning: No valid data for '{parameter_name}' found for location '{location}'. Skipping.")

    if len(data_groups) < 2:
        print("Insufficient valid data groups after filtering by location for ANOVA.")
        return

    # Perform the One-Way ANOVA test
    f_statistic, p_value = stats.f_oneway(*data_groups)

    print(f"
--- One-Way ANOVA Results for {parameter_name} across Locations ---")
    print(f"F-statistic: {f_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpret the results
    alpha = 0.05 # Significance level
    if p_value < alpha:
        print(f"
Conclusion: Since the p-value ({p_value:.4f}) is less than the significance level ({alpha}),")
        print("we reject the null hypothesis. There is a statistically significant difference")
        print(f"in the mean {parameter_name} among the compared locations.")
    else:
        print(f"
Conclusion: Since the p-value ({p_value:.4f}) is greater than the significance level ({alpha}),")
        print("we fail to reject the null hypothesis. There is no statistically significant difference")
        print(f"in the mean {parameter_name} among the compared locations.")

if __name__ == "__main__":
    # Define the path to your processed data
    processed_data_file = 'data/processed/processed_data.csv'

    # --- Example Usage for your project ---
    # You can choose any parameter from your 'processed_data.csv' that you want to compare
    # List of common parameters: 'pH Level', 'Dissolved Oxygen (mg/L)', 'Surface Water Temp (°C)', 'WQI', etc.

    print("Running ANOVA for 'pH Level':")
    perform_one_way_anova_by_location(processed_data_file, 'pH Level')

    print("
" + "="*70 + "
")

    print("Running ANOVA for 'Dissolved Oxygen (mg/L)':")
    perform_one_way_anova_by_location(processed_data_file, 'Dissolved Oxygen (mg/L)')

    print("
" + "="*70 + "
")

    print("Running ANOVA for 'WQI':")
    perform_one_way_anova_by_location(processed_data_file, 'WQI')

    # You can add more calls to test other parameters as needed.
    # For example:
    # print("
" + "="*70 + "
")
    # print("Running ANOVA for 'Surface Water Temp (°C)':")
    # perform_one_way_anova_by_location(processed_data_file, 'Surface Water Temp (°C)') 