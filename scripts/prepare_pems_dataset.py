import pandas as pd
import numpy as np
import os

def process_and_package_data():
    input_file = 'outputs/traffic_data_cleaned.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: The file {input_file} was not found.")
        return

    print("1. Reading cleaned traffic data...")
    df = pd.read_csv(input_file)
    
    # Clean column names to prevent KeyErrors
    df.columns = df.columns.str.strip()
    
    # Remove rows where the data literally contains the header string "timestamp"
    df = df[df['timestamp'] != 'timestamp']
    
    # Convert to datetime safely. errors='coerce' forces invalid formats to become NaT (Not a Time)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop any rows that failed the datetime conversion
    df = df.dropna(subset=['timestamp'])

    # ---------------------------------------------------------
    # STEP 1: CONSTRUCT SPATIO-TEMPORAL MATRIX
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # STEP 1: CONSTRUCT SPATIO-TEMPORAL MATRIX
    # ---------------------------------------------------------
    print("2. Constructing continuous time-series matrix...")
    
    # Change values='current_speed' to values='current_avg_speed_kmh'
    df_pivot = df.pivot_table(index='timestamp', columns='road_name', values='current_avg_speed_kmh')
    
    # FIX: Resample to strict 15-minute bins and aggregate using the mean
    df_pivot = df_pivot.resample('15min').mean()
    
    missing_rate_before = df_pivot.isna().mean().mean() * 100
    print(f"-> Missing Rate BEFORE imputation: {missing_rate_before:.2f}%")

    # ---------------------------------------------------------
    # STEP 2: DATA IMPUTATION (LINEAR INTERPOLATION)
    # ---------------------------------------------------------
    print("3. Performing Linear Interpolation...")
    df_imputed = df_pivot.interpolate(method='linear', limit_direction='both')
    
    missing_rate_after = df_imputed.isna().mean().mean() * 100
    print(f"-> Missing Rate AFTER imputation: {missing_rate_after:.2f}%")
    
    if missing_rate_after < 5:
        print("-> Target achieved: Missing Rate is < 5%!")
    else:
        print("-> Warning: Missing Rate is still > 5%. Check for prolonged network outages.")

    df_imputed.to_csv('outputs/traffic_data_imputed.csv', index_label='timestamp')

    # ---------------------------------------------------------
    # STEP 3: PACKAGE TO .NPZ FORMAT (PeMSD4 STANDARD)
    # ---------------------------------------------------------
    print("4. Packaging data to standard .npz format (T, N, F)...")
    data_matrix = df_imputed.values 
    T, N = data_matrix.shape
    F = 1 
    data_npz = data_matrix.reshape((T, N, F))
    
    npz_output_path = 'outputs/HCMC_Traffic_Data.npz'
    np.savez(npz_output_path, data=data_npz)
    print(f"-> Successfully saved Numpy Zipped file: {npz_output_path}")
    print(f"-> Tensor Shape: {data_npz.shape} (Time steps, Nodes, Features)")

    nodes = list(df_imputed.columns)
    mapping_df = pd.DataFrame({'node_id': range(len(nodes)), 'road_name': nodes})
    mapping_df.to_csv('outputs/road_nodes_mapping.csv', index=False)
    print("-> Saved Node ID mapping to outputs/road_nodes_mapping.csv")
    print("Process completed successfully!")

if __name__ == "__main__":
    process_and_package_data()