import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. DATA PREPROCESSING
# ---------------------------------------------------------
# Read the CSV file
df = pd.read_csv('traffic_data.csv')

# Convert the timestamp column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Round time to 15-minute intervals to synchronize time series
df['timestamp_rounded'] = df['timestamp'].dt.round('15min')

# ---------------------------------------------------------
# 2. PLOT HEATMAP (CORRELATION MATRIX)
# ---------------------------------------------------------
# Create a pivot table to calculate the correlation matrix between roads
pivot_df_rounded = df.pivot_table(index='timestamp_rounded', 
                                  columns='road_name', 
                                  values='congestion_ratio_0_to_1')

# Calculate Pearson correlation matrix
corr_matrix_rounded = pivot_df_rounded.corr()

# Set figure size and plot the Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix_rounded, 
            annot=True,          # Display numbers in each cell
            fmt=".2f",           # Round to 2 decimal places
            cmap='coolwarm',     # Color map from cold (blue) to hot (red)
            vmin=0, vmax=1,      # Limit the scale from 0 to 1
            linewidths=0.5)      # Cell borders
plt.title('Pearson Correlation of Congestion Ratio (Data-driven Hyperedge)', fontsize=14, pad=15)
plt.xlabel('Road Name')
plt.ylabel('Road Name')
plt.tight_layout()

# Save and display the image
plt.savefig('heatmap_correlation.png', dpi=300) # dpi=300 for high resolution
plt.show()

# ---------------------------------------------------------
# 3. PLOT LINE CHART (TIME SERIES ON SEPT 10)
# ---------------------------------------------------------
# Filter data for the best performing day (2026-09-10)
target_date = pd.to_datetime('2026-09-10').date()
df_day = df[df['timestamp'].dt.date == target_date]

# Create a pivot table for Sept 10
df_day_rounded = df_day.pivot_table(index='timestamp_rounded', 
                                    columns='road_name', 
                                    values='congestion_ratio_0_to_1')

# Set figure size and plot the line chart
plt.figure(figsize=(14, 6))
for column in df_day_rounded.columns:
    plt.plot(df_day_rounded.index, 
             df_day_rounded[column], 
             marker='o',         # Display data points
             linewidth=2,        # Line width
             label=column)

plt.title('Congestion Ratio over Time (2026-09-10)', fontsize=16, pad=15)
plt.xlabel('Time (Hours)', fontsize=12)
plt.ylabel('Congestion Ratio', fontsize=12)
plt.legend(title="Road Name", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6) # Add grid for readability
plt.xticks(rotation=45) # Rotate X-axis labels to avoid overlap
plt.tight_layout()

# Save and display the image
plt.savefig('timeseries_congestion.png', dpi=300)
plt.show()

print("Successfully generated and saved 2 charts: heatmap_correlation.png and timeseries_congestion.png")