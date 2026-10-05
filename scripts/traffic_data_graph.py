import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / 'data' / 'traffic_data.csv'
OUTPUT_DIR = ROOT_DIR / 'outputs'

CSV_COLUMNS = [
    'timestamp', 'node_id', 'road_name', 'latitude', 'longitude',
    'current_avg_speed_kmh', 'free_flow_speed_kmh',
    'current_travel_time_seconds', 'free_flow_travel_time_seconds',
    'confidence_score_0_to_1', 'congestion_ratio_0_to_1',
]


def filter_traffic_data(data):
    data['confidence_score_0_to_1'] = pd.to_numeric(
        data['confidence_score_0_to_1'], errors='coerce'
    )
    data['congestion_ratio_0_to_1'] = pd.to_numeric(
        data['congestion_ratio_0_to_1'], errors='coerce'
    )

    confident_data = data.loc[
        data['confidence_score_0_to_1'] > 0.8
    ].copy()

    by_road = confident_data.groupby('road_name', dropna=False)[
        'congestion_ratio_0_to_1'
    ]
    q1 = by_road.transform(lambda values: values.quantile(0.25))
    q3 = by_road.transform(lambda values: values.quantile(0.75))
    iqr = q3 - q1
    inlier_mask = confident_data['congestion_ratio_0_to_1'].between(
        q1 - 1.5 * iqr,
        q3 + 1.5 * iqr,
        inclusive='both',
    )
    filtered_data = confident_data.loc[inlier_mask].copy()

    print(
        f"Confidence filter: {len(data) - len(confident_data)} rows removed; "
        f"IQR filter: {len(confident_data) - len(filtered_data)} rows removed."
    )
    return filtered_data

# ---------------------------------------------------------
# 1. DATA PREPROCESSING
# ---------------------------------------------------------
# Read either the current headerless CSV or a CSV with column names.
df = pd.read_csv(DATA_FILE)
if 'timestamp' not in df.columns:
    df = pd.read_csv(DATA_FILE, header=None, names=CSV_COLUMNS)

df = filter_traffic_data(df)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_DIR / 'traffic_data_cleaned.csv', index=False)

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
corr_matrix_rounded.to_csv(OUTPUT_DIR / 'correlation_matrix.csv')

# Keep the full correlation matrix, but plot only the 15 roads with the
# greatest congestion variability so labels and values remain readable.
road_count = min(15, len(pivot_df_rounded.columns))
selected_roads = pivot_df_rounded.var().nlargest(road_count).index
plot_corr = corr_matrix_rounded.loc[selected_roads, selected_roads]

fig, ax = plt.subplots(figsize=(max(10, road_count * 0.7), max(8, road_count * 0.6)))
sns.heatmap(
    plot_corr,
    annot=True,
    fmt=".2f",
    cmap='RdBu_r',
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    annot_kws={'size': 8},
    cbar_kws={'label': 'Pearson correlation'},
    ax=ax,
)
ax.set_title(
    f'Congestion correlation: {road_count} roads with most variation',
    fontsize=14,
    pad=15,
)
ax.set_xlabel('Road name')
ax.set_ylabel('Road name')
ax.tick_params(axis='x', labelrotation=45)
ax.tick_params(axis='y', labelrotation=0)
plt.setp(ax.get_xticklabels(), ha='right')
fig.tight_layout()

# Save and display the image
fig.savefig(OUTPUT_DIR / 'heatmap_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 3. PLOT LINE CHART (TIME SERIES ON THE LATEST DATE)
# ---------------------------------------------------------
# Use the latest date available in the filtered data.
target_date = df['timestamp'].dt.date.max()
df_day = df[df['timestamp'].dt.date == target_date]

# Create a pivot table for the latest date
df_day_rounded = df_day.pivot_table(index='timestamp_rounded', 
                                    columns='road_name', 
                                    values='congestion_ratio_0_to_1')

# Plot only the 10 most variable roads to keep the time series readable.
day_road_count = min(10, len(df_day_rounded.columns))
day_roads = df_day_rounded.var().nlargest(day_road_count).index

plt.figure(figsize=(14, 6))
for column in day_roads:
    plt.plot(df_day_rounded.index, 
             df_day_rounded[column], 
             marker='o',         # Display data points
             linewidth=2,        # Line width
             label=column)

plt.title(
    f'Congestion Ratio over Time ({target_date}) - '
    f'{day_road_count} roads with most variation',
    fontsize=16,
    pad=15,
)
plt.xlabel('Time (Hours)', fontsize=12)
plt.ylabel('Congestion Ratio', fontsize=12)
plt.legend(title="Road Name", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6) # Add grid for readability
plt.xticks(rotation=45) # Rotate X-axis labels to avoid overlap
plt.tight_layout()

# Save and display the image
plt.savefig(OUTPUT_DIR / 'timeseries_congestion.png', dpi=300)
plt.show()

print("Successfully generated and saved 2 charts: heatmap_correlation.png and timeseries_congestion.png")