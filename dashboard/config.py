from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

NODES_FILE = DATA_DIR / "hcmc_nodes.csv"
TRAFFIC_FILE = DATA_DIR / "traffic_data.csv"
CORRELATION_FILE = OUTPUTS_DIR / "correlation_matrix.csv"

TRAFFIC_COLUMNS = [
    "timestamp",
    "node_id",
    "road_name",
    "latitude",
    "longitude",
    "current_avg_speed_kmh",
    "free_flow_speed_kmh",
    "current_travel_time_seconds",
    "free_flow_travel_time_seconds",
    "confidence_score_0_to_1",
    "congestion_ratio_0_to_1",
]

NUMERIC_TRAFFIC_COLUMNS = [
    "latitude",
    "longitude",
    "current_avg_speed_kmh",
    "free_flow_speed_kmh",
    "current_travel_time_seconds",
    "free_flow_travel_time_seconds",
    "confidence_score_0_to_1",
    "congestion_ratio_0_to_1",
]
