import requests
import csv
import os
import time
from datetime import datetime

# WARNING: Never push your real API key to GitHub! 
# Consider using environment variables (.env) before your final commit.
API_KEY = "vN1ggLlHhYiblZ6GXDxbPRVaemr1NBqQ"

# FILE PATHS (following the new GitHub-friendly directory structure)
INPUT_FILE = "data/hcmc_nodes.csv"
OUTPUT_FILE = "data/traffic_data.csv"

# ============================================================
# COLUMN LEGEND (what each column in traffic_data.csv means)
# ------------------------------------------------------------
# timestamp                      : when this reading was taken (local time)
# node_id                        : unique ID for the intersection
# road_name                      : name of the monitored road segment
# latitude / longitude           : coordinates of the monitoring point
# current_avg_speed_kmh          : real-time average speed of traffic (km/h)
# free_flow_speed_kmh            : expected speed with no congestion (km/h)
# current_travel_time_seconds    : time to travel the segment right now (s)
# free_flow_travel_time_seconds  : time to travel the segment with no traffic (s)
# confidence_score_0_to_1        : TomTom's confidence in this reading (0=low, 1=high)
# congestion_ratio_0_to_1        : 1 - (current_speed / free_flow_speed)
#                                  0 = no congestion, closer to 1 = heavier congestion

def fetch_data_for_point(road_name, lat, lon):
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "point": f"{lat},{lon}",
        "key": API_KEY,
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()["flowSegmentData"]

        current_speed = data.get("currentSpeed")
        free_flow_speed = data.get("freeFlowSpeed")

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "road_name": road_name,
            "latitude": lat,
            "longitude": lon,
            "current_avg_speed_kmh": current_speed,
            "free_flow_speed_kmh": free_flow_speed,
            "current_travel_time_seconds": data.get("currentTravelTime"),
            "free_flow_travel_time_seconds": data.get("freeFlowTravelTime"),
            "confidence_score_0_to_1": data.get("confidence"),
            "congestion_ratio_0_to_1": round(1 - (current_speed / free_flow_speed), 2)
            if current_speed is not None and free_flow_speed else None,
        }
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch data for {road_name}: {e}")
        return None

def collect_once():
    # Ensure the 'data' directory exists before reading/writing
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        print("Please run the OSM node generation script first and place it in the 'data/' folder.")
        return

    file_exists = os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8-sig") as out_f:
        columns = [
            "timestamp", "node_id", "road_name", "latitude", "longitude",
            "current_avg_speed_kmh", "free_flow_speed_kmh",
            "current_travel_time_seconds", "free_flow_travel_time_seconds",
            "confidence_score_0_to_1", "congestion_ratio_0_to_1",
        ]
        writer = csv.DictWriter(out_f, fieldnames=columns)

        if not file_exists:
            writer.writeheader()

        # Read coordinates directly from our generated CSV
        with open(INPUT_FILE, mode="r", encoding="utf-8-sig") as in_f:
            reader = csv.DictReader(in_f)
            
            for row in reader:
                node_id = row.get("node_id", "Unknown")
                road_name = row.get("road_name", "Unknown")
                lat = row.get("lat")
                lon = row.get("lon")
                
                if not lat or not lon:
                    continue
                
                result = fetch_data_for_point(road_name, lat, lon)
                
                if result:
                    # Add node_id to the result before writing
                    result["node_id"] = node_id
                    writer.writerow(result)
                    
                    print(
                        f"Node {node_id} | {road_name}: "
                        f"{result['current_avg_speed_kmh']} km/h "
                        f"(free flow: {result['free_flow_speed_kmh']} km/h)"
                    )
                
                # CRITICAL: Sleep to prevent TomTom API rate limiting
                time.sleep(0.5)

    print(f"\nData saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    collect_once()