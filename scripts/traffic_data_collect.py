import requests
import csv
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file for local testing
load_dotenv()

# Retrieve the comma-separated string of API keys from the environment
api_keys_env = os.getenv("TOMTOM_API_KEYS", "")

# Convert the string into a list of keys, removing any accidental whitespace
API_KEYS = [key.strip() for key in api_keys_env.split(",") if key.strip()]

current_key_index = 0

# FILE PATHS
INPUT_FILE = "data/hcmc_nodes.csv"
OUTPUT_FILE = "data/traffic_data.csv"

def fetch_data_for_point(road_name, lat, lon):
    global current_key_index
    
    if not API_KEYS:
        print("[CRITICAL] No API keys found. Please check your .env file or TOMTOM_API_KEYS variable.")
        return None
        
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    
    while current_key_index < len(API_KEYS):
        params = {
            "point": f"{lat},{lon}",
            "key": API_KEYS[current_key_index],
        }
        try:
            res = requests.get(url, params=params, timeout=10)
            
            # Rotate key if rate limit (429) or forbidden/quota exhausted (403) is hit
            if res.status_code in [403, 429]:
                print(f"[WARNING] API Key {current_key_index + 1} exhausted (Status {res.status_code}). Switching to next key...")
                current_key_index += 1
                continue

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
            
    print("[CRITICAL] All provided API keys have been exhausted for today!")
    return None

def collect_once():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
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

        with open(INPUT_FILE, mode="r", encoding="utf-8-sig") as in_f:
            reader = csv.DictReader(in_f)
            
            for row in reader:
                node_id = row.get("node_id", "Unknown")
                road_name = row.get("road_name", "Unknown")
                lat = row.get("lat")
                lon = row.get("lon")
                
                if not lat or not lon:
                    continue
                
                if current_key_index >= len(API_KEYS):
                    break
                
                result = fetch_data_for_point(road_name, lat, lon)
                
                if result:
                    result["node_id"] = node_id
                    writer.writerow(result)
                    
                    print(
                        f"Node {node_id} | {road_name}: "
                        f"{result['current_avg_speed_kmh']} km/h "
                        f"(free flow: {result['free_flow_speed_kmh']} km/h)"
                    )
                
                time.sleep(0.5)

    print(f"\nData saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    collect_once()