import requests
import pandas as pd
import time
import os

def generate_hcmc_nodes():
    # ==========================================
    # STEP 1: FETCH NODES (WITH AUTO-RETRY)
    # ==========================================
    print("Fetching traffic signal intersections from OpenStreetMap...")
    
    query = """
    [out:json][timeout:60];
    (
      node["highway"="traffic_signals"](10.75, 106.65, 10.83, 106.75);
    );
    out body 300;
    """
    
    headers = {'User-Agent': 'TDTU-Traffic-Research-Project/1.0'}
    
    # 2 server để dự phòng cho nhau
    urls = [
        "http://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter" 
    ]
    
    raw_data = None
    # Cho phép script thử lại tối đa 5 lần
    for attempt in range(5):
        for url in urls:
            print(f"Connecting to {url} ...")
            try:
                response = requests.post(url, data={'data': query}, headers=headers, timeout=60)
                if response.status_code == 200:
                    raw_data = response.json()
                    break
                else:
                    print(f"Server busy (Status: {response.status_code}).")
            except Exception as e:
                print(f"Error: {e}")
        
        if raw_data:
            break
            
        print("Retrying in 5 seconds...\n")
        time.sleep(5)
        
    if not raw_data:
        print("❌ All servers are overloaded right now. Please try again in 10 minutes.")
        return

    nodes = []
    for idx, element in enumerate(raw_data.get('elements', [])):
        nodes.append({
            'node_id': idx + 1,
            'road_name': f"Intersection_{idx + 1}",
            'lat': element['lat'],
            'lon': element['lon']
        })

    if not nodes:
        print("No nodes found.")
        return
        
    print(f"✅ Successfully fetched {len(nodes)} nodes.\n")

    # ==========================================
    # STEP 2: REVERSE GEOCODING (GET ROAD NAMES)
    # ==========================================
    print("Fetching real road names... (This will take about 5-6 minutes)")
    
    headers_nominatim = {
        'User-Agent': 'TDTU-Traffic-Research-Project/1.0'
    }
    
    for node in nodes:
        lat = node['lat']
        lon = node['lon']
        geocode_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=16"
        
        try:
            res = requests.get(geocode_url, headers=headers_nominatim, timeout=10)
            if res.status_code == 200:
                geo_data = res.json()
                address = geo_data.get('address', {})
                real_road_name = address.get('road') or address.get('pedestrian') or address.get('suburb')
                
                if real_road_name:
                    node['road_name'] = real_road_name
                    print(f"Node {node['node_id']} mapped to: {real_road_name}")
                else:
                    print(f"Node {node['node_id']}: Keeping placeholder.")
            else:
                print(f"API Error at Node {node['node_id']}: {res.status_code}")
        except Exception as e:
            print(f"Error at Node {node['node_id']}: {e}")
            
        # Delay 1.2s theo luật chống spam của server
        time.sleep(1.2)

    # ==========================================
    # STEP 3: SAVE TO CSV
    # ==========================================
    os.makedirs("data", exist_ok=True)
    output_file = "data/hcmc_nodes.csv"
    
    df = pd.DataFrame(nodes)
    df = df[['node_id', 'road_name', 'lat', 'lon']]
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"Done! Data saved to {output_file}")

if __name__ == "__main__":
    generate_hcmc_nodes()