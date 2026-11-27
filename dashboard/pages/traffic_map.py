import pandas as pd
import pydeck as pdk
import streamlit as st

from dashboard.components.layout import render_data_status
from dashboard.utils.data import latest_readings, load_data


st.title("Bản đồ giao thông")
st.caption("Vị trí lấy trực tiếp từ data/hcmc_nodes.csv; màu xám biểu thị nút chưa có số đo.")
bundle = load_data()
render_data_status(bundle)

if bundle.nodes.empty:
    st.info("Chưa có tọa độ nút hợp lệ để dựng bản đồ.")
    st.stop()

nodes = bundle.nodes.copy()
readings = latest_readings(bundle.traffic)
if not readings.empty:
    readings = readings.drop(columns=["road_name"], errors="ignore")
    nodes = nodes.merge(readings, on="node_id", how="left")
else:
    nodes["congestion_ratio_0_to_1"] = pd.NA
    nodes["current_avg_speed_kmh"] = pd.NA
    nodes["timestamp"] = pd.NaT

max_nodes = min(300, len(nodes))
min_nodes = min(200, max_nodes)
node_count = st.sidebar.slider(
    "Số nút hiển thị",
    min_value=min_nodes,
    max_value=max_nodes,
    value=max_nodes,
    key="map_node_count",
)
road_options = sorted(nodes["road_name"].dropna().astype(str).unique().tolist())
selected_roads = st.sidebar.multiselect(
    "Lọc tuyến đường",
    options=road_options,
    default=road_options,
    key="map_roads",
)
nodes = nodes.loc[nodes["road_name"].astype(str).isin(selected_roads)].head(node_count).copy()

def congestion_color(value: object) -> list[int]:
    if pd.isna(value):
        return [148, 163, 184, 210]
    score = float(value)
    if score < 0.3:
        return [34, 197, 94, 220]
    if score < 0.6:
        return [245, 158, 11, 230]
    return [239, 68, 68, 240]


nodes["color"] = nodes["congestion_ratio_0_to_1"].map(congestion_color)
nodes["node_id"] = nodes["node_id"].astype(str)
nodes["congestion_display"] = nodes["congestion_ratio_0_to_1"].map(
    lambda value: "Chưa có" if pd.isna(value) else f"{float(value):.2f}"
)
nodes["speed_display"] = nodes["current_avg_speed_kmh"].map(
    lambda value: "Chưa có" if pd.isna(value) else f"{float(value):.1f} km/h"
)

if nodes.empty:
    st.info("Không có nút phù hợp bộ lọc.")
    st.stop()

layer = pdk.Layer(
    "ScatterplotLayer",
    data=nodes,
    get_position="[lon, lat]",
    get_fill_color="color",
    get_radius=90,
    pickable=True,
    auto_highlight=True,
    radius_min_pixels=5,
    radius_max_pixels=16,
)
view = pdk.ViewState(
    latitude=float(nodes["lat"].mean()),
    longitude=float(nodes["lon"].mean()),
    zoom=11,
    pitch=0,
)
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view,
    map_style="light",
    tooltip={
        "html": (
            "<b>{road_name}</b><br/>Node: {node_id}<br/>"
            "Congestion ratio: {congestion_display}<br/>Speed: {speed_display}"
        ),
        "style": {"backgroundColor": "#ffffff", "color": "#0f172a"},
    },
)
selection = st.pydeck_chart(
    deck,
    width="stretch",
    on_select="rerun",
    selection_mode="single-object",
    key="traffic_nodes_map",
)

selected_id = None
if isinstance(selection, dict):
    selected = selection.get("selection", {})
    if isinstance(selected, dict):
        objects = selected.get("objects", {})
        if isinstance(objects, dict):
            for layer_objects in objects.values():
                if layer_objects and isinstance(layer_objects[0], dict):
                    selected_id = str(layer_objects[0].get("node_id", ""))
                    break

all_ids = nodes["node_id"].tolist()
default_index = all_ids.index(selected_id) if selected_id in all_ids else 0
chosen_id = st.selectbox(
    "Nút cần xem chi tiết (có thể chọn trực tiếp trên bản đồ)",
    options=all_ids,
    index=default_index,
    format_func=lambda value: (
        f"{nodes.loc[nodes['node_id'] == value, 'road_name'].iloc[0]} · node {value}"
    ),
)
detail = nodes.loc[nodes["node_id"] == chosen_id].iloc[0]
columns = st.columns(4)
columns[0].metric("Tuyến đường", str(detail["road_name"]))
columns[1].metric("Congestion ratio", str(detail["congestion_display"]))
columns[2].metric("Tốc độ", str(detail["speed_display"]))
columns[3].metric("Tọa độ", f"{detail['lat']:.6f}, {detail['lon']:.6f}")
st.caption("Xanh: < 0,30 · Vàng: 0,30–<0,60 · Đỏ: ≥ 0,60 · Xám: chưa có số đo.")
