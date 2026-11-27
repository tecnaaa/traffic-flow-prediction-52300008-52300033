import pandas as pd
import streamlit as st

from dashboard.components.layout import render_data_status, render_filters
from dashboard.components.charts import show_timeseries
from dashboard.utils.data import latest_readings, load_data


st.title("Tổng quan giao thông TP.HCM")
st.caption("Tổng hợp từ dữ liệu thu thập trong project; không áp dụng dữ liệu minh họa.")
bundle = load_data()
render_data_status(bundle)
filtered = render_filters(bundle.traffic, "overview")
snapshot = latest_readings(filtered)

total_nodes = len(bundle.nodes)
mean_speed = snapshot["current_avg_speed_kmh"].mean() if not snapshot.empty else None
congestion_points = (
    int(snapshot["congestion_ratio_0_to_1"].ge(0.5).sum())
    if not snapshot.empty
    else None
)
last_timestamp = bundle.traffic["timestamp"].max() if not bundle.traffic.empty else pd.NaT

kpis = st.columns(4)
kpis[0].metric("Tổng số nút có tọa độ", f"{total_nodes:,}")
kpis[1].metric("Tốc độ trung bình mới nhất", f"{mean_speed:.1f} km/h" if pd.notna(mean_speed) else "Chưa có")
kpis[2].metric("Điểm ùn tắc (ratio ≥ 0,5)", f"{congestion_points:,}" if congestion_points is not None else "Chưa có")
kpis[3].metric("Số bản ghi", f"{len(bundle.traffic):,}")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Diễn biến ùn tắc")
    show_timeseries(
        filtered,
        "overview_congestion",
        "congestion_ratio_0_to_1",
        "Congestion ratio theo thời gian",
        "Congestion ratio",
        max_series=6,
    )
with right:
    st.subheader("Trạng thái dữ liệu")
    if bundle.errors:
        st.error("Có lỗi schema hoặc file dữ liệu. Vui lòng kiểm tra thông báo phía trên.")
    elif bundle.traffic.empty:
        st.warning("Đã nhận diện file nhưng chưa có bản ghi lưu lượng.")
    else:
        st.success("Schema nhận diện được và có bản ghi lưu lượng.")
    st.write("**Bản ghi trong bộ lọc:**", f"{len(filtered):,}")
    st.write("**Nút có dữ liệu trong bộ lọc:**", f"{snapshot['node_id'].nunique():,}")
    st.write("**Thời điểm ghi nhận mới nhất:**", "—" if pd.isna(last_timestamp) else last_timestamp.strftime("%d/%m/%Y %H:%M:%S"))
    if not snapshot.empty:
        st.write("**Độ tin cậy trung bình:**", f"{snapshot['confidence_score_0_to_1'].mean():.1%}")

with st.expander("Xem snapshot bản ghi mới nhất theo nút"):
    if snapshot.empty:
        st.info("Chưa có snapshot để hiển thị.")
    else:
        st.dataframe(
            snapshot.sort_values("congestion_ratio_0_to_1", ascending=False),
            width="stretch",
            hide_index=True,
        )
