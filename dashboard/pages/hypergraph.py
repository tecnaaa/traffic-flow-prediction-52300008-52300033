import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from dashboard.components.charts import show_correlation_heatmap
from dashboard.components.layout import render_data_status
from dashboard.utils.data import calculate_correlation, load_data
from dashboard.config import OUTPUTS_DIR


st.title("Phân tích Hypergraph")
st.caption("Các cấu trúc không có trong dữ liệu đầu vào sẽ được đánh dấu chờ tích hợp.")
bundle = load_data()
render_data_status(bundle)

physical, dynamic, pearson, sliding = st.tabs(
    ["Physical Hyperedge", "Dynamic Hyperedge", "Pearson Correlation", "Sliding Window"]
)
with physical:
    adjacency_path = OUTPUTS_DIR / "adj_matrix.npy"
    if adjacency_path.exists():
        adjacency = np.load(adjacency_path)
        fig = px.imshow(
            adjacency,
            color_continuous_scale="Blues",
            aspect="auto",
            origin="lower",
            labels={"x": "Nút", "y": "Nút", "color": "Trọng số"},
            title="Ma trận kề vật lý/dữ liệu HCMC",
        )
        fig.update_layout(template="plotly_white", height=520)
        st.plotly_chart(fig, width="stretch", key="physical_hyperedge_matrix")
        st.caption("Nguồn: outputs/adj_matrix.npy")
    else:
        st.info("Chưa có outputs/adj_matrix.npy để hiển thị topology.")
with dynamic:
    tau = st.select_slider(
        "Ngưỡng tương quan tau",
        options=["0.50", "0.60", "0.70", "0.80"],
        value="0.70",
    )
    incidence_path = OUTPUTS_DIR / f"hyperedge_incidence_tau{tau}.npy"
    operator_path = OUTPUTS_DIR / f"hyperedge_operator_tau{tau}.npy"
    if incidence_path.exists() and operator_path.exists():
        incidence = np.load(incidence_path)
        operator = np.load(operator_path)
        col1, col2 = st.columns([1.2, 1])
        with col1:
            if incidence.shape[1] > 40:
                shown = incidence[:, :40]
                st.caption("Hiển thị 40 hyperedge đầu tiên để giữ biểu đồ dễ đọc.")
            else:
                shown = incidence
            fig = px.imshow(
                shown,
                color_continuous_scale="Viridis",
                aspect="auto",
                origin="lower",
                labels={"x": "Hyperedge", "y": "Nút", "color": "Membership"},
                title=f"Incidence matrix · tau {tau}",
            )
            fig.update_layout(template="plotly_white", height=520)
            st.plotly_chart(fig, width="stretch", key="dynamic_hyperedge_incidence")
        with col2:
            display_max = float(np.nanquantile(operator, 0.99))
            if display_max <= 0:
                display_max = float(np.nanmax(operator))
            fig = px.imshow(
                operator,
                color_continuous_scale="Blues",
                aspect="auto",
                origin="lower",
                zmin=0,
                zmax=display_max,
                labels={"x": "Nút", "y": "Nút", "color": "Trọng số (màu giới hạn p99)"},
                title=f"Hyperedge operator · tau {tau}",
            )
            fig.update_layout(template="plotly_white", height=520)
            st.plotly_chart(fig, width="stretch", key="dynamic_hyperedge_operator")
            st.caption(
                f"Thang màu giới hạn ở percentile 99 ({display_max:.4g}); "
                "hover vẫn hiển thị trọng số gốc."
            )
    else:
        st.info("Chưa có incidence/operator matrix HCMC tương ứng cho ngưỡng đã chọn.")
with pearson:
    matrix = bundle.correlation
    source = "outputs/correlation_matrix.csv"
    if matrix is None:
        matrix = calculate_correlation(bundle.traffic)
        source = "tính trực tiếp từ traffic_data.csv (cửa sổ 15 phút)"
    if matrix is not None and not matrix.empty:
        st.caption(f"Nguồn ma trận: {source}")
        show_correlation_heatmap(matrix, "hypergraph_pearson")
    else:
        st.info("Chưa đủ dữ liệu theo thời gian để lập ma trận Pearson.")
with sliding:
    traffic = bundle.traffic.dropna(subset=["timestamp", "congestion_ratio_0_to_1"])
    if traffic.empty:
        st.info("Chưa có số đo ùn tắc hợp lệ để hiển thị các cửa sổ quan sát.")
    else:
        traffic = traffic.copy()
        traffic["window_start"] = traffic["timestamp"].dt.floor("15min")
        windows = (
            traffic.groupby("window_start", as_index=False)
            .agg(
                mean_congestion=("congestion_ratio_0_to_1", "mean"),
                observed_records=("node_id", "size"),
                observed_nodes=("node_id", "nunique"),
            )
        )
        fig = px.line(
            windows,
            x="window_start",
            y="mean_congestion",
            markers=True,
            hover_data=["observed_records", "observed_nodes"],
            labels={
                "window_start": "Bắt đầu cửa sổ",
                "mean_congestion": "Congestion ratio trung bình",
                "observed_records": "Số bản ghi",
                "observed_nodes": "Số nút quan sát",
            },
            title="Các cửa sổ quan sát 15 phút (dữ liệu thực tế)",
        )
        fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig, width="stretch", key="sliding_window_chart")
        st.caption("Đây là tổng hợp số đo theo cửa sổ thời gian, không phải hyperedge được suy diễn.")
