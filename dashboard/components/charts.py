from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BLUE = "#2563eb"
LIGHT_BLUE = "#dbeafe"


def show_correlation_heatmap(correlation: pd.DataFrame, key: str) -> None:
    if correlation.empty:
        st.info("Chưa đủ dữ liệu để tính ma trận tương quan.")
        return
    values = correlation.apply(pd.to_numeric, errors="coerce")
    shared = values.index.intersection(values.columns)
    values = values.loc[shared, shared]
    if values.empty:
        st.info("Ma trận tương quan không có cột hàng/cột tương ứng.")
        return
    if len(values) > 20:
        candidates = values.var(axis=1).nlargest(20).index
        values = values.loc[candidates, candidates]
        st.caption("Hiển thị tối đa 20 tuyến có độ biến thiên tương quan lớn nhất để dễ đọc.")
    fig = px.imshow(
        values,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        origin="lower",
        aspect="auto",
        labels={"x": "Tuyến đường", "y": "Tuyến đường", "color": "Pearson r"},
    )
    fig.update_layout(height=max(480, min(850, 24 * len(values))), margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, width="stretch", key=key)


def show_timeseries(
    traffic: pd.DataFrame,
    key: str,
    value_column: str,
    title: str,
    y_label: str,
    max_series: int = 8,
) -> None:
    if traffic.empty or value_column not in traffic:
        st.info("Không có dữ liệu phù hợp cho chuỗi thời gian.")
        return
    available = traffic.dropna(subset=["timestamp", value_column])
    if available.empty:
        st.info("Không có giá trị số hợp lệ để vẽ.")
        return
    roads = (
        available.groupby("road_name")[value_column]
        .var()
        .nlargest(max_series)
        .index.astype(str)
        .tolist()
    )
    subset = available.loc[available["road_name"].astype(str).isin(roads)]
    fig = px.line(
        subset,
        x="timestamp",
        y=value_column,
        color="road_name",
        markers=True,
        title=title,
        labels={"timestamp": "Thời gian", value_column: y_label, "road_name": "Tuyến đường"},
        hover_data=["node_id"],
    )
    fig.update_layout(legend_title_text="Tuyến đường", height=440, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, width="stretch", key=key)


def show_empty_prediction_chart(traffic: pd.DataFrame, node_id: str | None) -> None:
    fig = go.Figure()
    if not traffic.empty and node_id is not None:
        actual = traffic.loc[traffic["node_id"].astype(str) == str(node_id)].dropna(
            subset=["timestamp", "current_avg_speed_kmh"]
        )
        if not actual.empty:
            fig.add_trace(
                go.Scatter(
                    x=actual["timestamp"],
                    y=actual["current_avg_speed_kmh"],
                    mode="lines+markers",
                    name="Thực tế",
                    line={"color": BLUE, "width": 2},
                )
            )
    fig.update_layout(
        title="Tốc độ thực tế và dự báo",
        xaxis_title="Thời gian",
        yaxis_title="Tốc độ (km/h)",
        height=440,
        margin=dict(l=20, r=20, t=55, b=20),
        template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch", key="prediction_actual_chart")
