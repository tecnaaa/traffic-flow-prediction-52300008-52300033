from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from dashboard.utils.data import DataBundle


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.8rem; padding-bottom: 3rem; }
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e6edf5;
            padding: 1rem 1.1rem; border-radius: 14px;
            box-shadow: 0 3px 14px rgba(30, 64, 120, .05);
        }
        [data-testid="stSidebar"] { border-right: 1px solid #e9eef5; }
        div[data-testid="stAlert"] { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_data_status(bundle: DataBundle) -> None:
    for message in bundle.errors:
        st.error(message)
    for message in bundle.warnings:
        st.warning(message)
    if not bundle.errors:
        st.success(
            f"Dữ liệu hợp lệ · {len(bundle.nodes):,} nút có tọa độ · "
            f"{len(bundle.traffic):,} bản ghi lưu lượng"
        )


def render_filters(
    traffic: pd.DataFrame,
    key: str,
    include_roads: bool = True,
) -> pd.DataFrame:
    if traffic.empty:
        return traffic.copy()
    filtered = traffic.copy()
    timestamps = filtered["timestamp"].dropna()
    if not timestamps.empty:
        min_day = timestamps.min().date()
        max_day = timestamps.max().date()
        selected = st.sidebar.date_input(
            "Khoảng ngày",
            value=(min_day, max_day),
            min_value=min_day,
            max_value=max_day,
            key=f"{key}_dates",
        )
        if isinstance(selected, tuple) and len(selected) == 2:
            start, end = selected
            filtered = filtered.loc[
                filtered["timestamp"].dt.date.between(start, end)
            ]
        elif isinstance(selected, date):
            filtered = filtered.loc[filtered["timestamp"].dt.date == selected]
    if include_roads and "road_name" in filtered and not filtered.empty:
        roads = sorted(filtered["road_name"].dropna().astype(str).unique().tolist())
        selected_roads = st.sidebar.multiselect(
            "Tuyến đường",
            options=roads,
            default=roads,
            key=f"{key}_roads",
        )
        filtered = filtered.loc[filtered["road_name"].astype(str).isin(selected_roads)]
    return filtered


def render_empty_state(message: str) -> None:
    st.info(message)
