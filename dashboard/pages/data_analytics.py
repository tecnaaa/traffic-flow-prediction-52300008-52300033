import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.charts import show_correlation_heatmap, show_timeseries
from dashboard.components.layout import render_data_status, render_filters
from dashboard.utils.data import calculate_correlation, load_data


st.title("Phân tích dữ liệu")
bundle = load_data()
render_data_status(bundle)
traffic = render_filters(bundle.traffic, "analytics")
tabs = st.tabs(
    ["Heatmap & tương quan", "Time series", "Missing values", "Outlier", "Confidence score"]
)

with tabs[0]:
    matrix = bundle.correlation
    source = "outputs/correlation_matrix.csv"
    if matrix is None:
        matrix = calculate_correlation(traffic)
        source = "tính từ bản ghi đã lọc (cửa sổ 15 phút)"
    st.caption(f"Nguồn: {source}")
    show_correlation_heatmap(matrix if matrix is not None else pd.DataFrame(), "analytics_correlation")
with tabs[1]:
    value = st.selectbox(
        "Chỉ số",
        ["congestion_ratio_0_to_1", "current_avg_speed_kmh", "confidence_score_0_to_1"],
        format_func={
            "congestion_ratio_0_to_1": "Congestion ratio",
            "current_avg_speed_kmh": "Tốc độ hiện tại (km/h)",
            "confidence_score_0_to_1": "Confidence score",
        }.get,
    )
    show_timeseries(traffic, "analytics_timeseries", value, "Diễn biến theo thời gian", value)
with tabs[2]:
    if traffic.empty:
        st.info("Không có bản ghi trong bộ lọc.")
    else:
        missing = traffic.isna().sum().rename("Giá trị thiếu").to_frame()
        missing["Tỷ lệ thiếu (%)"] = (missing["Giá trị thiếu"] / len(traffic) * 100).round(2)
        st.dataframe(missing, width="stretch")
        fig = px.bar(
            missing.reset_index(),
            x="index",
            y="Tỷ lệ thiếu (%)",
            labels={"index": "Cột", "Tỷ lệ thiếu (%)": "Tỷ lệ thiếu (%)"},
            title="Tỷ lệ giá trị thiếu theo cột",
        )
        st.plotly_chart(fig, width="stretch", key="missing_values_chart")
with tabs[3]:
    numeric = traffic.select_dtypes(include="number")
    if numeric.empty:
        st.info("Không có cột số để phân tích outlier.")
    else:
        rows = []
        for column in numeric:
            values = numeric[column].dropna()
            if values.empty:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
            rows.append({"Cột": column, "Số outlier (IQR)": count, "Số giá trị": len(values)})
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("Outlier được xác định theo ngưỡng 1.5×IQR trên dữ liệu đang lọc, không loại bỏ bản ghi.")
        else:
            st.info("Không có giá trị hợp lệ để phân tích.")
with tabs[4]:
    confidence = traffic["confidence_score_0_to_1"].dropna() if "confidence_score_0_to_1" in traffic else pd.Series(dtype=float)
    if confidence.empty:
        st.info("Không có confidence score trong dữ liệu đang lọc.")
    else:
        columns = st.columns(3)
        columns[0].metric("Trung bình", f"{confidence.mean():.3f}")
        columns[1].metric("Trung vị", f"{confidence.median():.3f}")
        columns[2].metric("Số phép đo", f"{len(confidence):,}")
        fig = px.histogram(
            x=confidence,
            nbins=20,
            labels={"x": "Confidence score"},
            title="Phân phối confidence score thực tế",
        )
        st.plotly_chart(fig, width="stretch", key="confidence_histogram")
