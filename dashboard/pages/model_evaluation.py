import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.layout import render_data_status
from dashboard.utils.data import load_data
from dashboard.utils.evaluation import load_evaluation_results, load_horizon_results


st.title("Đánh giá mô hình")
st.caption("Chỉ hiển thị metric khi có kết quả đánh giá thực tế trong project.")
bundle = load_data()
render_data_status(bundle)

metrics = load_evaluation_results()
if metrics.empty:
    st.warning("Chưa tìm thấy file metric đánh giá thực tế trong thư mục outputs.")
else:
    display = metrics.drop(columns=["Seed", "File"], errors="ignore")
    st.dataframe(display, width="stretch", hide_index=True)

    tabs = st.tabs(["MAE / RMSE", "MAPE", "Theo horizon"])
    with tabs[0]:
        fig = go.Figure()
        for column, label, color in (
            ("MAE", "MAE", "#2563eb"),
            ("RMSE", "RMSE", "#f59e0b"),
        ):
            fig.add_bar(
                x=metrics["Mô hình"],
                y=metrics[column],
                name=label,
                marker_color=color,
            )
        fig.update_layout(
            barmode="group",
            height=440,
            template="plotly_white",
            yaxis_title="Sai số",
            xaxis_title="Mô hình",
        )
        st.plotly_chart(fig, width="stretch", key="model_overall_mae_rmse")
    with tabs[1]:
        fig = go.Figure(
            go.Bar(
                x=metrics["Mô hình"],
                y=metrics["MAPE (%)"],
                marker_color="#ef4444",
                text=metrics["MAPE (%)"].round(2),
                textposition="outside",
            )
        )
        fig.update_layout(
            height=440,
            template="plotly_white",
            yaxis_title="MAPE (%)",
            xaxis_title="Mô hình",
        )
        st.plotly_chart(fig, width="stretch", key="model_overall_mape")
    with tabs[2]:
        horizon = load_horizon_results()
        if horizon.empty:
            st.info("Không có metric theo horizon.")
        else:
            horizon_metric = st.selectbox("Metric horizon", ["MAE", "RMSE", "MAPE (%)"])
            fig = go.Figure()
            for model_name in horizon["Mô hình"].unique():
                subset = horizon.loc[horizon["Mô hình"] == model_name]
                fig.add_scatter(
                    x=subset["Cửa sổ"],
                    y=subset[horizon_metric],
                    mode="lines+markers",
                    name=model_name,
                )
            fig.update_layout(
                height=440,
                template="plotly_white",
                yaxis_title=horizon_metric,
                xaxis_title="Cửa sổ dự báo",
                legend_title_text="Mô hình",
            )
            st.plotly_chart(fig, width="stretch", key="model_horizon_chart")
