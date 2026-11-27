import streamlit as st

from dashboard.components.charts import show_empty_prediction_chart
from dashboard.components.layout import render_data_status
from dashboard.utils.data import load_data


st.title("Dự báo lưu lượng")
st.caption("Trang này chỉ hiển thị dự báo khi có đầu ra mô hình thật được tích hợp.")
bundle = load_data()
render_data_status(bundle)

node_ids = sorted(bundle.traffic["node_id"].dropna().astype(str).unique().tolist())
if not node_ids:
    st.info("Chưa có node_id trong dữ liệu để chọn.")
    st.stop()

controls = st.columns(3)
node_id = controls[0].selectbox("Nút giao", node_ids)
model = controls[1].selectbox(
    "Mô hình",
    ["STGCN", "AGCRN", "STAEformer", "HGMWN gốc", "HGMWN đề xuất"],
)
horizon = controls[2].selectbox("Thời gian dự báo", [15, 30, 60], format_func=lambda value: f"{value} phút")

st.warning(
    f"Chưa có artifact hoặc dữ liệu dự báo cho **{model}**. "
    "Biểu đồ bên dưới chỉ vẽ quan sát thực tế; không sinh dự báo giả."
)
st.caption(f"Khoảng dự báo được chọn: {horizon} phút · Đầu ra dự báo: chưa tích hợp.")
show_empty_prediction_chart(bundle.traffic, node_id)
