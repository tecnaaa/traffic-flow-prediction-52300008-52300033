import sys
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from dashboard.components.layout import apply_styles


st.set_page_config(
    page_title="Traffic Flow · TP.HCM",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_styles()
st.sidebar.title("Traffic Flow · TP.HCM")
st.sidebar.caption("Dashboard dữ liệu giao thông")

pages = [
    st.Page("pages/overview.py", title="Overview", icon="📊", default=True),
    st.Page("pages/traffic_map.py", title="Traffic Map", icon="🗺️"),
    st.Page("pages/prediction.py", title="Traffic Prediction", icon="📈"),
    st.Page("pages/hypergraph.py", title="Hypergraph", icon="🕸️"),
    st.Page("pages/model_evaluation.py", title="Model Evaluation", icon="🧪"),
    st.Page("pages/data_analytics.py", title="Data Analytics", icon="🔎"),
]
navigation = st.navigation(pages)
navigation.run()
