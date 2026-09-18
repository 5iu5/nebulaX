"""Shared Streamlit entry point for the attempted train-monitoring subsystems."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from acv.app.page import render as render_acv
from door.app import render as render_door
from rail.app import render as render_rail
from shm_v2.app import render as render_shm

st.set_page_config(page_title="NebulaX | Train condition monitoring", page_icon="🚆", layout="wide")
st.markdown(
    """<style>
.stApp {background:#f5f7fa;} h1,h2,h3 {color:#142d41;}
[data-testid="stMetric"] {background:white;border:1px solid #dbe3eb;border-radius:12px;padding:16px;}
.stButton > button {border-radius:8px;}
</style>""",
    unsafe_allow_html=True,
)

PAGES = {
    "ACV · Air conditioning": render_acv,
    "Door · Abnormal resistance": render_door,
    "Rail · Corrugation": render_rail,
    "SHM · Fatigue damage": render_shm,
}

st.caption("NEBULAX  ·  TRAIN CONDITION MONITORING")
with st.sidebar:
    st.header("Analysis")
    subsystem = st.selectbox("Subsystem", list(PAGES))
    st.divider()
    st.write("**How to use**")
    st.write("1. Select a subsystem.\n2. Upload its telemetry file.\n3. Run the analysis.\n4. Review and download the results.")
    st.caption("The file is processed locally by this app.")

PAGES[subsystem]()
