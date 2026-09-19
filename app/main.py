"""Shared Streamlit entry point for the attempted train-monitoring subsystems."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from app.ui import workflow_steps
from acv.app.page import render as render_acv
from door.app import render as render_door
from rail.app import render as render_rail
from shm.app import render as render_shm

st.set_page_config(page_title="NebulaX | Train condition monitoring", page_icon="🚆", layout="wide")
st.markdown(
    """<style>
:root {
  --ink:#102a3c; --muted:#587083; --line:#dbe5ec; --surface:#ffffff;
  --canvas:#f4f7f9; --teal:#167d9a; --teal-dark:#0f6178; --teal-soft:#e7f3f6;
  --normal:#19715f; --normal-bg:#e8f5f0; --advisory:#8a5a00; --advisory-bg:#fff4d6;
  --high:#a63d40; --high-bg:#fdebec; --critical:#8f1d2c; --critical-bg:#f9dfe3;
  --shadow:0 10px 28px rgba(16,42,60,.07);
}
html, body, [class*="css"] {font-family:Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
.stApp {background:linear-gradient(180deg,#f8fafb 0,#f3f7f9 36rem,#f4f7f9 100%);color:var(--ink);}
[data-testid="stAppViewContainer"] > .main .block-container {max-width:1480px;padding-top:1.35rem;padding-bottom:4rem;}
p, li, label {line-height:1.55;}
h1,h2,h3 {color:var(--ink);letter-spacing:-.025em;}
h2 {font-size:1.35rem!important;margin-top:1.7rem!important;}
h3 {font-size:1.05rem!important;}
[data-testid="stHeader"] {background:transparent;}
/* Hide the individual chrome controls, NOT the whole toolbar: the sidebar expand
   button lives inside stToolbar, so hiding the container strands a collapsed sidebar. */
#MainMenu, footer, [data-testid="stMainMenuButton"],
[data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] {display:none!important;}
[data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] button {display:flex!important;visibility:visible!important;}
/* Swap Streamlit's double-chevron for a panel-toggle icon (Material Symbols ligature). */
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {font-size:0!important;line-height:1;}
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]:before {content:"left_panel_close";font-size:1.3rem;}
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]:before {content:"left_panel_open";font-size:1.3rem;}

.app-brandbar {display:flex;align-items:center;justify-content:space-between;margin:0 0 1rem;padding:.15rem .1rem;}
.brand-lockup {display:flex;align-items:center;gap:.65rem;color:var(--ink);font-weight:800;letter-spacing:.08em;font-size:.78rem;}
.brand-mark {display:grid;place-items:center;width:30px;height:30px;border-radius:9px;background:var(--ink);color:white;font-size:.9rem;box-shadow:0 5px 12px rgba(16,42,60,.18);}
.app-context {color:var(--muted);font-size:.78rem;font-weight:600;}

.page-hero {position:relative;overflow:hidden;background:linear-gradient(125deg,#fff 0%,#f8fcfd 68%,#e9f5f7 100%);border:1px solid var(--line);border-radius:20px;padding:1.8rem 2rem 1.65rem;margin-bottom:1.4rem;box-shadow:var(--shadow);}
.page-hero:after {content:"";position:absolute;width:210px;height:210px;border-radius:50%;right:-100px;top:-130px;background:rgba(22,125,154,.10);}
.page-eyebrow {color:var(--teal-dark);font-size:.72rem;line-height:1;font-weight:800;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.65rem;}
.page-hero h1 {font-size:clamp(1.75rem,3vw,2.45rem);line-height:1.08;margin:0 0 .65rem;max-width:820px;}
.page-hero p {color:var(--muted);font-size:1.02rem;line-height:1.55;max-width:850px;margin:0;}
.hero-chips {display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1.05rem;}
.hero-chip {display:inline-flex;padding:.3rem .65rem;border-radius:999px;background:#fff;border:1px solid #cfe0e8;color:#365469;font-size:.72rem;font-weight:700;}

.section-intro {margin:1.8rem 0 .75rem;}
.section-intro h2 {font-size:1.35rem!important;line-height:1.2;margin:0!important;}
.section-intro p {color:var(--muted);margin:.32rem 0 0;max-width:760px;}
.status-badge {display:inline-flex;align-items:center;gap:.45rem;border-radius:999px;padding:.32rem .7rem;font-size:.72rem;font-weight:800;letter-spacing:.025em;margin-bottom:.45rem;border:1px solid transparent;}
.status-dot {width:.48rem;height:.48rem;border-radius:50%;background:currentColor;}
.status-normal {color:var(--normal);background:var(--normal-bg);border-color:#bce2d5;}
.status-advisory {color:var(--advisory);background:var(--advisory-bg);border-color:#eed99b;}
.status-high {color:var(--high);background:var(--high-bg);border-color:#f0c3c5;}
.status-critical {color:var(--critical);background:var(--critical-bg);border-color:#eab1ba;}

[data-testid="stSidebar"] {background:#fff;border-right:1px solid var(--line);}
[data-testid="stSidebar"] > div:first-child {padding-top:1.2rem;}
.sidebar-brand {padding:.85rem .9rem;background:var(--ink);color:#fff;border-radius:14px;margin-bottom:1.1rem;box-shadow:0 8px 20px rgba(16,42,60,.16);}
.sidebar-brand strong {display:block;font-size:.9rem;letter-spacing:.08em;}
.sidebar-brand span {display:block;color:#c7d7e1;font-size:.72rem;margin-top:.2rem;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.3rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {width:100%;margin:0;padding:.5rem .65rem;border:1px solid var(--line);border-radius:10px;background:#fff;cursor:pointer;transition:border-color .12s ease,background .12s ease;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {border-color:#9fc0cd;background:#f7fbfc;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {border-color:var(--teal);background:var(--teal-soft);}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-size:.82rem;font-weight:700;color:var(--ink);}

.workflow {counter-reset:step;display:grid;gap:.62rem;margin:.7rem 0 .9rem;}
.workflow div {display:flex;align-items:center;gap:.6rem;color:#496276;font-size:.79rem;}
.workflow div:before {counter-increment:step;content:counter(step);display:grid;place-items:center;flex:0 0 1.45rem;height:1.45rem;border-radius:50%;background:var(--teal-soft);color:var(--teal-dark);font-weight:800;font-size:.68rem;}
.privacy-note {padding:.7rem .8rem;border-radius:10px;background:#f3f7f9;color:#587083;font-size:.72rem;border:1px solid #e2eaf0;}

.ranking-board {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem;margin:.25rem 0 1.1rem;}
.ranking-card {position:relative;background:#fff;border:1px solid var(--line);border-radius:14px;padding:.85rem .9rem;min-height:116px;box-shadow:0 4px 14px rgba(16,42,60,.045);transition:transform .12s ease,box-shadow .12s ease,border-color .12s ease;}
.ranking-card:hover {transform:translateY(-2px);box-shadow:0 8px 20px rgba(16,42,60,.09);border-color:#b7ccd7;}
.ranking-card-top {display:flex;align-items:center;justify-content:space-between;gap:.5rem;}
.ranking-number {display:grid;place-items:center;width:1.65rem;height:1.65rem;border-radius:50%;background:#eaf0f4;color:#496276;font-size:.72rem;font-weight:850;}
.ranking-priority {color:#708595;font-size:.65rem;font-weight:750;text-transform:uppercase;letter-spacing:.055em;text-align:right;}
.ranking-car {color:var(--ink);font-size:1.12rem;font-weight:820;letter-spacing:-.02em;margin-top:.7rem;}
.ranking-evidence {color:var(--muted);font-size:.75rem;margin-top:.2rem;}
.ranking-first {border:1.5px solid #d57a7d;background:linear-gradient(145deg,#fff 0%,#fff5f5 100%);box-shadow:0 6px 18px rgba(166,61,64,.10);}
.ranking-first .ranking-number {background:var(--high);color:#fff;}
.ranking-first .ranking-priority {color:var(--high);}
.ranking-second {border-color:#e6c778;background:linear-gradient(145deg,#fff 0%,#fffbef 100%);}
.ranking-second .ranking-number {background:#b17a13;color:#fff;}
.ranking-second .ranking-priority {color:#8a5a00;}
.ranking-unavailable {background:#f4f6f7;border-style:dashed;box-shadow:none;opacity:.76;}
.ranking-unavailable .ranking-evidence {color:#7d5660;}

[data-testid="stMetric"] {background:#fff;border:1px solid var(--line);border-radius:14px;padding:1rem 1.05rem;box-shadow:0 4px 14px rgba(16,42,60,.045);min-height:104px;}
[data-testid="stMetricLabel"] {color:var(--muted);font-size:.76rem;font-weight:700;letter-spacing:.02em;}
[data-testid="stMetricValue"] {color:var(--ink);font-weight:780;letter-spacing:-.025em;}
[data-testid="stVerticalBlockBorderWrapper"] {background:rgba(255,255,255,.92);border-color:var(--line)!important;border-radius:16px!important;box-shadow:0 7px 22px rgba(16,42,60,.055);}
.upload-intro {display:block;margin:1.8rem 0 .75rem;padding:0 .1rem;}
.upload-intro h2 {font-size:1.38rem!important;line-height:1.15;margin:0!important;color:var(--ink);}
.upload-intro p {color:var(--muted);font-size:.88rem;margin:.28rem 0 0;}
[data-testid="stFileUploader"] {background:#fff;border:1px solid var(--line);border-radius:16px;padding:.45rem .8rem .7rem;box-shadow:0 5px 16px rgba(16,42,60,.045);}
[data-testid="stFileUploader"] > label {font-size:.82rem;font-weight:750;color:var(--ink);padding:.2rem 0 .35rem;}
[data-testid="stFileUploaderDropzone"] {display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:.55rem;min-height:132px;background:linear-gradient(135deg,#fbfdfe 0%,#f2f8fa 100%);border:1px dashed #9fc0cd;border-radius:12px;padding:1.15rem 1.25rem;transition:border-color .12s ease,background .12s ease,box-shadow .12s ease;}
[data-testid="stFileUploaderDropzone"]:before {content:"Drag and drop your file here";font-size:.9rem;font-weight:750;color:var(--ink);}
[data-testid="stFileUploaderDropzoneInstructions"] {flex:0 1 auto;color:var(--muted);font-size:.74rem;}
[data-testid="stFileChips"] {justify-content:center;}
[data-testid="stFileUploader"] {margin-bottom:.1rem;}
.file-rows {display:grid;gap:.35rem;margin:-.35rem 0 .1rem;}
.file-row {margin:0;}
[data-testid="stColumn"] .file-row {margin:0;}
.file-row {display:flex;align-items:center;gap:.65rem;height:42px;padding:0 2.6rem 0 .75rem;background:#fff;border:1px solid var(--line);border-radius:10px;box-shadow:0 2px 6px rgba(16,42,60,.04);}
.file-row .fr-icon {display:grid;place-items:center;flex:0 0 1.6rem;height:1.6rem;border-radius:7px;background:var(--teal-soft);color:var(--teal-dark);font-size:.9rem;}
.file-row .fr-name {flex:1 1 auto;color:var(--ink);font-size:.82rem;font-weight:650;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.file-row .fr-size {flex:0 0 auto;color:var(--muted);font-size:.74rem;font-variant-numeric:tabular-nums;}
[data-testid="stFileUploaderDropzone"]:hover {background:#eef7f9;border-color:var(--teal);box-shadow:inset 0 0 0 1px rgba(22,125,154,.08);}
[data-testid="stFileUploaderDropzone"] button {min-width:150px;}
[data-testid="stAlert"] {border-radius:12px;border:1px solid rgba(16,42,60,.08);}

.workflow div.is-done:before {background:var(--teal);color:#fff;content:"✓";}
.workflow div.is-active:before {background:var(--teal-dark);color:#fff;box-shadow:0 0 0 3px rgba(22,125,154,.16);}
.workflow div.is-active {color:var(--ink);font-weight:750;}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#fff;}

.stButton > button {border-radius:10px;border:1px solid #bfd0da;font-weight:700;min-height:2.55rem;transition:all .12s ease;}
.stButton > button:hover {border-color:var(--teal);color:var(--teal-dark);transform:translateY(-1px);}
.stButton > button[kind="primary"] {background:var(--teal);color:#fff;border-color:var(--teal);box-shadow:0 4px 12px rgba(22,125,154,.2);}
[data-testid="stDownloadButton"] > button {background:var(--teal);color:#fff;border:1px solid var(--teal);border-radius:10px;font-weight:750;min-height:2.7rem;box-shadow:0 4px 12px rgba(22,125,154,.20);transition:transform .12s ease,box-shadow .12s ease,background .12s ease;}
[data-testid="stDownloadButton"] > button:hover {background:var(--teal-dark);color:#fff;border-color:var(--teal-dark);transform:translateY(-1px);box-shadow:0 7px 17px rgba(22,125,154,.27);}
[data-testid="stDownloadButton"] > button:focus {color:#fff;}

.stTabs [data-baseweb="tab-list"] {gap:.25rem;background:#eaf0f4;border-radius:11px;padding:.25rem;width:max-content;max-width:100%;}
.stTabs [data-baseweb="tab"] {border-radius:8px;padding:.4rem .85rem;font-size:.82rem;font-weight:700;color:#526a7c;}
.stTabs [aria-selected="true"] {background:#fff;color:var(--ink);box-shadow:0 2px 8px rgba(16,42,60,.08);}
.stTabs [data-baseweb="tab-highlight"] {display:none;}
hr {border-color:var(--line);}

@media (max-width:760px) {
  [data-testid="stAppViewContainer"] > .main .block-container {padding:1rem .8rem 3rem;}
  .page-hero {padding:1.35rem 1.2rem;border-radius:16px;}
  .app-context {display:none;}
  .ranking-board {grid-template-columns:repeat(2,minmax(0,1fr));}
  .upload-intro {align-items:flex-start;margin-top:1.4rem;}
  [data-testid="stFileUploaderDropzone"] {min-height:116px;padding:1rem;}
  [data-testid="stMetric"] {min-height:auto;}
}
</style>""",
    unsafe_allow_html=True,
)

PAGES = {
    "Door · Abnormal resistance": render_door,
    "ACV · Air conditioning": render_acv,
    "Rail · Corrugation": render_rail,
    "SHM · Fatigue damage": render_shm,
}

st.markdown(
    """<div class="app-brandbar">
      <div class="app-context">Train condition monitoring · Operations workspace</div>
    </div>""",
    unsafe_allow_html=True,
)
WORKFLOW = ("Select a subsystem", "Upload telemetry", "Run analysis", "Review and download")

# Pages call st.stop() in their empty states, which would skip anything rendered after
# them - so the sidebar is built up front and the step is derived here instead.
RESULT_KEYS = {
    "Door · Abnormal resistance": ("door_result",),
    "ACV · Air conditioning": ("acv_results",),
    "Rail · Corrugation": ("rail_results",),
    "SHM · Fatigue damage": ("shm_results",),
}

with st.sidebar:
    st.markdown(
        """<div class="sidebar-brand"><strong>NEBULAX</strong><span>Condition monitoring workspace</span></div>""",
        unsafe_allow_html=True,
    )
    st.caption("ANALYSIS MODULE")
    subsystem = st.radio("Subsystem", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.markdown("**Workflow**")
    has_result = any(st.session_state.get(key) for key in RESULT_KEYS.get(subsystem, ()))
    st.session_state["nx_step"] = 4 if has_result else 2
    workflow_steps(WORKFLOW)
    st.markdown('<div class="privacy-note">🔒 Files are processed locally and are not uploaded to an external service.</div>', unsafe_allow_html=True)

PAGES[subsystem]()
