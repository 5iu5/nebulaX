"""Streamlit page for Door cycle detection and fault classification."""

from pathlib import Path

import plotly.express as px
import streamlit as st

from .inference import DEFAULT_FEATURE_PATH, DEFAULT_MODEL_PATH, load_artifact, predict_stream, predictions_csv

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def _model(model_path: str, feature_path: str, model_stamp: int, feature_stamp: int):
    del model_stamp, feature_stamp
    return load_artifact(model_path, feature_path)


def render() -> None:
    """Render the Door analysis workflow inside the shared app."""
    st.title("Detect abnormal door resistance")
    st.write("Find each door cycle in a continuous telemetry stream and classify it as normal or abnormal resistance.")

    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_FEATURE_PATH.exists():
        st.error("The trained Door model or feature list is missing.")
        st.stop()
    model, feature_names = _model(
        str(DEFAULT_MODEL_PATH),
        str(DEFAULT_FEATURE_PATH),
        DEFAULT_MODEL_PATH.stat().st_mtime_ns,
        DEFAULT_FEATURE_PATH.stat().st_mtime_ns,
    )

    upload = st.file_uploader(
        "Upload Door telemetry",
        type=["csv"],
        help="Upload one continuous stream with the same columns as Door Test.csv.",
        key="door_upload",
    )
    with st.expander("Try the supplied test stream"):
        demo = ROOT / "PS3/02_Datasets/Door/Test.csv"
        demo_clicked = st.button("Analyze supplied Door stream", disabled=not demo.exists())
    run = st.button("Analyze Door stream", type="primary", disabled=upload is None)

    if run or demo_clicked:
        source = upload if run else demo
        try:
            with st.spinner("Detecting and classifying door cycles…"):
                st.session_state.door_result = predict_stream(source, model, feature_names)
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"Unable to analyze this Door stream: {exc}")
            st.stop()

    result = st.session_state.get("door_result")
    if result is None:
        st.info("Upload a Door CSV to view detected cycle boundaries and classifications.")
        return

    csv = predictions_csv(result)
    st.download_button("Download Door prediction CSV", csv, "door_predictions.csv", "text/csv")
    abnormal = int(result.predictions["prediction"].eq("Abnormal resistance").sum())
    normal = len(result.predictions) - abnormal
    a, b, c = st.columns(3)
    a.metric("Cycles detected", len(result.predictions))
    b.metric("Abnormal resistance", abnormal)
    c.metric("Normal", normal)

    overview, details = st.tabs(["Cycle overview", "Prediction details"])
    with overview:
        chart = result.predictions.copy()
        chart["cycle"] = range(1, len(chart) + 1)
        counts = chart["prediction"].value_counts().rename_axis("Status").reset_index(name="Cycles")
        st.plotly_chart(px.bar(counts, x="Status", y="Cycles", color="Status"), width="stretch")
        st.caption("Cycles are separated at telemetry gaps greater than one second, matching the training segmentation rule.")
    with details:
        table = result.predictions.copy()
        table.insert(0, "cycle", range(1, len(table) + 1))
        table["duration_seconds"] = result.segments["duration_seconds"].round(3).to_numpy()
        if result.probabilities is not None:
            classes = list(model.classes_)
            abnormal_index = classes.index(1)
            table["abnormal_probability"] = result.probabilities[:, abnormal_index].round(4)
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption("Probabilities are diagnostic model outputs; only segment boundaries and labels are scored.")
