"""Streamlit page for structural-health damage estimation."""

from pathlib import Path

import plotly.express as px
import streamlit as st

from .inference import DEFAULT_FEATURE_PATH, DEFAULT_MODEL_PATH, load_artifact, predict_files, predictions_csv

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def _model(model_path: str, feature_path: str, model_stamp: int, feature_stamp: int):
    del model_stamp, feature_stamp
    return load_artifact(model_path, feature_path)


def render() -> None:
    """Render SHM upload, inference, diagnostics, and export controls."""
    st.title("Estimate cumulative fatigue damage")
    st.write("Summarize each dynamic-stress signal and estimate its non-negative cumulative fatigue damage.")

    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_FEATURE_PATH.exists():
        st.error("The trained SHM model or feature list is missing.")
        st.stop()
    model, feature_names = _model(
        str(DEFAULT_MODEL_PATH),
        str(DEFAULT_FEATURE_PATH),
        DEFAULT_MODEL_PATH.stat().st_mtime_ns,
        DEFAULT_FEATURE_PATH.stat().st_mtime_ns,
    )

    uploads = st.file_uploader(
        "Upload SHM stress signals",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more CSV files. The first column must contain the numeric stress signal.",
        key="shm_upload",
    )
    with st.expander("Try a supplied test signal"):
        demo = ROOT / "PS3/02_Datasets/SHM/Test/test01.csv"
        demo_clicked = st.button("Analyze supplied SHM signal", disabled=not demo.exists())
    run = st.button("Analyze SHM files", type="primary", disabled=not uploads)

    if run or demo_clicked:
        sources = [(upload.name, upload) for upload in uploads] if run else [(demo.name, demo)]
        try:
            with st.spinner("Extracting stress features and estimating damage…"):
                st.session_state.shm_result = predict_files(sources, model, feature_names)
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"Unable to analyze these SHM files: {exc}")
            st.stop()

    result = st.session_state.get("shm_result")
    if result is None:
        st.info("Upload SHM signal CSVs to receive one cumulative-damage estimate per file.")
        return

    csv = predictions_csv(result)
    st.download_button("Download SHM prediction CSV", csv, "shm_predictions.csv", "text/csv")
    predictions = result.predictions
    a, b, c = st.columns(3)
    a.metric("Files analyzed", len(predictions))
    b.metric("Mean predicted damage", f"{predictions.prediction.mean():.4f}")
    c.metric("Maximum predicted damage", f"{predictions.prediction.max():.4f}")

    chart = px.bar(predictions, x="file_id", y="prediction", labels={"file_id": "Signal file", "prediction": "Predicted cumulative damage"})
    st.plotly_chart(chart, width="stretch")
    st.dataframe(predictions, width="stretch", hide_index=True)
    st.caption("The Gradient Boosting model uses 27 statistical and spectral signal features. Predictions are clipped at zero because damage is non-negative.")
