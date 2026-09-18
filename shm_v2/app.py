"""Streamlit page for analytical SHM fatigue-damage inference."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .inference import DamagePrediction, predict_source, validate_predictions
from .models import PhysicsDamageModel

ACCENT = "#167d9a"


@st.cache_resource
def _model() -> PhysicsDamageModel:
    return PhysicsDamageModel.load()


@st.cache_data(show_spinner=False)
def _analyse(contents: bytes, filename: str) -> DamagePrediction:
    return predict_source(BytesIO(contents), file_id=filename, model=_model())


def _amplitude_figure(result: DamagePrediction) -> go.Figure:
    amplitude = result.cycles["sigma_a"].to_numpy(dtype=float)
    weights = result.cycles["count"].to_numpy(dtype=float)
    counts, edges = np.histogram(amplitude, bins=50, weights=weights)
    centres = 0.5 * (edges[:-1] + edges[1:])
    figure = go.Figure(go.Bar(x=centres, y=counts, marker_color=ACCENT))
    figure.update_layout(title="Rainflow stress-amplitude histogram", xaxis_title="Stress amplitude", yaxis_title="Equivalent cycles", margin=dict(l=20, r=20, t=55, b=20))
    return figure


def _damage_figure(result: DamagePrediction) -> go.Figure:
    curve = result.cumulative
    stride = max(1, len(curve) // 2_000)
    plotted = curve.iloc[::stride]
    if len(curve) and (plotted.empty or plotted.index[-1] != curve.index[-1]):
        plotted = pd.concat([plotted, curve.iloc[[-1]]])
    figure = go.Figure(go.Scatter(x=100 * plotted["progress"], y=plotted["cumulative_damage"], mode="lines", line=dict(color=ACCENT, width=3)))
    figure.update_layout(title="Accumulated Miner damage through the signal", xaxis_title="Signal progress (%)", yaxis_title="Cumulative damage", margin=dict(l=20, r=20, t=55, b=20))
    return figure


def render() -> None:
    st.title("Structural fatigue damage")
    st.write("Upload one or more headerless, single-column dynamic-stress CSV files. The analytical model applies ASTM E1049 rainflow counting and Miner's rule.")
    st.caption("Validated model: fixed S-N exponent m = 5.00 · leave-one-out score 0.9729")

    uploads = st.file_uploader("Dynamic stress CSV files", type=["csv"], accept_multiple_files=True)
    sample_requested = st.button("Analyze supplied SHM signal")
    sources = [(upload.name, upload.getvalue()) for upload in uploads]
    if sample_requested and not sources:
        phase = np.linspace(0, 24 * np.pi, 8_000)
        sample = 8.0 * np.sin(phase) + 2.0 * np.sin(3.7 * phase)
        sources = [("sample_shm.csv", pd.Series(sample).to_csv(index=False, header=False).encode("utf-8"))]
    if not sources:
        st.info("Upload files or analyze the supplied synthetic signal to calculate cumulative fatigue damage.")
        return

    results: list[DamagePrediction] = []
    progress = st.progress(0, text="Preparing analysis…")
    try:
        for index, (filename, contents) in enumerate(sources, start=1):
            progress.progress((index - 1) / len(sources), text=f"Rainflow counting {filename}")
            results.append(_analyse(contents, filename))
        progress.progress(1.0, text="Analysis complete")
    except ValueError as exc:
        progress.empty()
        st.error(f"Could not analyse the upload: {exc}")
        return

    predictions = pd.DataFrame(
        ({"file_id": result.file_id, "prediction": result.prediction} for result in results),
        columns=["file_id", "prediction"],
    )
    try:
        validate_predictions(predictions, expected_file_ids={filename for filename, _ in sources}, expected_rows=len(sources))
    except ValueError as exc:
        st.error(f"Output validation failed: {exc}")
        return

    st.success(f"Analysed {len(results)} file{'s' if len(results) != 1 else ''}; output schema validated.")
    st.metric("Files analyzed", len(results))
    columns = st.columns(min(4, len(results)))
    for index, result in enumerate(results):
        columns[index % len(columns)].metric(result.file_id, f"{result.prediction:.6f}", help="Predicted cumulative Miner damage")

    selected_name = st.selectbox("Inspect file", [result.file_id for result in results])
    selected = next(result for result in results if result.file_id == selected_name)
    left, right = st.columns(2)
    left.plotly_chart(_amplitude_figure(selected), use_container_width=True)
    right.plotly_chart(_damage_figure(selected), use_container_width=True)
    st.caption("Why: every rainflow cycle contributes count × stress_amplitude⁵ / C; high-amplitude cycles dominate fatigue damage.")

    st.download_button(
        "Download SHM prediction CSV",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name="shm_predictions.csv",
        mime="text/csv",
        type="primary",
    )
