"""Streamlit page for rail corrugation inference."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import welch

from .features import FS, channel_groups
from .inference import RailPrediction, predict_source, validate_predictions
from .models import load_model

ACCENT = "#167d9a"
SIDE_I_COLOUR = "#7b2cbf"
SIDE_II_COLOUR = "#e76f51"


@st.cache_resource
def _model():
    return load_model()


@st.cache_data(show_spinner=False)
def _analyse(contents: bytes, filename: str) -> RailPrediction:
    return predict_source(BytesIO(contents), file_id=filename, model=_model())


def _sample_csv() -> bytes:
    rng = np.random.default_rng(42)
    n = 10_000
    t = np.arange(n) / FS
    data: dict[str, np.ndarray] = {"speed_signal": (np.sin(2 * np.pi * 90 * t) > 0).astype(int)}
    for car in range(1, 9):
        for position in range(1, 9):
            side_boost = 1.8 if position in (1, 3, 5, 7) else 1.0
            carrier = np.sin(2 * np.pi * (340 + 10 * position) * t)
            noise = rng.normal(0, 0.35, n)
            data[f"position {position} vibration car {car}"] = side_boost * carrier + noise
            data[f"position {position} shock car {car}"] = rng.normal(0, 0.5, n)
    return pd.DataFrame(data).to_csv(index=False).encode("utf-8")


def _spectrum_figure(contents: bytes) -> go.Figure:
    data = pd.read_csv(BytesIO(contents))
    groups = channel_groups(data)
    figure = go.Figure()
    for label, cols, colour in [
        ("Side I sensors", groups["side1_vibration"], SIDE_I_COLOUR),
        ("Side II sensors", groups["side2_vibration"], SIDE_II_COLOUR),
    ]:
        signal = data[cols].mean(axis=1).to_numpy(dtype=float)
        f, psd = welch(signal, fs=FS, nperseg=min(2048, len(signal)))
        figure.add_trace(go.Scatter(x=f, y=psd, mode="lines", name=label, line=dict(color=colour, width=2)))
    for x0, x1 in [(100, 250), (250, 500), (500, 1000)]:
        figure.add_vrect(x0=x0, x1=x1, fillcolor=ACCENT, opacity=0.08, line_width=0)
    figure.update_layout(
        title="Mean vibration PSD by rail side",
        xaxis_title="Frequency (Hz)",
        yaxis_title="PSD",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return figure


def render() -> None:
    st.title("Rail corrugation")
    st.write("Upload one or more 1-second axle-box vibration CSV recordings. The model classifies each file as Normal, Side I corrugation, or Side II corrugation.")
    st.caption("Model: balanced Random Forest using side-wise time, spectral, and spatial asymmetry features · metric: macro F1")

    uploads = st.file_uploader("Rail recording CSV files", type=["csv"], accept_multiple_files=True)
    sample_requested = st.button("Analyze supplied rail sample")
    sources = [(upload.name, upload.getvalue()) for upload in uploads]
    if sample_requested and not sources:
        sources = [("sample_rail.csv", _sample_csv())]
    if not sources:
        st.info("Upload rail CSV files or analyze the supplied synthetic rail sample.")
        return

    results: list[RailPrediction] = []
    progress = st.progress(0, text="Preparing rail analysis…")
    try:
        for index, (filename, contents) in enumerate(sources, start=1):
            progress.progress((index - 1) / len(sources), text=f"Extracting features from {filename}")
            results.append(_analyse(contents, filename))
        progress.progress(1.0, text="Analysis complete")
    except Exception as exc:  # Streamlit page should show a friendly message, not a traceback.
        progress.empty()
        st.error(f"Could not analyse the upload: {exc}")
        return

    predictions = pd.DataFrame(({"file_id": r.file_id, "prediction": r.prediction} for r in results), columns=["file_id", "prediction"])
    try:
        validate_predictions(predictions, expected_file_ids={name for name, _ in sources}, expected_rows=len(sources))
    except ValueError as exc:
        st.error(f"Output validation failed: {exc}")
        return

    st.success(f"Analysed {len(results)} file{'s' if len(results) != 1 else ''}; output schema validated.")
    cols = st.columns(min(4, len(results)))
    for i, result in enumerate(results):
        cols[i % len(cols)].metric(result.file_id, result.prediction, f"confidence {result.confidence:.0%}")

    selected_name = st.selectbox("Inspect file", [r.file_id for r in results])
    selected = next(r for r in results if r.file_id == selected_name)
    selected_contents = next(contents for name, contents in sources if name == selected_name)

    left, right = st.columns([2, 1])
    left.plotly_chart(_spectrum_figure(selected_contents), use_container_width=True)
    probability_table = pd.DataFrame(
        {"class": list(selected.probabilities), "probability": list(selected.probabilities.values())}
    ).sort_values("probability", ascending=False)
    right.bar_chart(probability_table.set_index("class"), color=ACCENT)
    st.caption("Why: the classifier compares vibration energy, spectral peaks, and side-to-side axle-position asymmetry; a side prediction means that side showed the stronger corrugation signature.")

    st.download_button(
        "Download rail prediction CSV",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name="rail_predictions.csv",
        mime="text/csv",
        type="primary",
    )
