"""Streamlit page for rail corrugation inference."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import welch

from app.ui import page_header, section_intro, severity_badge, upload_intro, uploaded_files_list
from .features import FS, channel_groups
from .inference import RailPrediction, predict_source, validate_predictions
from .models import load_model

ACCENT = "#167d9a"
SIDE_I_COLOUR = "#7b2cbf"
SIDE_II_COLOUR = "#e76f51"
SEVERITY_ORDER = {"High": 0, "Advisory": 1, "Normal": 2}



def _operator_decision(result: RailPrediction) -> tuple[str, str, str, str]:
    ordered = sorted(result.probabilities.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
    strength = result.confidence
    if result.prediction == "Normal" and (strength < 0.60 or margin < 0.15):
        return "Advisory", "Result is close to the corrugation threshold", "Repeat the recording or request engineering review.", "Low"
    if result.prediction == "Normal":
        return "Normal", "No corrugation signature detected", "Continue routine track monitoring.", "High" if margin >= 0.30 else "Moderate"
    side = result.prediction
    if strength >= 0.65 and margin >= 0.15:
        return "High", f"Possible rail corrugation on {side}", f"Create a track inspection work order for {side} and assess grinding or reprofiling.", "High"
    return "Advisory", f"Weak corrugation indication on {side}", "Repeat the measurement and have an engineer review the spectrum.", "Low"


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
        title="Vibration energy by rail side",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Vibration energy",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return figure


def render() -> None:
    page_header(
        eyebrow="Track condition",
        title="Find rippled, worn track",
        description="Works out which side of the track has worn into ripples, puts the worst recordings first, and shows the vibration pattern behind each result.",
        tags=("One or more CSV files", "Tells you which rail side", "Adjusts for train speed"),
    )
    upload_intro("Upload track vibration recordings", "Drop in one or more CSV files. Each holds one second of vibration measured at the wheel bearings.")
    with st.container():
        uploads = st.file_uploader(
            "Rail recording CSV files", type=["csv"], accept_multiple_files=True, key="rail_upload"
        )
        uploads = uploaded_files_list(uploads, "rail")
        run = st.button(
            "Analyze uploaded recordings",
            type="primary",
            disabled=not uploads,
            width="stretch",
            help=None if uploads else "Add at least one CSV above to enable analysis.",
        )
        sample_requested = st.button("Analyze supplied rail sample", width="stretch")

    # Analysis runs only on an explicit click; uploading a file must not start it.
    sources: list = []
    if run:
        sources = [(upload.name, upload.getvalue()) for upload in uploads]
    elif sample_requested:
        sources = [("sample_rail.csv", _sample_csv())]

    if sources:
        _run_analysis(sources)

    results = st.session_state.get("rail_results")
    if not results:
        return
    _render_results(results, st.session_state["rail_sources"], st.session_state["rail_predictions"])


def _run_analysis(sources) -> None:
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

    # Persisted so results survive the next interaction instead of vanishing.
    st.session_state.rail_results = results
    st.session_state.rail_sources = dict(sources)
    st.session_state.rail_predictions = predictions


def _render_results(results, sources, predictions) -> None:
    st.success(f"Analysed {len(results)} file{'s' if len(results) != 1 else ''}; output schema validated.")
    triage = []
    for result in results:
        severity, verdict, action, confidence = _operator_decision(result)
        triage.append({"File": result.file_id, "Severity": severity, "Verdict": verdict, "Confidence": confidence, "Action": action})
    triage.sort(key=lambda row: SEVERITY_ORDER[row["Severity"]])
    section_intro("What to check first", "Recordings are sorted with the most urgent at the top.")
    st.dataframe(pd.DataFrame(triage), hide_index=True, width="stretch")

    selected_name = st.selectbox("Inspect result", [row["File"] for row in triage])
    selected = next(r for r in results if r.file_id == selected_name)
    selected_contents = sources[selected_name]
    severity, verdict, action, confidence = _operator_decision(selected)
    with st.container(border=True):
        severity_badge(severity)
        st.subheader(verdict)
        a, b = st.columns(2)
        a.metric("Decision confidence", confidence)
        b.metric("Affected side", selected.prediction if selected.prediction != "Normal" else "None detected")
        st.write(f"**Do now:** {action}")
        st.caption("A Critical warning is never raised on this result alone. It also needs a separate, agreed vibration or shock safety limit to be exceeded.")

    section_intro("Why this result", "The vibration pattern for each side of the track, so you can see what the result is based on.")
    st.plotly_chart(_spectrum_figure(selected_contents), width="stretch")
    side_name = selected.prediction if selected.prediction != "Normal" else "both rail sides"
    st.write(f"The vibration spectrum for {side_name} produced the strongest periodic corrugation evidence after comparing energy and axle-position patterns between Side I and Side II.")
    with st.expander("Model evidence"):
        probability_table = pd.DataFrame({"Class": list(selected.probabilities), "Relative model evidence": list(selected.probabilities.values())}).sort_values("Relative model evidence", ascending=False)
        st.dataframe(probability_table, hide_index=True, width="stretch")
        st.caption("These figures show how strong the evidence is, not the chance of a fault.")

    st.download_button(
        "Download rail prediction CSV",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name="rail_predictions.csv",
        mime="text/csv",
        icon=":material/download:",
        type="primary",
    )
