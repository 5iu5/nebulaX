"""Streamlit page for analytical SHM fatigue-damage inference."""

from __future__ import annotations

from io import BytesIO
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui import page_header, section_intro, severity_badge, upload_intro, uploaded_files_list
from .inference import DamagePrediction, predict_source, validate_predictions
from .models import DEFAULT_METADATA_PATH, PhysicsDamageModel

ACCENT = "#167d9a"
SEVERITY_ORDER = {"Critical": 0, "High": 1, "Advisory": 2, "Normal": 3}
SEVERITY_COLOURS = {"Critical": "#9b2226", "High": "#d1495b", "Advisory": "#e9a23b", "Normal": "#2a9d8f"}


@st.cache_resource
def _model() -> PhysicsDamageModel:
    return PhysicsDamageModel.load()


@st.cache_resource
def _damage_reference() -> dict[str, float]:
    metadata = json.loads(DEFAULT_METADATA_PATH.read_text(encoding="utf-8"))
    reference = metadata["segment_damage_reference"]
    return {"p50": float(reference["p50"]), "p90": float(reference["p90"]), "failure": float(reference["failure_criterion"])}



def _severity(damage: float, reference: dict[str, float]) -> str:
    if damage >= reference["failure"]:
        return "Critical"
    if damage > reference["p90"]:
        return "High"
    if damage > reference["p50"]:
        return "Advisory"
    return "Normal"


def _guidance(severity: str) -> tuple[str, str]:
    return {
        "Normal": ("Typical damage accumulation for one recording segment", "Continue routine monitoring."),
        "Advisory": ("Elevated damage accumulation in this segment", "Compare adjacent segments and review at the next planned maintenance visit."),
        "High": ("Unusually high damage accumulation in this segment", "Prioritise a structural inspection before extended operation."),
        "Critical": ("The D=1 failure criterion was reached within this segment", "Escalate immediately and inspect before return to service."),
    }[severity]


def _decision_confidence(damage: float, reference: dict[str, float]) -> str:
    boundaries = [reference["p50"], reference["p90"], reference["failure"]]
    near_boundary = any(abs(damage - boundary) <= max(0.05 * boundary, 0.005) for boundary in boundaries)
    return "Moderate — close to a severity boundary" if near_boundary else "High — clearly separated from severity boundaries"


@st.cache_data(show_spinner=False)
def _analyse(contents: bytes, filename: str) -> DamagePrediction:
    return predict_source(BytesIO(contents), file_id=filename, model=_model())


def _amplitude_figure(result: DamagePrediction) -> go.Figure:
    amplitude = result.cycles["sigma_a"].to_numpy(dtype=float)
    weights = result.cycles["count"].to_numpy(dtype=float)
    counts, edges = np.histogram(amplitude, bins=50, weights=weights)
    centres = 0.5 * (edges[:-1] + edges[1:])
    figure = go.Figure(go.Bar(x=centres, y=counts, marker_color=ACCENT))
    figure.update_layout(title="Stress cycles, grouped by size", xaxis_title="Size of stress cycle", yaxis_title="Number of cycles", margin=dict(l=20, r=20, t=55, b=20))
    return figure


def _damage_figure(result: DamagePrediction, reference: dict[str, float]) -> go.Figure:
    curve = result.cumulative
    stride = max(1, len(curve) // 2_000)
    plotted = curve.iloc[::stride]
    if len(curve) and (plotted.empty or plotted.index[-1] != curve.index[-1]):
        plotted = pd.concat([plotted, curve.iloc[[-1]]])
    figure = go.Figure(go.Scatter(x=100 * plotted["progress"], y=plotted["cumulative_damage"], mode="lines", name="This segment", line=dict(color=ACCENT, width=3)))
    figure.add_hline(y=reference["p50"], line_dash="dot", line_color=SEVERITY_COLOURS["Normal"], annotation_text="Training median")
    figure.add_hline(y=reference["p90"], line_dash="dash", line_color=SEVERITY_COLOURS["High"], annotation_text="Training p90")
    figure.add_hline(y=reference["failure"], line_dash="dash", line_color=SEVERITY_COLOURS["Critical"], annotation_text="D=1 criterion")
    figure.update_layout(title="Wear building up across this recording", xaxis_title="Signal progress (%)", yaxis_title="Wear (1.0 = end of life)", margin=dict(l=20, r=20, t=55, b=20), showlegend=False)
    return figure


def render() -> None:
    page_header(
        eyebrow="Structural wear",
        title="How much structural life has been used",
        description="Shows how much wear each recording added to the train's frame, worst first, and what caused it.",
        tags=("One or more CSV files", "Wear per recording", "2.7% average error in testing"),
    )
    reference = _damage_reference()
    upload_intro("Upload stress recordings", "Drop in one or more CSV files. Each should be a single column of numbers with no header row.")
    with st.container():
        uploads = st.file_uploader(
            "Dynamic stress CSV files", type=["csv"], accept_multiple_files=True, key="shm_upload"
        )
        uploads = uploaded_files_list(uploads, "shm")
        run = st.button(
            "Analyze uploaded recordings",
            type="primary",
            disabled=not uploads,
            width="stretch",
            help=None if uploads else "Add at least one CSV above to enable analysis.",
        )
        sample_requested = st.button("Analyze supplied SHM signal", width="stretch")

    # Analysis runs only on an explicit click; uploading a file must not start it.
    sources: list = []
    if run:
        sources = [(upload.name, upload.getvalue()) for upload in uploads]
    elif sample_requested:
        phase = np.linspace(0, 24 * np.pi, 8_000)
        sample = 8.0 * np.sin(phase) + 2.0 * np.sin(3.7 * phase)
        sources = [("sample_shm.csv", pd.Series(sample).to_csv(index=False, header=False).encode("utf-8"))]

    if sources:
        _run_analysis(sources)

    results = st.session_state.get("shm_results")
    if not results:
        return
    _render_results(results, reference, st.session_state["shm_predictions"])


def _run_analysis(sources) -> None:
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

    # Persisted so results survive the next interaction instead of vanishing.
    st.session_state.shm_results = results
    st.session_state.shm_predictions = predictions


def _render_results(results, reference, predictions) -> None:
    st.success(f"Analysed {len(results)} file{'s' if len(results) != 1 else ''}; output schema validated.")
    triage = []
    for result in results:
        severity = _severity(result.prediction, reference)
        verdict, action = _guidance(severity)
        triage.append({"File": result.file_id, "Severity": severity, "Verdict": verdict, "Action": action, "Damage in segment": result.prediction})
    triage.sort(key=lambda row: (SEVERITY_ORDER[row["Severity"]], -row["Damage in segment"]))

    section_intro("What to do now", "Recordings sorted with the most worn at the top. Open one only if you want the detail.")
    st.metric("Files analyzed", len(results))
    st.dataframe(pd.DataFrame(triage).drop(columns=["Damage in segment"]), hide_index=True, width="stretch")
    st.caption("Sorted with the most worn first. File names do not say where on the train the sensor was - a real deployment would need to record the train, the sensor position and the time.")

    selected_name = st.selectbox("Inspect result", [row["File"] for row in triage])
    selected = next(result for result in results if result.file_id == selected_name)
    severity = _severity(selected.prediction, reference)
    verdict, action = _guidance(severity)
    comparable_segments = reference["failure"] / selected.prediction
    with st.container(border=True):
        severity_badge(severity)
        st.subheader(verdict)
        a, b, c = st.columns(3)
        a.metric("Damage accrued in this segment", f"D = {selected.prediction:.3f}")
        b.metric("Equivalent rate", f"~{comparable_segments:.1f} segments to D=1")
        c.metric("Decision confidence", _decision_confidence(selected.prediction, reference))
        st.write(f"**Do now:** {action}")
        st.caption(f"Normal means at or below the typical recording ({reference['p50']:.3f}). High means worse than 9 out of 10 recordings ({reference['p90']:.3f}). Critical is 1.0, the point of failure.")

    section_intro("Why this result", "How this recording compares with normal, and which stress cycles caused the wear.")
    st.plotly_chart(_damage_figure(selected, reference), width="stretch")
    st.plotly_chart(_amplitude_figure(selected), width="stretch")
    high_cycle_share = 100 * selected.cycles.nlargest(max(1, len(selected.cycles) // 100), "sigma_a").eval("count * sigma_a ** 5").sum() / selected.cycles.eval("count * sigma_a ** 5").sum()
    st.write(f"Repeated high-amplitude stress cycles drove this result: the highest-amplitude 1% of counted cycles contributed approximately {high_cycle_share:.1f}% of the segment damage.")
    st.caption("1.0 is the accepted point of failure. This number is the wear added by this one recording - it is not the total wear across the train's life, which needs the full history of past recordings.")

    st.download_button(
        "Download SHM prediction CSV",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name="shm_predictions.csv",
        mime="text/csv",
        icon=":material/download:",
        type="primary",
    )
