"""Streamlit page for Door cycle detection and fault classification."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui import page_header, section_intro, severity_badge, upload_intro, uploaded_files_list
from .inference import DEFAULT_FEATURE_PATH, DEFAULT_MODEL_PATH, load_artifact, predict_stream, predictions_csv

ROOT = Path(__file__).resolve().parents[1]
SEVERITY_ORDER = {"Critical": 0, "High": 1, "Advisory": 2, "Normal": 3}



def _abnormal_probabilities(result, model) -> np.ndarray:
    if result.probabilities is None:
        return np.where(result.predictions["prediction"].eq("Abnormal resistance"), 1.0, 0.0)
    classes = list(model.classes_)
    return result.probabilities[:, classes.index(1)]


def _event_decisions(result, model) -> list[dict[str, object]]:
    probabilities = _abnormal_probabilities(result, model)
    abnormal = result.predictions["prediction"].eq("Abnormal resistance").to_numpy()
    normal_durations = result.segments.loc[~abnormal, "duration_seconds"]
    duration_limit = float(normal_durations.quantile(0.90)) if len(normal_durations) else float("inf")
    terminal_columns = [column for column in ["Door Opened", "Door Locked"] if column in result.stream]
    decisions = []
    for index, (prediction, segment) in enumerate(zip(result.predictions.itertuples(index=False), result.segments.itertuples(index=False), strict=True)):
        if not abnormal[index]:
            severity, confidence = "Normal", "High" if probabilities[index] < 0.25 else "Moderate"
            verdict, action = "Door movement completed normally", "No immediate action; continue routine monitoring."
        else:
            cycle = result.stream.iloc[int(segment.start_idx) : int(segment.end_idx) + 1]
            terminal_reached = True
            if terminal_columns:
                terminal_values = cycle[terminal_columns].tail(max(1, len(cycle) // 10)).apply(pd.to_numeric, errors="coerce")
                terminal_reached = bool((terminal_values > 0).any().any())
            repeated = (index > 0 and abnormal[index - 1]) or (index + 1 < len(abnormal) and abnormal[index + 1])
            prolonged = float(segment.duration_seconds) > duration_limit
            if not terminal_reached:
                severity, verdict = "Critical", "Resistance detected and the movement did not reach a terminal state"
                action = "Isolate the affected door under operating procedure and inspect it before return to service."
            elif repeated or prolonged or probabilities[index] >= 0.80:
                severity, verdict = "High", "Clear abnormal resistance during door movement"
                action = "Inspect the guide rail, rollers, seals and travel path before extended operation."
            else:
                severity, verdict = "Advisory", "Possible abnormal resistance during one door movement"
                action = "Monitor subsequent events and inspect at the next planned stop."
            confidence = "High" if probabilities[index] >= 0.80 else "Moderate" if probabilities[index] >= 0.65 else "Low"
        decisions.append(
            {
                "index": index,
                "time": prediction.start_time,
                "severity": severity,
                "verdict": verdict,
                "action": action,
                "confidence": confidence,
                "probability": float(probabilities[index]),
            }
        )
    return decisions


def _current_profile_figure(result, event_index: int) -> go.Figure:
    grid = np.linspace(0, 100, 101)
    normal_profiles = []
    for index, segment in enumerate(result.segments.itertuples(index=False)):
        if result.predictions.iloc[index]["prediction"] != "Normal":
            continue
        current = result.stream.iloc[int(segment.start_idx) : int(segment.end_idx) + 1]["Motor current(mA)"].to_numpy(dtype=float)
        normal_profiles.append(np.interp(grid, np.linspace(0, 100, len(current)), current))
    selected_segment = result.segments.iloc[event_index]
    selected_current = result.stream.iloc[int(selected_segment.start_idx) : int(selected_segment.end_idx) + 1]["Motor current(mA)"].to_numpy(dtype=float)
    selected_profile = np.interp(grid, np.linspace(0, 100, len(selected_current)), selected_current)
    figure = go.Figure()
    if normal_profiles:
        profiles = np.asarray(normal_profiles)
        low, median, high = np.quantile(profiles, [0.10, 0.50, 0.90], axis=0)
        figure.add_trace(go.Scatter(x=grid, y=high, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=grid, y=low, fill="tonexty", fillcolor="rgba(22,125,154,0.16)", line=dict(width=0), name="Normal 10–90% range"))
        figure.add_trace(go.Scatter(x=grid, y=median, line=dict(color="#167d9a", dash="dash"), name="Normal median"))
    figure.add_trace(go.Scatter(x=grid, y=selected_profile, line=dict(color="#d1495b", width=3), name="Selected event"))
    figure.update_layout(title="Motor current as the door moves", xaxis_title="Movement progress (%)", yaxis_title="Motor current (mA)", margin=dict(l=20, r=20, t=55, b=20))
    return figure


@st.cache_resource
def _model(model_path: str, feature_path: str, model_stamp: int, feature_stamp: int):
    del model_stamp, feature_stamp
    return load_artifact(model_path, feature_path)


def render() -> None:
    """Render the Door analysis workflow inside the shared app."""
    page_header(
        eyebrow="Train doors",
        title="Determine abnormal and normal door operations",
        description="Finds every door opening and closing in the recording, flags the ones that met resistance, and shows the motor current behind each.",
        tags=("One continuous recording", "Finds each door cycle", "Spots stiff doors"),
    )

    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_FEATURE_PATH.exists():
        st.error("The trained Door model or feature list is missing.")
        st.stop()
    model, feature_names = _model(
        str(DEFAULT_MODEL_PATH),
        str(DEFAULT_FEATURE_PATH),
        DEFAULT_MODEL_PATH.stat().st_mtime_ns,
        DEFAULT_FEATURE_PATH.stat().st_mtime_ns,
    )
    upload_intro("Upload a door recording", "Drop in one CSV of continuous door data. Each opening and closing is found for you and labelled with the time it happened.")
    with st.container():
        upload = st.file_uploader(
            "Upload Door telemetry",
            type=["csv"],
            help="Upload one continuous stream with the same columns as Door Test.csv.",
            key="door_upload",
        )
        upload = (uploaded_files_list(upload, "door") or [None])[0]
        with st.expander("Try the supplied training stream"):
            demo = ROOT / "PS3/02_Datasets/Door/Train.csv"
            demo_clicked = st.button("Analyze supplied Door stream", disabled=not demo.exists())
        run = st.button(
            "Analyze Door stream",
            type="primary",
            disabled=upload is None,
            width="stretch",
            help=None if upload is not None else "Add a telemetry CSV above to enable analysis.",
        )

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
        return

    decisions = _event_decisions(result, model)
    urgent = sorted(decisions, key=lambda row: (SEVERITY_ORDER[row["severity"]], row["time"]))[0]
    section_intro("What to do now", "The most urgent problem first, with one clear next step.")
    with st.container(border=True):
        if urgent["severity"] == "Normal":
            severity_badge("Normal")
            st.subheader("No abnormal resistance detected")
            st.write("**Do now:** Continue routine monitoring.")
        else:
            severity_badge(str(urgent["severity"]))
            st.subheader(str(urgent["verdict"]))
            st.write(f"**Event time:** {urgent['time']}")
            st.write(f"**Do now:** {urgent['action']}")
            st.write(f"**Decision confidence:** {urgent['confidence']}")
        st.caption("The recording is one continuous stream with no door number in it, so events are labelled by time only.")

    abnormal_events = [row for row in decisions if row["severity"] != "Normal"]
    if abnormal_events:
        st.dataframe(
            pd.DataFrame(abnormal_events)[["time", "severity", "verdict", "confidence", "action"]].rename(columns=str.title),
            hide_index=True,
            width="stretch",
        )

    csv = predictions_csv(result)
    st.download_button(
        "Download Door prediction CSV",
        csv,
        "door_predictions.csv",
        "text/csv",
        icon=":material/download:",
        width="stretch",
    )
    abnormal = int(result.predictions["prediction"].eq("Abnormal resistance").sum())
    normal = len(result.predictions) - abnormal
    a, b, c = st.columns(3)
    a.metric("Cycles detected", len(result.predictions))
    b.metric("Abnormal resistance", abnormal)
    c.metric("Normal", normal)

    overview, details = st.tabs(["Why this result", "Prediction details"])
    with overview:
        selected_event = st.selectbox(
            "Event to explain",
            list(range(len(decisions))),
            index=int(urgent["index"]),
            format_func=lambda index: f"{decisions[index]['time']} · {decisions[index]['severity']}",
        )
        st.plotly_chart(_current_profile_figure(result, selected_event), width="stretch")
        cycle = result.stream.iloc[int(result.segments.iloc[selected_event].start_idx) : int(result.segments.iloc[selected_event].end_idx) + 1]
        selected_peak = float(cycle["Motor current(mA)"].max())
        normal_peaks = []
        for index, segment in enumerate(result.segments.itertuples(index=False)):
            if result.predictions.iloc[index]["prediction"] == "Normal":
                normal_peaks.append(float(result.stream.iloc[int(segment.start_idx) : int(segment.end_idx) + 1]["Motor current(mA)"].max()))
        comparison = float(np.median(normal_peaks)) if normal_peaks else float("nan")
        if np.isfinite(comparison):
            st.write(f"This movement reached {selected_peak:.0f} mA peak motor current, compared with a {comparison:.0f} mA median peak across normal movements in the same stream.")
        else:
            st.write("This result came from the motor-current pattern during the movement. There was no normal movement in this recording to compare it against.")
        st.caption("The shaded band is the normal current range. Drawing more current than that to move the same distance usually means something is rubbing or blocking - a guide, a roller, a seal or an obstruction.")
    with details:
        table = result.predictions.copy()
        table.insert(0, "cycle", range(1, len(table) + 1))
        table["duration_seconds"] = result.segments["duration_seconds"].round(3).to_numpy()
        table["model_evidence"] = _abnormal_probabilities(result, model).round(4)
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption("This is a guide to what to check, not a probability that the door will fail.")
