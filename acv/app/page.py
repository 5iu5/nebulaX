from pathlib import Path
from html import escape
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
import plotly.express as px
import streamlit as st
from app.ui import page_header, section_intro, severity_badge, upload_intro, uploaded_files_list
from acv.common import save_json
from acv.inference import load_artifact, rank_case, predictions_csv, prediction_zip
from acv.robustness import bootstrap_stability


SEVERITY_ORDER = {"High": 0, "Advisory": 1}


def _operator_decision(r):
    top = r.ranked_cars[0]
    row = r.table.loc[top]
    second_score = float(r.table.iloc[1]["score"]) if len(r.table) > 1 else 0.0
    margin = float(row["score"] - second_score)
    peer_difference = float(row["peer_mean"]) if pd.notna(row["peer_mean"]) else 0.0
    persistence = float(row.get("peer_gt_05", 0.0)) if pd.notna(row.get("peer_gt_05", 0.0)) else 0.0
    usable_hours = float(row["usable_hours"])
    if usable_hours < 1 or margin < 0.10:
        return {
            "severity": "Advisory",
            "verdict": f"No clear single-car outlier; inspect Car {top} first",
            "action": "Repeat monitoring and review the first two ranked cars at the next depot visit.",
            "confidence": "Low",
            "top": top,
            "margin": margin,
            "peer_difference": peer_difference,
            "persistence": persistence,
        }
    if peer_difference >= 0.5 and persistence >= 0.25:
        return {
            "severity": "High",
            "verdict": f"Possible cooling-system leak in Car {top}",
            "action": f"Inspect Car {top}'s refrigerant circuit, compressor operation, valves and temperature sensors.",
            "confidence": "High" if margin >= 0.20 else "Moderate",
            "top": top,
            "margin": margin,
            "peer_difference": peer_difference,
            "persistence": persistence,
        }
    return {
        "severity": "Advisory",
        "verdict": f"Car {top} shows the strongest cooling concern",
        "action": f"Monitor Car {top} and inspect it at the next planned depot visit.",
        "confidence": "Moderate",
        "top": top,
        "margin": margin,
        "peer_difference": peer_difference,
        "persistence": persistence,
    }


def _ranking_board(r) -> None:
    cards = []
    for rank, car in enumerate(r.ranked_cars, start=1):
        row = r.table.loc[car]
        available = bool(row["available"])
        peer_difference = float(row["peer_mean"]) if available and pd.notna(row["peer_mean"]) else None
        if not available:
            evidence = "No usable cooling data"
        elif peer_difference is None:
            evidence = "Peer comparison unavailable"
        else:
            evidence = f"{peer_difference:+.2f} °C vs peers"
        priority = "Inspect first" if rank == 1 else "Next candidate" if rank == 2 else "Review if needed"
        css_class = "ranking-card ranking-first" if rank == 1 else "ranking-card ranking-second" if rank == 2 else "ranking-card"
        if not available:
            css_class += " ranking-unavailable"
        # Emitted without leading whitespace: Markdown treats 4+ space indentation as a
        # code block, which would render the cards as literal HTML text.
        cards.append(
            f'<article class="{css_class}" aria-label="Rank {rank}, Car {escape(str(car))}">'
            f'<div class="ranking-card-top"><span class="ranking-number">{rank}</span>'
            f'<span class="ranking-priority">{priority}</span></div>'
            f'<div class="ranking-car">Car {escape(str(car))}</div>'
            f'<div class="ranking-evidence">{escape(evidence)}</div>'
            f'</article>'
        )
    st.markdown('<div class="ranking-board">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render() -> None:
    page_header(
        eyebrow="Air conditioning",
        title="Ranking train cars with faulty ACV",
        description="Compares all eight cars over the same period to find the one that has lost cooling power, and tells you which to check first.",
        tags=("Compares all 8 cars", "Points to the faulty car"),
    )

    artifact = ROOT / "artifacts/acv/final"
    if not (artifact / "manifest.json").exists():
        st.info("The trained ACV model is not available yet. Complete model evaluation and final training before analysis.")
        st.stop()

    @st.cache_resource
    def model(path, stamp):
        return load_artifact(path)

    bundle = model(str(artifact), (artifact / "manifest.json").stat().st_mtime_ns)
    manifest = json.loads((artifact / "manifest.json").read_text())
    upload_intro("Upload air-conditioning data", "Drop in one or more Excel files. Each should cover all eight cars over the same time period.")
    with st.container():
        uploads = st.file_uploader(
            "Upload ACV telemetry",
            type=["xlsx"],
            accept_multiple_files=True,
            key="acv_upload",
            help="Each workbook should contain one telemetry sheet with Time and Car <ID> columns.",
        )
        uploads = uploaded_files_list(uploads, "acv")
        with st.expander("Try a supplied training case"):
            options = sorted((ROOT / "PS3/02_Datasets/ACV/Train").glob("*.xlsx"))
            demo = st.selectbox("Training case", options, format_func=lambda p: p.name) if options else None
            demo_clicked = st.button("Analyze training example", disabled=demo is None)
        run = st.button(
            "Analyze uploaded files",
            type="primary",
            disabled=not uploads,
            width="stretch",
            help=None if uploads else "Add at least one workbook above to enable analysis.",
        )

    if run or demo_clicked:
        sources = [(p.name, p) for p in uploads] if run else [(demo.name, demo)]
        if len({n for n, _ in sources}) != len(sources):
            st.error("Choose files with distinct filenames.")
            st.stop()
        results = []
        progress = st.progress(0, text="Reading telemetry and comparing cars…")
        try:
            for i, (name, source) in enumerate(sources):
                with st.spinner(f"Analyzing {name}…"):
                    results.append(rank_case(source, bundle, file_id=name))
                progress.progress((i + 1) / len(sources), text=f"Analyzed {name}")
            st.session_state.acv_results = results
            st.session_state.acv_stabilities = {}
            st.session_state.acv_is_demo = not run
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"Unable to analyze this workbook: {exc}")
            st.stop()

    # Only uploaded files feed the submission export; the training demo must never
    # overwrite it, or a training-case prediction ships as the held-out answer.
    if run:
        try:
            export = ROOT / "outputs/acv/app_export"
            export.mkdir(parents=True, exist_ok=True)
            csv = predictions_csv(results)
            (export / "acv_predictions.csv").write_bytes(csv)
            (export / "predictions.zip").write_bytes(prediction_zip(csv))
            save_json(
                export / "provenance.json",
                {
                    "source": "streamlit_app",
                    "input_hashes": {r.file_id: r.features.case.metadata["source_hash"] for r in results},
                    "model_sha256": manifest["model_sha256"],
                },
            )
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"Unable to analyze this workbook: {exc}")
            st.stop()

    if "acv_results" not in st.session_state:
        st.stop()
    if st.session_state.get("acv_is_demo"):
        st.warning(
            "These are training-case results, shown for demonstration. They are not written to the "
            "submission export - upload the held-out test workbook to generate submission predictions.",
            icon=":material/science:",
        )

    results = st.session_state.acv_results
    csv = predictions_csv(results)
    triage = []
    for result in results:
        decision = _operator_decision(result)
        triage.append({"File": result.file_id, "Severity": decision["severity"], "Verdict": decision["verdict"], "Confidence": decision["confidence"], "Action": decision["action"]})
    triage.sort(key=lambda row: SEVERITY_ORDER[row["Severity"]])
    section_intro("What to do now", "The car to check first and what to do about it. Open the tabs further down only if you want the detail.")
    if len(triage) > 1:
        st.dataframe(pd.DataFrame(triage), hide_index=True, width="stretch")

    selected = st.selectbox("Analyzed file", [row["File"] for row in triage])
    r = next(r for r in results if r.file_id == selected)
    decision = _operator_decision(r)
    top = decision["top"]
    with st.container(border=True):
        severity_badge(decision["severity"])
        st.subheader(decision["verdict"])
        st.write(f"**Do now:** {decision['action']}")
        st.write(f"**Decision confidence:** {decision['confidence']}")
        st.caption("Every car is always ranked, even when nothing is wrong. If the top car is only slightly ahead, that is shown as uncertain rather than a confirmed fault.")

    a, b, c = st.columns(3)
    a.metric("First car to inspect", top)
    b.metric("Cars ranked", len(r.ranked_cars))
    c.metric("Usable cooling data · top car", f"{r.table.loc[top, 'usable_hours']:.1f} hours")
    section_intro("Which car to check first", "All eight cars, most likely problem at the top. Cars with missing data are marked as such - they are not assumed to be fine.")
    _ranking_board(r)
    for warning in r.warnings:
        st.caption(warning)
    left, right = st.columns(2)
    left.download_button(
        "Download prediction CSV",
        csv,
        "acv_predictions.csv",
        "text/csv",
        icon=":material/download:",
        width="stretch",
    )
    right.download_button(
        "Download predictions.zip",
        prediction_zip(csv),
        "predictions.zip",
        "application/zip",
        icon=":material/download:",
        width="stretch",
    )

    ranking_tab, evidence_tab, quality_tab = st.tabs([f"Why Car {top} ranks first", "Cooling evidence", "Data quality"])
    with ranking_tab:
        st.subheader("Cooling performance compared with peer cars")
        chart = r.table.reset_index().rename(columns={"car": "Car", "peer_mean": "Temperature above peers (°C)"})
        chart["Status"] = chart["Car"].map(lambda car: "Inspect first" if car == top else "Other cars")
        fig = px.bar(
            chart,
            x="Temperature above peers (°C)",
            y="Car",
            orientation="h",
            color="Status",
            color_discrete_map={"Inspect first": "#d1495b", "Other cars": "#b4cad9"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#142d41")
        fig.update_yaxes(type="category", categoryorder="array", categoryarray=r.ranked_cars[::-1])
        fig.update_layout(height=350, margin=dict(t=10, b=20), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        st.write(
            f"Car {top} averaged {decision['peer_difference']:+.2f} °C relative to cars operating in a compatible cooling state and exceeded peers by more than 0.5 °C during {decision['persistence']:.0%} of usable cooling observations."
        )
        with st.expander("Technical ranking table"):
            view = r.table[["rank", "score", "evidence", "usable_hours", "err_mean", "peer_mean"]].rename(
                columns={
                    "rank": "Rank",
                    "score": "Relative evidence",
                    "evidence": "Data status",
                    "usable_hours": "Cooling hours",
                    "err_mean": "Mean setpoint error (°C)",
                    "peer_mean": "Mean peer difference (°C)",
                }
            )
            st.dataframe(view.round(3), width="stretch")
        if st.button("Check ranking stability"):
            with st.spinner("Resampling synchronized one-hour blocks…"):
                st.session_state.acv_stabilities[r.file_id] = bootstrap_stability(r.features, bundle)
        stable = st.session_state.acv_stabilities.get(r.file_id)
        if stable:
            st.caption(stable["method"] + ". These figures show how often each car came top when the data was resampled - they are not fault probabilities.")
            st.dataframe(pd.DataFrame(stable["cars"]).T, width="stretch")
    with evidence_tab:
        cars = st.multiselect("Cars to compare", r.ranked_cars, default=r.ranked_cars[:3])
        f = r.features.case.frame
        plot = f.loc[f.car.isin(cars)].reset_index()
        plot = plot.groupby([pd.Grouper(key="time", freq="5min"), "car"])[["cabin", "target", "peer"]].mean().reset_index()
        st.plotly_chart(
            px.line(plot, x="time", y="cabin", color="car", labels={"time": "Time", "cabin": "Cabin temperature (°C)", "car": "Car"}), width="stretch"
        )
        st.plotly_chart(
            px.line(plot, x="time", y="peer", color="car", labels={"time": "Time", "peer": "Setpoint error above peers (°C)", "car": "Car"}), width="stretch"
        )
        st.caption("A positive number means that car ran warmer than the others while cooling. Gaps mean no usable data for that period.")
        if r.features.case.metadata["schema"] == "rich":
            st.write("**Pressure and compressor diagnostics**")
            st.caption("Background detail only. The units of these pressure readings are not stated in the data, and they do not affect the ranking.")
            st.dataframe(r.table[["pressure_low_asymmetry", "pressure_high_asymmetry", "compressor_duty_imbalance", "both_compressors_hours"]], width="stretch")
    with quality_tab:
        st.dataframe(r.table[["evidence", "missing_fraction", "invalid_fraction", "peer_coverage", "usable_hours"]], width="stretch")
        st.write(
            "Missing sensors, invalid observations, and unavailable cars are tracked separately. Unavailable telemetry does not establish that a car is healthy."
        )
        st.caption("Checked against the six cases we have answers for. How it performs on the unseen test case is not known.")
        metrics = manifest["selection"]["validation_summary"]
        st.write(f"Nested development rank score: {metrics['primary_metric']:.4f} · Top-1: {metrics['top1_accuracy']:.1%}")
