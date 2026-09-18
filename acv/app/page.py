from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
import plotly.express as px
import streamlit as st
from acv.common import save_json
from acv.inference import load_artifact, rank_case, predictions_csv, prediction_zip
from acv.robustness import bootstrap_stability


def render() -> None:
    st.title("Find the car that needs attention")
    st.write("Compare cooling performance across the train and review the evidence behind each car’s ranking.")

    artifact = ROOT / "artifacts/acv/final"
    if not (artifact / "manifest.json").exists():
        st.info("The trained ACV model is not available yet. Complete model evaluation and final training before analysis.")
        st.stop()

    @st.cache_resource
    def model(path, stamp):
        return load_artifact(path)

    bundle = model(str(artifact), (artifact / "manifest.json").stat().st_mtime_ns)
    manifest = json.loads((artifact / "manifest.json").read_text())
    uploads = st.file_uploader(
        "Upload ACV telemetry",
        type=["xlsx"],
        accept_multiple_files=True,
        help="Each workbook should contain one telemetry sheet with Time and Car <ID> columns.",
    )
    with st.expander("Try a supplied training case"):
        options = sorted((ROOT / "PS3/02_Datasets/ACV/Train").glob("*.xlsx"))
        demo = st.selectbox("Training case", options, format_func=lambda p: p.name) if options else None
        demo_clicked = st.button("Analyze training example", disabled=demo is None)
    run = st.button("Analyze uploaded files", type="primary", disabled=not uploads)

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
            # An explicit Analyze action is the source of submission exports.
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
        st.info("Upload an ACV workbook to see a complete car ranking and cooling-performance evidence.")
        st.stop()

    results = st.session_state.acv_results
    csv = predictions_csv(results)
    left, right = st.columns(2)
    left.download_button("Download prediction CSV", csv, "acv_predictions.csv", "text/csv", width="stretch")
    right.download_button("Download predictions.zip", prediction_zip(csv), "predictions.zip", "application/zip", width="stretch")
    selected = st.selectbox("Analyzed file", [r.file_id for r in results])
    r = next(r for r in results if r.file_id == selected)
    top = r.ranked_cars[0]
    a, b, c = st.columns(3)
    a.metric("First car to inspect", top)
    b.metric("Cars ranked", len(r.ranked_cars))
    c.metric("Usable cooling data · top car", f"{r.table.loc[top, 'usable_hours']:.1f} hours")
    for warning in r.warnings:
        st.caption(warning)

    ranking_tab, evidence_tab, quality_tab = st.tabs(["Car ranking", "Cooling evidence", "Data quality"])
    with ranking_tab:
        st.subheader("Inspection order")
        chart = r.table.reset_index().rename(columns={"car": "Car", "score": "Relative evidence"})
        fig = px.bar(chart, x="Relative evidence", y="Car", orientation="h", color="Relative evidence", color_continuous_scale=["#b4cad9", "#167d9a"])
        fig.update_yaxes(type="category", categoryorder="array", categoryarray=r.ranked_cars[::-1])
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350, margin=dict(t=10, b=20))
        st.plotly_chart(fig, width="stretch")
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
            st.caption(stable["method"] + ". Frequencies are not fault probabilities.")
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
        st.caption("Positive peer differences indicate warmer operation relative to cars in a compatible cooling state. Gaps indicate unavailable evidence.")
        if r.features.case.metadata["schema"] == "rich":
            st.write("**Pressure and compressor diagnostics**")
            st.caption("Diagnostic evidence only. Pressure units are unspecified; these channels do not affect the production ranking.")
            st.dataframe(r.table[["pressure_low_asymmetry", "pressure_high_asymmetry", "compressor_duty_imbalance", "both_compressors_hours"]], width="stretch")
    with quality_tab:
        st.dataframe(r.table[["evidence", "missing_fraction", "invalid_fraction", "peer_coverage", "usable_hours"]], width="stretch")
        st.write(
            "Missing sensors, invalid observations, and unavailable cars are tracked separately. Unavailable telemetry does not establish that a car is healthy."
        )
        st.caption("Validation uses six labeled cases. Hidden-test accuracy is unknown.")
        metrics = manifest["selection"]["validation_summary"]
        st.write(f"Nested development rank score: {metrics['primary_metric']:.4f} · Top-1: {metrics['top1_accuracy']:.1%}")
