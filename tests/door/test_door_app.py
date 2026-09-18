"""Exercise the Door page through the shared Streamlit entry point."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_shared_app_runs_door_model():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(ROOT / "app/main.py", default_timeout=90).run()
    subsystem = next(item for item in app.selectbox if item.label == "Subsystem")
    subsystem.select("Door · Abnormal resistance").run()
    assert not app.exception

    button = next(item for item in app.button if item.label == "Analyze supplied Door stream")
    button.click().run(timeout=90)
    assert not app.exception
    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("Cycles detected", "38"),
        ("Abnormal resistance", "9"),
        ("Normal", "29"),
    ]
    assert [item.label for item in app.get("download_button")] == ["Download Door prediction CSV"]
