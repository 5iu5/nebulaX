"""Exercise SHM through the shared Streamlit entry point."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_shared_app_runs_shm_model():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(ROOT / "app/main.py", default_timeout=90).run()
    subsystem = next(item for item in app.selectbox if item.label == "Subsystem")
    subsystem.select("SHM · Fatigue damage").run(timeout=90)
    assert not app.exception

    button = next(item for item in app.button if item.label == "Analyze supplied SHM signal")
    button.click().run(timeout=90)
    assert not app.exception
    assert app.metric[0].label == "Files analyzed"
    assert app.metric[0].value == "1"
    assert [item.label for item in app.get("download_button")] == ["Download SHM prediction CSV"]
