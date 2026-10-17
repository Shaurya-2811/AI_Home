"""The Streamlit app starts and renders its controls without errors (headless AppTest)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_app_starts():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Human detection and tracking"
    labels = [s.label for s in at.sidebar.selectbox]
    assert "Tracker" in labels and "Detector checkpoint" in labels


def test_save_upload_validation(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    import importlib

    app = importlib.import_module("app")
    monkeypatch.setattr(app, "UPLOADS", tmp_path)

    class Up:
        def __init__(self, name, data):
            self.name, self._d, self.size = name, data, len(data)

        def getbuffer(self):
            return memoryview(self._d)

    with pytest.raises(ValueError, match="Unsupported"):
        app.save_upload(Up("x.exe", b"abc"), app.BASE_CFG)
    with pytest.raises(ValueError, match="empty"):
        app.save_upload(Up("x.mp4", b""), app.BASE_CFG)
    p = app.save_upload(Up("../../evil name.mp4", b"abc"), app.BASE_CFG)
    assert p.parent == tmp_path and "evil" not in p.name and p.suffix == ".mp4"
