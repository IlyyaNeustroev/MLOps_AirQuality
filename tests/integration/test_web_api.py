from typing import Any

import pytest
from flask.testing import FlaskClient

from airquality.config import Settings
from airquality.web import app as web_app
from airquality.web.app import create_app


class FakeML:
    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        return {"iaq_class": 2, "probabilities": {2: 1.0}, "recommendation": "🟡 Удовл."}


def _client(settings: Settings) -> FlaskClient:
    app = create_app(settings, ml_client=FakeML())  # type: ignore[arg-type]
    app.config["TESTING"] = True
    return app.test_client()


def test_health(empty_settings: Settings) -> None:
    assert _client(empty_settings).get("/health").get_json()["status"] == "healthy"


def test_receive_data(empty_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list[dict[str, Any]] = []
    monkeypatch.setattr(web_app, "save_measurement", lambda _s, record: saved.append(record))

    resp = _client(empty_settings).post(
        "/data", json={"room_id": 3, "date": "2026-10-26", "time": "10:00:00", "temp": 23.1}
    )
    body = resp.get_json()
    assert resp.status_code == 200
    assert body == {
        "status": "success",
        "packet_number": 1,
        "iaq_class": 2,
        "recommendation": "🟡 Удовл.",
    }
    assert saved[0]["room_id"] == 3
    assert saved[0]["season"] == 4
    assert saved[0]["hum"] == 0.0


def test_receive_invalid_json(empty_settings: Settings) -> None:
    resp = _client(empty_settings).post("/data", data="not json", content_type="application/json")
    assert resp.status_code == 400
