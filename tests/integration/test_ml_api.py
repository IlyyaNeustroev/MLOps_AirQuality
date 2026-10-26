from pathlib import Path
from typing import Any

from flask.testing import FlaskClient

from airquality.config import Settings
from airquality.ml.synthetic import generate_sensor_data
from airquality.ml_service.app import create_app


def test_status(ml_client: FlaskClient) -> None:
    data = ml_client.get("/status").get_json()
    assert data["model_loaded"] is True
    assert len(data["features"]) == 16


def test_predict(ml_client: FlaskClient, sample_input: dict[str, Any]) -> None:
    resp = ml_client.post("/predict", json=sample_input)
    assert resp.status_code == 200
    assert {"iaq_class", "probabilities", "recommendation"} <= resp.get_json().keys()


def test_predict_validation_error(ml_client: FlaskClient) -> None:
    resp = ml_client.post("/predict", json={"season": 99})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "validation_error"


def test_health(ml_client: FlaskClient) -> None:
    for url in ("/health", "/monitoring/health"):
        assert ml_client.get(url).get_json()["status"] == "ok"


def test_predict_without_model_returns_503(empty_settings: Settings) -> None:
    client = create_app(empty_settings).test_client()
    assert client.post("/predict", json={}).status_code == 503


def test_drift_endpoint(ml_client: FlaskClient, trained_settings: Settings) -> None:
    data_dir: Path = trained_settings.data_dir
    generate_sensor_data(50, seed=1).to_csv(data_dir / "cur.csv", index=False)
    generate_sensor_data(50, seed=2).to_csv(data_dir / "base.csv", index=False)
    resp = ml_client.post(
        "/monitoring/drift", json={"current_csv": "cur.csv", "baseline_csv": "base.csv"}
    )
    assert resp.status_code == 200
    assert len(resp.get_json()["features"]) == 16
    assert list(trained_settings.monitoring_dir.glob("drift_*.csv"))


def test_drift_rejects_path_traversal(ml_client: FlaskClient) -> None:
    resp = ml_client.post(
        "/monitoring/drift", json={"current_csv": "../../etc/passwd", "baseline_csv": "x.csv"}
    )
    assert resp.status_code == 400
