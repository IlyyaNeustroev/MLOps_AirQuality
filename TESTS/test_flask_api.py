import pytest
from air_quality_ml import create_app, predictor


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_status_endpoint(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "model_loaded" in data
    assert "features" in data


def test_predict_endpoint(client):
    payload = {
        "season": 2,
        "weekday": 3,
        "temp": 22.5,
        "hum": 45.0,
        "mq7": 120,
        "mq135": 110,
        "ky028_analog": 35,
        "ky028_digital": 0,
        "bmp_temp": 15.0,
        "pressure": 1013.0,
        "altitude": 82.0,
        "aht21_temp": 22.0,
        "aht21_hum": 45.0,
        "ens_iaq": 75,
        "ens_tvoc": 250,
        "ens_co2": 800,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "iaq_class" in data
    assert "probabilities" in data
    assert "recommendation" in data


def test_health_endpoint(client):
    resp = client.get("/monitoring/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"