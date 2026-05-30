import os
import json
import pandas as pd
import numpy as np
import pytest

from air_quality_ml import AirQualityPredictor


@pytest.fixture(scope="session")
def predictor():
    p = AirQualityPredictor()
    return p


@pytest.fixture
def sample_input():
    return {
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


def test_model_loads(predictor):
    assert predictor is not None
    assert predictor.feature_columns is not None
    assert len(predictor.feature_columns) == 16


def test_predict_returns_expected_keys(predictor, sample_input):
    if predictor.model is None:
        pytest.skip("Model is not trained yet")
    result = predictor.predict(sample_input)
    assert "iaq_class" in result
    assert "probabilities" in result
    assert "recommendation" in result
    assert isinstance(result["iaq_class"], int)
    assert isinstance(result["probabilities"], dict)
    assert isinstance(result["recommendation"], str)


def test_predict_probability_sum(predictor, sample_input):
    if predictor.model is None:
        pytest.skip("Model is not trained yet")
    result = predictor.predict(sample_input)
    s = sum(result["probabilities"].values())
    assert 0.95 <= s <= 1.05


def test_predict_handles_missing_fields(predictor):
    if predictor.model is None:
        pytest.skip("Model is not trained yet")
    partial = {
        "season": 1,
        "temp": 21.0,
        "hum": 50.0,
    }
    result = predictor.predict(partial)
    assert "iaq_class" in result


def test_preprocess_replaces_minus_one(predictor):
    df = pd.DataFrame([{
        "season": 1,
        "weekday": 1,
        "temp": -1,
        "hum": -1,
        "mq7": -1,
        "mq135": -1,
        "ky028_analog": -1,
        "ky028_digital": -1,
        "bmp_temp": -1,
        "pressure": -1,
        "altitude": -1,
        "aht21_temp": -1,
        "aht21_hum": -1,
        "ens_iaq": -1,
        "ens_tvoc": -1,
        "ens_co2": -1,
    }])
    X = predictor.preprocess_features(df)
    assert (X.values != -1).all()


def test_get_recommendation_mapping(predictor):
    assert predictor.get_recommendation(0) == "✅ Отлично"
    assert predictor.get_recommendation(5) == "🚨 АВАРИЯ!"
    assert predictor.get_recommendation(999) == "Неизвестно"


def test_health_returns_structure(predictor):
    h = predictor.health()
    assert "model_loaded" in h
    assert "features" in h


def test_generate_test_data_shape(predictor):
    df = predictor.generate_test_data(n=25, with_target=True)
    assert len(df) == 25
    for col in predictor.feature_columns:
        assert col in df.columns
    assert "iaq_class" in df.columns


def test_drift_report_runs(predictor):
    baseline = predictor.generate_test_data(n=50, with_target=True)
    current = predictor.generate_test_data(n=50, with_target=True)
    out = predictor.drift_report(current, baseline)
    assert isinstance(out, list)


def test_save_and_load_roundtrip(tmp_path):
    p = AirQualityPredictor(
        model_path=str(tmp_path / "model.joblib"),
        scaler_path=str(tmp_path / "scaler.joblib"),
        metadata_path=str(tmp_path / "metadata.json"),
    )
    df = p.generate_test_data(n=120, with_target=True)
    result = p.fit(df, automl_iter=2)
    assert "status" in result
    p2 = AirQualityPredictor(
        model_path=str(tmp_path / "model.joblib"),
        scaler_path=str(tmp_path / "scaler.joblib"),
        metadata_path=str(tmp_path / "metadata.json"),
    )
    assert p2.model is not None