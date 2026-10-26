from typing import Any

import pytest

from airquality.config import Settings
from airquality.ml.predictor import AirQualityPredictor, ModelNotLoadedError, get_recommendation


def test_recommendation_mapping() -> None:
    assert get_recommendation(0) == "✅ Отлично"
    assert get_recommendation(5) == "🚨 АВАРИЯ!"
    assert get_recommendation(999) == "Неизвестно"


def test_predict_structure(predictor: AirQualityPredictor, sample_input: dict[str, Any]) -> None:
    result = predictor.predict(sample_input)
    assert isinstance(result["iaq_class"], int)
    assert 0 <= result["iaq_class"] <= 5
    assert 0.95 <= sum(result["probabilities"].values()) <= 1.05
    assert isinstance(result["recommendation"], str)


def test_predict_with_partial_input(predictor: AirQualityPredictor) -> None:
    assert "iaq_class" in predictor.predict({"season": 1, "temp": 21.0})


def test_predict_without_model_raises(empty_settings: Settings) -> None:
    with pytest.raises(ModelNotLoadedError):
        AirQualityPredictor(empty_settings).predict({})


def test_health(predictor: AirQualityPredictor) -> None:
    health = predictor.health()
    assert health["model_loaded"] is True
    assert len(health["features"]) == 16
