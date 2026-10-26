"""Проверка, что обучение пишет все артефакты мониторинга."""

import json

import pytest

from airquality.config import Settings
from airquality.ml.predictor import AirQualityPredictor
from airquality.ml.synthetic import generate_sensor_data

pytestmark = pytest.mark.slow


def test_fit_writes_artifacts(empty_settings: Settings) -> None:
    p = AirQualityPredictor(empty_settings, autoload=False)
    result = p.fit(generate_sensor_data(n=120), automl_iter=1)

    assert result["status"] == "trained"
    assert empty_settings.model_path.exists()
    meta = json.loads(empty_settings.metadata_path.read_text(encoding="utf-8"))
    assert meta["version"] == result["version"]
    assert "sklearn_version" in meta
    assert list(empty_settings.reports_dir.glob("train_report_*.txt"))
    assert list(empty_settings.reports_dir.glob("train_report_*.json"))
    assert (empty_settings.monitoring_dir / "experiments.jsonl").exists()

    # модель перезагружается с диска
    assert AirQualityPredictor(empty_settings).is_loaded


def test_predict_writes_predictions_log(predictor: AirQualityPredictor) -> None:
    predictor.predict({"temp": 22.0})
    assert predictor.tracker.predictions_path.exists()


def test_same_seed_same_metrics(
    empty_settings: Settings, tmp_path_factory: pytest.TempPathFactory
) -> None:
    df = generate_sensor_data(n=120)
    r1 = AirQualityPredictor(empty_settings, autoload=False).fit(df, automl_iter=1)
    other = empty_settings.model_copy(update={"artifacts_dir": tmp_path_factory.mktemp("a2")})
    r2 = AirQualityPredictor(other, autoload=False).fit(df, automl_iter=1)
    assert r1["test_f1_macro"] == r2["test_f1_macro"]
