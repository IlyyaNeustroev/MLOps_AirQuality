"""Общие фикстуры.

Тесты не трогают настоящие artifacts/ и не требуют 18-МБ модели из репозитория:
маленькая модель обучается один раз на синтетических данных во временном каталоге.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from flask.testing import FlaskClient

from airquality.config import Settings
from airquality.ml.predictor import AirQualityPredictor
from airquality.ml.synthetic import generate_sensor_data
from airquality.ml_service.app import create_app


def make_settings(base: Path) -> Settings:
    return Settings(
        _env_file=None,  # не читать локальный .env
        data_dir=base / "data",
        artifacts_dir=base / "artifacts",
        reports_dir=base / "reports",
        monitoring_dir=base / "monitoring",
        fast_mode=True,
        automl_iter=1,
    )


@pytest.fixture(scope="session")
def trained_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    settings = make_settings(tmp_path_factory.mktemp("ml"))
    settings.ensure_dirs()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    AirQualityPredictor(settings, autoload=False).fit(generate_sensor_data(n=200), automl_iter=1)
    return settings


@pytest.fixture
def predictor(trained_settings: Settings) -> AirQualityPredictor:
    return AirQualityPredictor(trained_settings)


@pytest.fixture
def empty_settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def ml_client(trained_settings: Settings) -> Iterator[FlaskClient]:
    app = create_app(trained_settings)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_input() -> dict[str, Any]:
    return {
        "season": 2, "weekday": 3, "temp": 22.5, "hum": 45.0, "mq7": 120, "mq135": 110,
        "ky028_analog": 35, "ky028_digital": 0, "bmp_temp": 15.0, "pressure": 1013.0,
        "altitude": 82.0, "aht21_temp": 22.0, "aht21_hum": 45.0, "ens_iaq": 75,
        "ens_tvoc": 250, "ens_co2": 800,
    }  # fmt: skip
