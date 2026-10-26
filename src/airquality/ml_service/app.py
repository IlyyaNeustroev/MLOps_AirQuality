"""Фабрика Flask-приложения ML-сервиса."""

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from flask import Blueprint, Flask, current_app, jsonify, request
from flask.typing import ResponseReturnValue
from pydantic import ValidationError

from airquality.config import Settings, get_settings
from airquality.db.repository import fetch_training_data
from airquality.ml.predictor import AirQualityPredictor, ModelNotLoadedError
from airquality.ml_service.schemas import (
    DriftRequest,
    RetrainRequest,
    SensorReading,
    resolve_data_file,
)

logger = logging.getLogger(__name__)
bp = Blueprint("ml", __name__)

INDEX_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ML Air Quality</title></head><body>
<h1>ML Air Quality Predictor</h1>
<ul>
<li>POST /predict — прогноз IAQ</li>
<li>GET /status — статус модели</li>
<li>POST /retrain — переобучение из БД или CSV</li>
<li>POST /monitoring/drift — отчёт по дрейфу</li>
<li>GET /health, GET /monitoring/health — health-check</li>
</ul></body></html>"""


def _predictor() -> AirQualityPredictor:
    predictor: AirQualityPredictor = current_app.extensions["predictor"]
    return predictor


def _settings() -> Settings:
    settings: Settings = current_app.extensions["settings"]
    return settings


def _json_body() -> dict[str, Any]:
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else {}


@bp.get("/")
def index() -> ResponseReturnValue:
    return INDEX_HTML


@bp.post("/predict")
def predict() -> ResponseReturnValue:
    reading = SensorReading.model_validate(_json_body())
    return jsonify(_predictor().predict(reading.model_dump()))


@bp.post("/retrain")
def retrain() -> ResponseReturnValue:
    req = RetrainRequest.model_validate(_json_body())
    settings = _settings()
    if req.csv_file:
        df = pd.read_csv(resolve_data_file(settings.data_dir, req.csv_file))
    else:
        df = fetch_training_data(settings, req.limit)
        if len(df) < settings.min_retrain_rows:
            return jsonify({"error": "Недостаточно данных в БД", "rows": len(df)}), 409
    return jsonify(_predictor().fit(df))


@bp.get("/status")
def status() -> ResponseReturnValue:
    return jsonify(_predictor().health())


@bp.post("/monitoring/drift")
def monitoring_drift() -> ResponseReturnValue:
    req = DriftRequest.model_validate(_json_body())
    data_dir = _settings().data_dir
    current_df = pd.read_csv(resolve_data_file(data_dir, req.current_csv))
    baseline_df = pd.read_csv(resolve_data_file(data_dir, req.baseline_csv))
    return jsonify({"features": _predictor().drift_report(current_df, baseline_df)})


@bp.get("/health")
@bp.get("/monitoring/health")
def health() -> ResponseReturnValue:
    return jsonify(
        {
            "status": "ok",
            "model_loaded": _predictor().is_loaded,
            "timestamp": datetime.now().isoformat(),
        }
    )


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ValidationError)
    def _validation(err: ValidationError) -> ResponseReturnValue:
        return jsonify({"error": "validation_error", "details": err.errors(include_url=False)}), 400

    @app.errorhandler(ModelNotLoadedError)
    def _no_model(err: ModelNotLoadedError) -> ResponseReturnValue:
        return jsonify({"error": str(err)}), 503

    @app.errorhandler(FileNotFoundError)
    def _not_found(err: FileNotFoundError) -> ResponseReturnValue:
        return jsonify({"error": str(err)}), 404

    @app.errorhandler(ValueError)
    def _bad_value(err: ValueError) -> ResponseReturnValue:
        return jsonify({"error": str(err)}), 400


def create_app(
    settings: Settings | None = None, predictor: AirQualityPredictor | None = None
) -> Flask:
    """Создаёт приложение. В тестах можно передать свои settings и predictor."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    app = Flask(__name__)
    app.json.ensure_ascii = False  # type: ignore[attr-defined]
    app.extensions["settings"] = settings
    app.extensions["predictor"] = predictor or AirQualityPredictor(settings)
    app.register_blueprint(bp)
    _register_error_handlers(app)
    return app
