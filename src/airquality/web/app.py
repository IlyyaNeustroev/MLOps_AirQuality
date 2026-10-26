"""Фабрика Flask-приложения основного веб-сервера."""

import logging
from datetime import datetime
from typing import Any

import psycopg2
from flask import Blueprint, Flask, current_app, jsonify, render_template, request
from flask.typing import ResponseReturnValue
from flask_cors import CORS

from airquality.config import Settings, get_settings
from airquality.constants import SEASON_NAMES, WEEKDAY_NAMES
from airquality.db.repository import build_measurement_record, fetch_recent_iaq, save_measurement
from airquality.ml.features import add_calendar_features
from airquality.web.dashboard import bp as dashboard_bp
from airquality.web.ml_client import MLServiceClient
from airquality.web.state import SensorState

logger = logging.getLogger(__name__)
bp = Blueprint("core", __name__)

RECENT_POINTS = 10


def _state() -> SensorState:
    state: SensorState = current_app.extensions["sensor_state"]
    return state


def _settings() -> Settings:
    settings: Settings = current_app.extensions["settings"]
    return settings


def _ml() -> MLServiceClient:
    client: MLServiceClient = current_app.extensions["ml_client"]
    return client


@bp.get("/")
def index() -> ResponseReturnValue:
    state = _state()
    data = state.latest()
    try:
        labels, values = fetch_recent_iaq(_settings(), RECENT_POINTS)
    except psycopg2.Error:
        logger.exception("Не удалось получить историю IAQ")
        labels, values = ["?"] * RECENT_POINTS, [1] * RECENT_POINTS
    page: dict[str, Any] = {
        **data,
        "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "packet_count": state.packet_count,
        "season_name": SEASON_NAMES.get(data["season"], "?"),
        "weekday_name": WEEKDAY_NAMES.get(data["weekday"], "?"),
        "recent_iaq_labels": labels,
        "recent_iaq_data": values,
    }
    return render_template("index.html", **page)


@bp.get("/api/data")
def get_data() -> ResponseReturnValue:
    state = _state()
    return jsonify(
        {
            **state.latest(),
            "timestamp": datetime.now().isoformat(),
            "packet_count": state.packet_count,
        }
    )


@bp.post("/data")
def receive_data() -> ResponseReturnValue:
    """Приём пакета от ESP32 → прогноз ML → запись в БД → ответ контроллеру."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "Invalid JSON"}), 400

    data = add_calendar_features(payload)
    try:
        room_id = int(data.get("room_id", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "room_id must be integer"}), 400

    state = _state()
    room_data, packet_number = state.update(room_id, data)

    iaq = _ml().predict(room_data)
    state.set_prediction(room_id, iaq["iaq_class"], iaq["recommendation"])

    record = build_measurement_record(
        room_data, iaq["iaq_class"], iaq.get("probabilities", {}), packet_number
    )
    try:
        save_measurement(_settings(), record)
    except psycopg2.Error:
        # Ответ ESP32 важнее записи в БД: логируем ошибку и продолжаем.
        logger.exception("Ошибка записи в БД (пакет #%d)", packet_number)

    logger.info("Пакет #%d, комната %d, IAQ=%s", packet_number, room_id, iaq["iaq_class"])
    return jsonify(
        {
            "status": "success",
            "packet_number": packet_number,
            "iaq_class": iaq["iaq_class"],
            "recommendation": iaq["recommendation"],
        }
    )


@bp.get("/health")
def health() -> ResponseReturnValue:
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


def create_app(settings: Settings | None = None, ml_client: MLServiceClient | None = None) -> Flask:
    """Создаёт приложение веб-сервера."""
    settings = settings or get_settings()
    app = Flask(__name__)
    app.json.ensure_ascii = False  # type: ignore[attr-defined]
    CORS(app)
    app.extensions["settings"] = settings
    app.extensions["sensor_state"] = SensorState()
    app.extensions["ml_client"] = ml_client or MLServiceClient(
        settings.ml_service_url, settings.ml_service_timeout
    )
    app.register_blueprint(bp)
    app.register_blueprint(dashboard_bp)
    return app
