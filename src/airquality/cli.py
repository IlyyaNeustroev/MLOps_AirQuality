"""Точки входа командной строки (регистрируются в pyproject.toml → [project.scripts]).

airquality-train --data data/air_quality_training_dataset_2.csv [--full]
airquality-serve-ml        # ML-сервис, порт 5001
airquality-serve-web       # веб-сервер, порт 5000
"""

import argparse
import logging
from pathlib import Path

from airquality.config import get_settings
from airquality.logging_config import setup_logging

logger = logging.getLogger(__name__)


def train(argv: list[str] | None = None) -> None:
    """Офлайн-обучение модели из CSV (AutoML)."""
    parser = argparse.ArgumentParser(description="Обучение модели IAQ")
    parser.add_argument("--data", type=Path, required=True, help="путь к CSV с колонкой iaq_class")
    parser.add_argument("--full", action="store_true", help="полный AutoML вместо FAST_MODE")
    parser.add_argument("--iter", type=int, default=None, help="число итераций RandomizedSearchCV")
    args = parser.parse_args(argv)

    import pandas as pd  # noqa: PLC0415 — тяжёлый импорт только при обучении

    from airquality.ml.predictor import AirQualityPredictor  # noqa: PLC0415

    settings = get_settings()
    setup_logging(settings.log_level)
    if args.full:
        settings = settings.model_copy(update={"fast_mode": False})

    logger.info("Чтение данных: %s", args.data)
    df = pd.read_csv(args.data)
    result = AirQualityPredictor(settings, autoload=False).fit(df, automl_iter=args.iter)
    logger.info("Результат: %s", result)


def serve_ml() -> None:
    """ML-сервис через waitress (работает и на Windows, и в Linux)."""
    from waitress import serve  # noqa: PLC0415

    from airquality.ml_service.app import create_app  # noqa: PLC0415

    settings = get_settings()
    setup_logging(settings.log_level)
    serve(create_app(settings), host="0.0.0.0", port=settings.ml_port)  # noqa: S104


def serve_web() -> None:
    """Веб-сервер через waitress."""
    from waitress import serve  # noqa: PLC0415

    from airquality.web.app import create_app  # noqa: PLC0415

    settings = get_settings()
    setup_logging(settings.log_level)
    serve(create_app(settings), host="0.0.0.0", port=settings.web_port)  # noqa: S104


if __name__ == "__main__":
    train()
