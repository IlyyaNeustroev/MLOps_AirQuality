"""Сохранение и загрузка артефактов модели вместе с метаданными."""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelMetadata:
    """Паспорт обученной модели: что, когда и на каких данных обучено."""

    version: str
    trained_at: str
    features: list[str]
    best_params: dict[str, Any]
    train_rows: int
    test_rows: int
    train_accuracy: float
    test_accuracy: float
    train_f1_macro: float
    test_f1_macro: float
    data_hash: str
    sklearn_version: str = field(default=sklearn.__version__)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_model(
    model: Pipeline, metadata: ModelMetadata, model_path: Path, metadata_path: Path
) -> None:
    """Сохраняет пайплайн (joblib) и метаданные (JSON)."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Модель сохранена: %s (version=%s)", model_path, metadata.version)


def load_model(
    model_path: Path, metadata_path: Path
) -> tuple[Pipeline | None, dict[str, Any] | None]:
    """Загружает модель и метаданные. Если файла модели нет — возвращает ``(None, None)``."""
    if not model_path.exists():
        logger.warning("Файл модели не найден: %s", model_path)
        return None, None
    model: Pipeline = joblib.load(model_path)
    metadata: dict[str, Any] | None = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    logger.info("Модель загружена: %s", model_path)
    return model, metadata
