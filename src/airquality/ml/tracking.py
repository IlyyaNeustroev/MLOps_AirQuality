"""Журналы экспериментов и предсказаний (JSON Lines) + опционально MLflow."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:  # MLflow — необязательная зависимость (poetry install --with tracking)
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - зависит от окружения
    MLFLOW_AVAILABLE = False


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class ExperimentTracker:
    """Пишет ``experiments.jsonl`` и ``predictions.jsonl`` в каталог мониторинга."""

    def __init__(self, monitoring_dir: Path) -> None:
        self.monitoring_dir = monitoring_dir

    @property
    def experiments_path(self) -> Path:
        return self.monitoring_dir / "experiments.jsonl"

    @property
    def predictions_path(self) -> Path:
        return self.monitoring_dir / "predictions.jsonl"

    def log_experiment(
        self, run_name: str, rows: int, params: dict[str, Any], metrics: dict[str, float]
    ) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "rows": rows,
            "best_params": params,
            **metrics,
        }
        _append_jsonl(self.experiments_path, record)
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
            except Exception:  # MLflow не должен ронять обучение
                logger.exception("Не удалось записать эксперимент в MLflow")

    def log_prediction(self, features: dict[str, Any], iaq_class: int, max_proba: float) -> None:
        record = {
            **features,
            "iaq_class_pred": iaq_class,
            "max_proba": max_proba,
            "timestamp": datetime.now().isoformat(),
        }
        _append_jsonl(self.predictions_path, record)
