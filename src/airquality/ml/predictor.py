"""AirQualityPredictor — фасад над моделью: загрузка, предсказание, переобучение."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, TypedDict

import pandas as pd

from airquality.constants import FEATURE_COLUMNS, RECOMMENDATIONS, UNKNOWN_RECOMMENDATION
from airquality.ml.drift import drift_report, save_drift_report
from airquality.ml.preprocessing import preprocess_features
from airquality.ml.registry import load_model, save_model
from airquality.ml.reports import write_training_report
from airquality.ml.tracking import ExperimentTracker
from airquality.ml.training import TrainingConfig, train_model

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from airquality.config import Settings

logger = logging.getLogger(__name__)


class PredictionResult(TypedDict):
    iaq_class: int
    probabilities: dict[int, float]
    recommendation: str


class ModelNotLoadedError(RuntimeError):
    """Модель ещё не обучена или файл артефакта отсутствует."""


def get_recommendation(iaq_class: int) -> str:
    """Текстовая рекомендация по классу IAQ."""
    return RECOMMENDATIONS.get(int(iaq_class), UNKNOWN_RECOMMENDATION)


class AirQualityPredictor:
    """Потокобезопасная обёртка над обученным sklearn-пайплайном."""

    def __init__(self, settings: Settings, *, autoload: bool = True) -> None:
        self.settings = settings
        self.feature_columns = list(FEATURE_COLUMNS)
        self.tracker = ExperimentTracker(settings.monitoring_dir)
        self._model: Pipeline | None = None
        self._metadata: dict[str, Any] | None = None
        # Переобучение подменяет модель, пока другие потоки делают predict.
        self._lock = threading.RLock()
        if autoload:
            self.load()

    # ---------- состояние ----------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> dict[str, Any] | None:
        return self._metadata

    def load(self) -> bool:
        model, metadata = load_model(self.settings.model_path, self.settings.metadata_path)
        with self._lock:
            self._model, self._metadata = model, metadata
        return model is not None

    # ---------- предсказание ----------
    def predict(self, sensor_data: dict[str, Any]) -> PredictionResult:
        """Предсказывает класс IAQ (0–5) и вероятности классов."""
        with self._lock:
            model = self._model
        if model is None:
            raise ModelNotLoadedError("Модель не обучена")

        x = preprocess_features(pd.DataFrame([sensor_data]), self.feature_columns)
        proba = model.predict_proba(x)[0]
        classes = model.classes_
        iaq_class = int(classes[proba.argmax()])
        result: PredictionResult = {
            "iaq_class": iaq_class,
            "probabilities": {
                int(c): round(float(p), 3) for c, p in zip(classes, proba, strict=True)
            },
            "recommendation": get_recommendation(iaq_class),
        }
        self.tracker.log_prediction(
            {str(k): v for k, v in x.iloc[0].items()},
            iaq_class,
            max(result["probabilities"].values()),
        )
        return result

    # ---------- обучение ----------
    def fit(self, df: pd.DataFrame, automl_iter: int | None = None) -> dict[str, Any]:
        """Обучает модель, сохраняет артефакты, пишет отчёты и журнал экспериментов."""
        config = TrainingConfig(
            random_state=self.settings.random_state,
            fast_mode=self.settings.fast_mode,
            automl_iter=automl_iter or self.settings.automl_iter,
        )
        result = train_model(df, config)
        meta = result.metadata

        self.settings.ensure_dirs()
        save_model(result.model, meta, self.settings.model_path, self.settings.metadata_path)
        write_training_report(
            self.settings.reports_dir,
            meta.best_params,
            result.best_cv_score,
            result.y_test,
            result.y_test_pred,
        )
        metrics = {
            "train_accuracy": meta.train_accuracy,
            "test_accuracy": meta.test_accuracy,
            "train_f1_macro": meta.train_f1_macro,
            "test_f1_macro": meta.test_f1_macro,
        }
        self.tracker.log_experiment(
            f"air_quality_{meta.version}", len(df), meta.best_params, metrics
        )

        with self._lock:
            self._model, self._metadata = result.model, meta.to_dict()

        return {
            "status": "trained",
            "version": meta.version,
            "best_params": meta.best_params,
            "best_cv_score": result.best_cv_score,
            "mode": "FAST" if config.fast_mode else "FULL",
            **metrics,
        }

    # ---------- мониторинг ----------
    def drift_report(
        self, current_df: pd.DataFrame, baseline_df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        rows = drift_report(current_df, baseline_df, self.feature_columns)
        save_drift_report(rows, self.settings.monitoring_dir)
        return rows

    def health(self) -> dict[str, Any]:
        return {
            "model_loaded": self.is_loaded,
            "metadata": self._metadata,
            "features": self.feature_columns,
            "fast_mode": self.settings.fast_mode,
        }
