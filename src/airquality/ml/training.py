"""Обучение модели: разбиение, AutoML (RandomizedSearchCV), метрики, метаданные."""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from airquality.constants import FEATURE_COLUMNS, TARGET_COLUMN
from airquality.ml.pipeline import build_pipeline, param_space
from airquality.ml.preprocessing import preprocess_features
from airquality.ml.registry import ModelMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    fast_mode: bool = True
    automl_iter: int = 3


@dataclass(frozen=True)
class TrainingResult:
    model: Pipeline
    metadata: ModelMetadata
    best_cv_score: float
    y_test: pd.Series
    y_test_pred: Any


def hash_dataframe(df: pd.DataFrame) -> str:
    """SHA-256 содержимого DataFrame — для отслеживания версии данных."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    ).hexdigest()


def train_model(df: pd.DataFrame, config: TrainingConfig) -> TrainingResult:
    """Обучает пайплайн с подбором гиперпараметров и считает метрики."""
    if TARGET_COLUMN not in df.columns:
        msg = f"Нет целевого столбца {TARGET_COLUMN}"
        raise ValueError(msg)

    data = df.dropna(subset=[TARGET_COLUMN]).copy()
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").astype(int)

    x = preprocess_features(data)
    y = data[TARGET_COLUMN]
    stratify = y if y.value_counts().min() >= 2 else None  # noqa: PLR2004
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_size, random_state=config.random_state, stratify=stratify
    )

    search = RandomizedSearchCV(
        estimator=build_pipeline(FEATURE_COLUMNS, config.random_state),
        param_distributions=param_space(config.fast_mode),
        n_iter=config.automl_iter if config.fast_mode else 5,
        scoring="f1_macro",
        cv=2 if config.fast_mode else 3,
        n_jobs=1,
        random_state=config.random_state,
        refit=True,
    )
    logger.info("Старт AutoML: %d строк, fast_mode=%s", len(data), config.fast_mode)
    search.fit(x_train, y_train)
    model: Pipeline = search.best_estimator_

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    now = datetime.now()
    metadata = ModelMetadata(
        version=f"{now:%Y%m%d_%H%M%S}",
        trained_at=now.isoformat(),
        features=list(FEATURE_COLUMNS),
        best_params=dict(search.best_params_),
        train_rows=len(x_train),
        test_rows=len(x_test),
        train_accuracy=float(accuracy_score(y_train, y_train_pred)),
        test_accuracy=float(accuracy_score(y_test, y_test_pred)),
        train_f1_macro=float(f1_score(y_train, y_train_pred, average="macro", zero_division=0)),
        test_f1_macro=float(f1_score(y_test, y_test_pred, average="macro", zero_division=0)),
        data_hash=hash_dataframe(data[[*FEATURE_COLUMNS, TARGET_COLUMN]]),
    )
    logger.info("Обучение завершено: test_f1_macro=%.4f", metadata.test_f1_macro)
    return TrainingResult(model, metadata, float(search.best_score_), y_test, y_test_pred)
