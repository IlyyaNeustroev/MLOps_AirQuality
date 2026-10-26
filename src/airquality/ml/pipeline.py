"""Построение sklearn-пайплайна и пространства гиперпараметров для AutoML."""

from collections.abc import Sequence
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(feature_columns: Sequence[str], random_state: int = 42) -> Pipeline:
    """Импутация + масштабирование + RandomForest в одном объекте.

    Скалер хранится внутри пайплайна, поэтому модель и скалер не могут разойтись.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(feature_columns),
            )
        ]
    )
    clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    return Pipeline([("preprocessor", preprocessor), ("clf", clf)])


def param_space(fast_mode: bool) -> dict[str, list[Any]]:
    """Пространство поиска для RandomizedSearchCV."""
    if fast_mode:
        return {
            "clf__n_estimators": [80, 120],
            "clf__max_depth": [10, 15],
            "clf__min_samples_split": [2, 4],
            "clf__min_samples_leaf": [1, 2],
            "clf__max_features": ["sqrt"],
            "clf__bootstrap": [True],
        }
    return {
        "clf__n_estimators": [100, 150, 200],
        "clf__max_depth": [10, 15, 20],
        "clf__min_samples_split": [2, 4],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt", "log2"],
        "clf__bootstrap": [True, False],
    }
