"""Очистка входных признаков перед подачей в модель."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from airquality.constants import DEFAULT_MEANS, FEATURE_COLUMNS, MISSING_SENSOR_VALUE


def preprocess_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    defaults: Mapping[str, float] = DEFAULT_MEANS,
) -> pd.DataFrame:
    """Приводит признаки к числам и заменяет -1 / пропуски на типичные значения.

    Args:
        df: сырые данные; лишние столбцы отбрасываются, недостающие добавляются.
        feature_columns: порядок признаков, ожидаемый моделью.
        defaults: значения для подстановки.

    Returns:
        Новый DataFrame только со столбцами ``feature_columns``.
    """
    result = df.reindex(columns=list(feature_columns))
    for col in feature_columns:
        numeric = pd.to_numeric(result[col], errors="coerce")
        numeric = numeric.replace(MISSING_SENSOR_VALUE, np.nan)
        result[col] = numeric.fillna(defaults.get(col, 0.0)).astype(float)
    return result
