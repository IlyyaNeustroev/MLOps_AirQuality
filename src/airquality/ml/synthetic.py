"""Генерация синтетических данных датчиков для тестов и демонстрации."""

import numpy as np
import pandas as pd

from airquality.constants import TARGET_COLUMN


def generate_sensor_data(n: int = 20, *, with_target: bool = True, seed: int = 42) -> pd.DataFrame:
    """Случайные, но правдоподобные показания 16 признаков."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "season": rng.integers(1, 5, size=n),
            "weekday": rng.integers(1, 8, size=n),
            "temp": rng.normal(22, 3, size=n),
            "hum": rng.normal(45, 10, size=n),
            "mq7": rng.normal(120, 30, size=n),
            "mq135": rng.normal(110, 25, size=n),
            "ky028_analog": rng.integers(0, 1024, size=n),
            "ky028_digital": rng.integers(0, 2, size=n),
            "bmp_temp": rng.normal(15, 4, size=n),
            "pressure": rng.normal(1013, 8, size=n),
            "altitude": rng.normal(82, 5, size=n),
            "aht21_temp": rng.normal(22, 3, size=n),
            "aht21_hum": rng.normal(45, 10, size=n),
            "ens_iaq": rng.integers(0, 500, size=n),
            "ens_tvoc": rng.integers(0, 1000, size=n),
            "ens_co2": rng.integers(300, 2000, size=n),
        }
    )
    if with_target:
        df[TARGET_COLUMN] = rng.integers(0, 6, size=n)
    return df
