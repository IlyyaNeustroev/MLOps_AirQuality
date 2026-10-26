import pandas as pd

from airquality.constants import FEATURE_COLUMNS
from airquality.ml.drift import drift_report, population_stability_index
from airquality.ml.synthetic import generate_sensor_data


def test_psi_is_zero_for_identical_samples() -> None:
    s = pd.Series(range(100))
    assert population_stability_index(s, s) < 1e-9


def test_psi_detects_shift() -> None:
    base = pd.Series(range(100), dtype=float)
    assert population_stability_index(base, base + 50) > 0.25


def test_drift_report_covers_all_features() -> None:
    rows = drift_report(
        generate_sensor_data(50, seed=1), generate_sensor_data(50, seed=2), FEATURE_COLUMNS
    )
    assert {r["feature"] for r in rows} == set(FEATURE_COLUMNS)
