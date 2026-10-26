import pandas as pd

from airquality.constants import DEFAULT_MEANS, FEATURE_COLUMNS
from airquality.ml.preprocessing import preprocess_features


def test_replaces_minus_one_with_defaults() -> None:
    df = pd.DataFrame([dict.fromkeys(FEATURE_COLUMNS, -1)])
    x = preprocess_features(df)
    assert (x.to_numpy() != -1).all()
    assert x.loc[0, "temp"] == DEFAULT_MEANS["temp"]


def test_adds_missing_columns_and_drops_extra() -> None:
    x = preprocess_features(pd.DataFrame([{"temp": 21.0, "garbage": "x"}]))
    assert list(x.columns) == FEATURE_COLUMNS
    assert x.loc[0, "temp"] == 21.0
    assert x.loc[0, "hum"] == DEFAULT_MEANS["hum"]


def test_non_numeric_values_become_defaults() -> None:
    x = preprocess_features(pd.DataFrame([{"temp": "abc"}]))
    assert x.loc[0, "temp"] == DEFAULT_MEANS["temp"]
