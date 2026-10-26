import pytest

from airquality.ml.features import add_calendar_features, get_season_by_month


@pytest.mark.parametrize(
    ("month", "season"), [(1, 1), (2, 1), (12, 1), (3, 2), (5, 2), (6, 3), (8, 3), (9, 4), (11, 4)]
)
def test_season_by_month(month: int, season: int) -> None:
    assert get_season_by_month(month) == season


def test_calendar_features_from_date() -> None:
    result = add_calendar_features({"date": "2026-10-26", "temp": 21})
    assert result["season"] == 4
    assert result["weekday"] == 1  # понедельник
    assert result["temp"] == 21


def test_calendar_features_bad_date_is_ignored() -> None:
    assert "season" not in add_calendar_features({"date": "26.10.2026"})
