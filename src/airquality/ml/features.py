"""Вычисляемые признаки: сезон и день недели по дате измерения."""

from datetime import datetime
from typing import Any

DATE_FORMAT = "%Y-%m-%d"


def get_season_by_month(month: int) -> int:
    """Номер сезона: 1 — зима, 2 — весна, 3 — лето, 4 — осень."""
    if month in (12, 1, 2):
        return 1
    if month in (3, 4, 5):
        return 2
    if month in (6, 7, 8):
        return 3
    return 4


def add_calendar_features(data: dict[str, Any]) -> dict[str, Any]:
    """Добавляет ``season`` и ``weekday``, если в пакете есть корректное поле ``date``.

    Возвращает новый словарь, исходный не изменяется.
    """
    result = dict(data)
    raw_date = data.get("date")
    if not raw_date:
        return result
    try:
        parsed = datetime.strptime(str(raw_date), DATE_FORMAT)  # noqa: DTZ007 — дата без времени
    except ValueError:
        return result
    result["season"] = get_season_by_month(parsed.month)
    result["weekday"] = parsed.isoweekday()
    return result
