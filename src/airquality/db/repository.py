"""SQL-запросы к таблице sensor.data. Только параметризованные запросы."""

import json
from datetime import datetime
from typing import Any

import pandas as pd

from airquality.config import Settings
from airquality.constants import FEATURE_COLUMNS, TARGET_COLUMN
from airquality.db.connection import db_connection

_TRAINING_COLUMNS = ", ".join([*FEATURE_COLUMNS, TARGET_COLUMN])

TRAINING_QUERY = f"""
    SELECT {_TRAINING_COLUMNS}
    FROM sensor.data
    WHERE iaq_class IS NOT NULL
    ORDER BY created_at DESC
    LIMIT %(limit)s
"""  # noqa: S608 — список столбцов фиксирован в коде, пользовательский ввод только в params

INSERT_QUERY = """
    INSERT INTO sensor.data (
        room_id, date, time, season, weekday, temp, hum, mq7, mq135,
        ky028_analog, ky028_digital, bmp_temp, pressure, altitude,
        aht21_temp, aht21_hum, ens_iaq, ens_tvoc, ens_co2,
        packet_count, iaq_class, iaq_proba
    ) VALUES (
        %(room_id)s, %(date)s, %(time)s, %(season)s, %(weekday)s,
        %(temp)s, %(hum)s, %(mq7)s, %(mq135)s, %(ky028_analog)s, %(ky028_digital)s,
        %(bmp_temp)s, %(pressure)s, %(altitude)s, %(aht21_temp)s, %(aht21_hum)s,
        %(ens_iaq)s, %(ens_tvoc)s, %(ens_co2)s, %(packet_count)s,
        %(iaq_class)s, %(iaq_proba)s
    )
"""

RECENT_IAQ_QUERY = """
    SELECT iaq_class, created_at
    FROM sensor.data
    WHERE iaq_class IS NOT NULL
    ORDER BY id DESC
    LIMIT %(limit)s
"""

_INT_FIELDS = (
    "room_id",
    "season",
    "weekday",
    "ky028_analog",
    "ky028_digital",
    "ens_iaq",
    "ens_tvoc",
    "ens_co2",
)
_FLOAT_FIELDS = (
    "temp",
    "hum",
    "mq7",
    "mq135",
    "bmp_temp",
    "pressure",
    "altitude",
    "aht21_temp",
    "aht21_hum",
)


def _to_number(value: Any, *, as_int: bool, default: float = -1) -> float:
    if value is None or str(value).strip().lower() in {"", "null", "none"}:
        return default
    try:
        return int(float(value)) if as_int else float(value)
    except (TypeError, ValueError):
        return default


def fetch_training_data(settings: Settings, limit: int) -> pd.DataFrame:
    """Последние ``limit`` размеченных записей для переобучения."""
    with db_connection(settings) as conn:
        return pd.read_sql_query(TRAINING_QUERY, conn, params={"limit": int(limit)})


def build_measurement_record(
    data: dict[str, Any], iaq_class: int, probabilities: dict[int, float], packet_count: int
) -> dict[str, Any]:
    """Готовит запись для INSERT: приводит типы, подставляет -1 и текущие дату/время."""
    now = datetime.now()
    record: dict[str, Any] = {
        "date": data.get("date") or now.strftime("%Y-%m-%d"),
        "time": data.get("time") or now.strftime("%H:%M:%S"),
        "packet_count": packet_count,
        "iaq_class": iaq_class,
        "iaq_proba": json.dumps(probabilities),
    }
    record.update({f: _to_number(data.get(f), as_int=True) for f in _INT_FIELDS})
    record.update({f: _to_number(data.get(f), as_int=False) for f in _FLOAT_FIELDS})
    return record


def save_measurement(settings: Settings, record: dict[str, Any]) -> None:
    """Сохраняет одно измерение в sensor.data."""
    with db_connection(settings) as conn, conn.cursor() as cur:
        cur.execute(INSERT_QUERY, record)


def fetch_recent_iaq(settings: Settings, limit: int = 10) -> tuple[list[str], list[int]]:
    """Метки времени и классы IAQ последних измерений (по возрастанию времени)."""
    with db_connection(settings) as conn, conn.cursor() as cur:
        cur.execute(RECENT_IAQ_QUERY, {"limit": int(limit)})
        rows = sorted(cur.fetchall(), key=lambda r: r[1])
    return [r[1].strftime("%H:%M") for r in rows], [int(r[0] or 0) for r in rows]
