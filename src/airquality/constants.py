"""Константы предметной области: признаки, классы IAQ, справочники."""

from typing import Final

FEATURE_COLUMNS: Final[list[str]] = [
    "season",
    "weekday",
    "temp",
    "hum",
    "mq7",
    "mq135",
    "ky028_analog",
    "ky028_digital",
    "bmp_temp",
    "pressure",
    "altitude",
    "aht21_temp",
    "aht21_hum",
    "ens_iaq",
    "ens_tvoc",
    "ens_co2",
]

TARGET_COLUMN: Final[str] = "iaq_class"

# Значение, которое ESP32 присылает при сбое датчика.
MISSING_SENSOR_VALUE: Final[int] = -1

# Типичные значения признаков: подставляются вместо -1 и пропусков.
DEFAULT_MEANS: Final[dict[str, float]] = {
    "season": 2,
    "weekday": 3,
    "temp": 22.0,
    "hum": 45.0,
    "mq7": 120.0,
    "mq135": 110.0,
    "ky028_analog": 35.0,
    "ky028_digital": 0.0,
    "bmp_temp": 15.0,
    "pressure": 1013.0,
    "altitude": 82.0,
    "aht21_temp": 22.0,
    "aht21_hum": 45.0,
    "ens_iaq": 75.0,
    "ens_tvoc": 250.0,
    "ens_co2": 800.0,
}

RECOMMENDATIONS: Final[dict[int, str]] = {
    0: "✅ Отлично",
    1: "🟢 Хорошо",
    2: "🟡 Удовл.",
    3: "🟠 Плохо",
    4: "🔴 Очень плохо",
    5: "🚨 АВАРИЯ!",
}
UNKNOWN_RECOMMENDATION: Final[str] = "Неизвестно"

SEASON_NAMES: Final[dict[int, str]] = {1: "Зима", 2: "Весна", 3: "Лето", 4: "Осень"}
WEEKDAY_NAMES: Final[dict[int, str]] = {
    1: "Пн",
    2: "Вт",
    3: "Ср",
    4: "Чт",
    5: "Пт",
    6: "Сб",
    7: "Вс",
}
