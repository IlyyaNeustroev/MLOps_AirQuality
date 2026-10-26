"""Схемы входных данных API (валидация через pydantic)."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from airquality.constants import MISSING_SENSOR_VALUE


class SensorReading(BaseModel):
    """Показания датчиков. Отсутствующее поле = сбой датчика (-1)."""

    model_config = ConfigDict(extra="ignore")

    season: int = Field(default=MISSING_SENSOR_VALUE, ge=-1, le=4)
    weekday: int = Field(default=MISSING_SENSOR_VALUE, ge=-1, le=7)
    temp: float = MISSING_SENSOR_VALUE
    hum: float = MISSING_SENSOR_VALUE
    mq7: float = MISSING_SENSOR_VALUE
    mq135: float = MISSING_SENSOR_VALUE
    ky028_analog: float = MISSING_SENSOR_VALUE
    ky028_digital: float = MISSING_SENSOR_VALUE
    bmp_temp: float = MISSING_SENSOR_VALUE
    pressure: float = MISSING_SENSOR_VALUE
    altitude: float = MISSING_SENSOR_VALUE
    aht21_temp: float = MISSING_SENSOR_VALUE
    aht21_hum: float = MISSING_SENSOR_VALUE
    ens_iaq: float = MISSING_SENSOR_VALUE
    ens_tvoc: float = MISSING_SENSOR_VALUE
    ens_co2: float = MISSING_SENSOR_VALUE


class RetrainRequest(BaseModel):
    """Переобучение из CSV (имя файла внутри DATA_DIR) или из БД."""

    csv_file: str | None = None
    limit: int = Field(default=10_000, gt=0, le=5_000_000)


class DriftRequest(BaseModel):
    """Имена CSV-файлов внутри DATA_DIR."""

    current_csv: str
    baseline_csv: str


def resolve_data_file(data_dir: Path, filename: str) -> Path:
    """Безопасно превращает имя файла из запроса в путь внутри ``data_dir``.

    Запрещает выход за пределы каталога (``../../etc/passwd``).
    """
    base = data_dir.resolve()
    path = (base / filename).resolve()
    if not path.is_relative_to(base):
        msg = "Файл должен находиться в каталоге данных"
        raise ValueError(msg)
    if not path.is_file():
        msg = f"Файл не найден: {filename}"
        raise FileNotFoundError(msg)
    return path
