"""Конфигурация приложения.

Все параметры читаются из переменных окружения (и файла ``.env`` в корне проекта).
В коде нет паролей, адресов и абсолютных путей.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Параметры запуска сервисов и обучения."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- PostgreSQL ---
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "AirQualityMLDB"
    db_user: str = "postgres"
    db_password: SecretStr = SecretStr("")

    # --- ML ---
    fast_mode: bool = True
    random_state: int = 42
    automl_iter: int = 3
    retrain_limit: int = 10_000
    min_retrain_rows: int = 100

    # --- Файлы ---
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    monitoring_dir: Path = Path("monitoring")
    model_filename: str = "air_quality_model.joblib"
    metadata_filename: str = "metadata.json"

    # --- Сервисы ---
    ml_service_url: str = "http://localhost:5001"
    ml_service_timeout: float = 5.0
    ml_port: int = 5001
    web_port: int = 5000
    log_level: str = "INFO"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / self.model_filename

    @property
    def metadata_path(self) -> Path:
        return self.artifacts_dir / self.metadata_filename

    @property
    def db_params(self) -> dict[str, Any]:
        """Параметры для ``psycopg2.connect``."""
        return {
            "host": self.db_host,
            "port": self.db_port,
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password.get_secret_value(),
        }

    def ensure_dirs(self) -> None:
        """Создаёт рабочие каталоги, если их нет."""
        for directory in (self.artifacts_dir, self.reports_dir, self.monitoring_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Возвращает единственный экземпляр настроек (кэшируется)."""
    return Settings()
