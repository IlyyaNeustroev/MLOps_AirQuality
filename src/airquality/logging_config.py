"""Единая настройка логирования для всех сервисов."""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Настраивает корневой логгер: вывод в stdout (его собирает Docker)."""
    logging.basicConfig(
        level=level.upper(),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
