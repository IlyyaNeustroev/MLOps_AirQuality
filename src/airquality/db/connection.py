"""Подключение к PostgreSQL по параметрам из Settings."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg2
from psycopg2.extensions import connection as PgConnection  # noqa: N812

from airquality.config import Settings, get_settings

logger = logging.getLogger(__name__)


@contextmanager
def db_connection(settings: Settings | None = None) -> Iterator[PgConnection]:
    """Контекстный менеджер: commit при успехе, rollback при ошибке, всегда close."""
    conn = psycopg2.connect(**(settings or get_settings()).db_params)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_db_connection(settings: Settings | None = None) -> PgConnection | None:
    """Старый интерфейс для эндпоинтов дашборда: соединение или ``None``."""
    try:
        conn: PgConnection = psycopg2.connect(**(settings or get_settings()).db_params)
    except psycopg2.Error:
        logger.exception("Не удалось подключиться к БД")
        return None
    return conn
