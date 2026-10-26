"""Потокобезопасное хранилище последних показаний по каждой комнате."""

import threading
from copy import deepcopy
from typing import Any

EMPTY_READING: dict[str, Any] = {
    "mq7": 0, "mq135": 0, "temp": 0.0, "hum": 0.0, "ky028_analog": 0, "ky028_digital": 0,
    "bmp_temp": 0.0, "pressure": 0.0, "altitude": 0.0, "aht21_temp": 0.0, "aht21_hum": 0.0,
    "ens_iaq": 0, "ens_tvoc": 0, "ens_co2": 0, "room_id": 0, "date": "", "time": "",
    "season": 0, "weekday": 0, "iaq_class": -1, "recommendation": "Нет данных",
}  # fmt: skip


class SensorState:
    """Последние данные по комнатам и счётчик пакетов.

    Раньше был один глобальный словарь ``last_data``: данные разных комнат смешивались,
    а при многопоточном сервере возможна гонка. Теперь — словарь по ``room_id`` под Lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rooms: dict[int, dict[str, Any]] = {}
        self._last_room: int | None = None
        self._packet_count = 0

    def update(self, room_id: int, data: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Обновляет показания комнаты, возвращает (копия состояния комнаты, номер пакета)."""
        with self._lock:
            room = self._rooms.setdefault(room_id, deepcopy(EMPTY_READING))
            room.update({k: v for k, v in data.items() if k in EMPTY_READING})
            room["room_id"] = room_id
            self._last_room = room_id
            self._packet_count += 1
            return dict(room), self._packet_count

    def set_prediction(self, room_id: int, iaq_class: int, recommendation: str) -> None:
        with self._lock:
            room = self._rooms.setdefault(room_id, deepcopy(EMPTY_READING))
            room["iaq_class"] = iaq_class
            room["recommendation"] = recommendation

    def latest(self) -> dict[str, Any]:
        """Данные последней приславшей комнаты (для главной страницы)."""
        with self._lock:
            if self._last_room is None:
                return deepcopy(EMPTY_READING)
            return dict(self._rooms[self._last_room])

    @property
    def packet_count(self) -> int:
        with self._lock:
            return self._packet_count
