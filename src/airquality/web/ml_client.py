"""HTTP-клиент ML-сервиса.

Веб-сервер больше не импортирует модель напрямую: он обращается к ML-сервису
по ``ML_SERVICE_URL``, как и задумано в docker-compose (два отдельных контейнера).
"""

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

ML_OFF: dict[str, Any] = {"iaq_class": -1, "probabilities": {}, "recommendation": "ML off"}


class MLServiceClient:
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Прогноз IAQ. При недоступности ML-сервиса возвращает заглушку ``ML off``."""
        try:
            resp = self._session.post(
                f"{self.base_url}/predict", json=features, timeout=self.timeout
            )
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
        except requests.RequestException:
            logger.exception("ML-сервис недоступен: %s", self.base_url)
            return dict(ML_OFF)
        return result
