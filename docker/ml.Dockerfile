# syntax=docker/dockerfile:1
# ML-сервис. Сборка из корня репозитория:  docker build -f docker/ml.Dockerfile .

# ---------- 1. builder: Poetry ставит зависимости в /app/.venv ----------
FROM python:3.13-slim AS builder
ENV POETRY_VERSION=2.5.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1
RUN pip install "poetry==${POETRY_VERSION}"
WORKDIR /app
# Сначала только манифесты — слой с зависимостями кэшируется, пока они не меняются
COPY pyproject.toml poetry.lock readme.md ./
RUN poetry install --only main --no-root
COPY src ./src
RUN poetry install --only main

# ---------- 2. runtime: без Poetry и компиляторов, от непривилегированного пользователя ----------
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ARTIFACTS_DIR=/app/artifacts REPORTS_DIR=/app/reports \
    MONITORING_DIR=/app/monitoring DATA_DIR=/app/data
RUN mkdir -p artifacts reports monitoring data && chown -R app:app /app
USER app
EXPOSE 5001
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD curl -fs http://localhost:5001/health || exit 1
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "4", "--timeout", "300", \
     "airquality.ml_service.app:create_app()"]
