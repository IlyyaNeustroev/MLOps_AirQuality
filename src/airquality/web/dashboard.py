"""API дашборда (аналитика, управление, логи, мониторинг, настройки).

Перенесено из FRONTEND/web-server/server.py как Blueprint. Закомментированные
дубликаты эндпоинтов удалены. Логика запросов пока не менялась — модуль
кандидат на дальнейшую декомпозицию (см. per-file-ignores в pyproject.toml).
"""

import logging
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request
from psycopg2.extras import RealDictCursor

from airquality.db.connection import get_db_connection

logger = logging.getLogger(__name__)
bp = Blueprint("dashboard", __name__)


@bp.route("/api/analytics", methods=["GET"])
def get_analytics_data():
    """
    Возвращает данные для AnalyticsPage:
        - энергопотребление сегодня / сейчас / месяц / экономия
        - часовой график за сегодня
        - AQI за неделю
        - потребление по зонам (room/zone → percent)
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "database_error"}), 501

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        today = datetime.now().date()
        month_start = today.replace(day=1)
        week_start = today - timedelta(days=7)

        # 1. Потребление за сегодня (пример: packet_count ≈ энергия)
        cursor.execute(
            """
            SELECT
                SUM(packet_count) as today_kwh,
                AVG(packet_count) as now_kw
            FROM sensor.data
            WHERE date = %s
        """,
            (today,),
        )
        row_today = cursor.fetchone()
        today_kwh = float(row_today["today_kwh"] or 0) / 1000  # пример нормализации
        now_kw = float(row_today["now_kw"] or 0) / 100  # пример «сейчас»

        # 2. Потребление за месяц
        cursor.execute(
            """
            SELECT SUM(packet_count) as month_kwh
            FROM sensor.data
            WHERE date BETWEEN %s AND %s
        """,
            (month_start, today),
        )
        row_month = cursor.fetchone()
        month_kwh = float(row_month["month_kwh"] or 0) / 1000

        # 3. Часовой график за сегодня
        cursor.execute(
            """
            SELECT
                EXTRACT(HOUR FROM created_at) as hour,
                SUM(packet_count) as count
            FROM sensor.data
            WHERE date = %s
            GROUP BY hour
            ORDER BY hour
        """,
            (today,),
        )
        rows_hour = cursor.fetchall()

        # нормализуем 0–100%
        hours = [0] * 24
        for r in rows_hour:
            h = int(r["hour"])
            if 0 <= h < 24:
                hours[h] = float(r["count"] or 0)

        max_count = max(hours) if max(hours) > 0 else 100
        hourly = [int(h / max_count * 100) for h in hours]

        # 4. AQI по дням за неделю (средний IAQ)
        cursor.execute(
            """
            SELECT
                date,
                AVG(iaq_class) as avg_iaq
            FROM sensor.data
            WHERE date BETWEEN %s AND %s
              AND iaq_class IS NOT NULL
            GROUP BY date
            ORDER BY date
        """,
            (week_start, today),
        )
        rows_week = cursor.fetchall()

        # IAQ точки 0–120 (для SVG)
        points = []
        for _ in range(7):
            points.append(0)
        for i, r in enumerate(rows_week[-7:]):
            idx = 6 - (len(rows_week) - 1 - i)
            if 0 <= idx < 7:
                points[idx] = max(0, min(120, int(r["avg_iaq"] or 0) * 15))

        # 5. Потребление по зонам (room_id → zone)
        zone_names = ["Серверная", "Открытый офис", "Конференц-зал", "Лаборатория", "Остальные"]
        cursor.execute("""
            SELECT
                room_id,
                SUM(packet_count) as total
            FROM sensor.data
            GROUP BY room_id
        """)
        rows_zone = cursor.fetchall()

        cursor.close()
        conn.close()

        # распределение по 5 зонам (пример: room_id 1–5 → 5 зон)
        total_all = sum([r["total"] or 0 for r in rows_zone]) + 1
        percents = [0] * 5
        for r in rows_zone:
            idx = (r["room_id"] - 1) % 5
            percents[idx] = (r["total"] or 0) / total_all * 100

        # 6. Ответ клиенту
        return jsonify(
            {
                # AnalyticsStats
                "today_kwh": today_kwh,
                "today_delta_pct": -8,  # пример: «↓ 8% vs вчера»
                "now_kw": now_kw,
                "month_kwh": month_kwh,
                "budget_kwh": 1500,
                "save_pct": 18,  # ↑ 18%
                # ConsumptionChart
                "hourly_consumption": hourly,
                # AQIChart
                "weekly_aqi": points,
                # ZonesConsumption
                "zones": [{"label": zone_names[i], "percent": percents[i]} for i in range(5)],
            }
        )

    except Exception:
        logger.exception("Ошибка /api/analytics")
        return jsonify({"error": "server_error"}), 500


# эндпоинт для страниц управления


# эндпоинт для страниц управления
@bp.route("/api/control", methods=["GET"])
def get_control_state():
    """
    Возвращает состояние всех зон (каждый room_id как "зона").
    Формат:
    {
        "zones": [
            {"id": "1", "name": "...", "power": 85, "temp": 22.4, "status": "on|off|auto"}
        ],
        "global_mode": "auto|manual"
    }
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "database_error"}), 501

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Последнее показание по каждой комнате + имя из sensor.rooms
        cursor.execute("""
            SELECT
                d.room_id,
                r.name_room as name,
                d.aht21_temp as temp,
                d.packet_count % 100 as power,
                CASE
                    WHEN d.iaq_class >= 3 THEN 'on'
                    WHEN d.iaq_class = 2 THEN 'auto'
                    ELSE 'off'
                END as status
            FROM sensor.data d
            INNER JOIN sensor.rooms r ON r.room_id = d.room_id
            WHERE d.id IN (
                SELECT MAX(id) FROM sensor.data GROUP BY room_id
            )
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        zones = []
        for r in rows:
            id = str(r["room_id"])
            zones.append(
                {
                    "id": id,
                    "name": r["name"] or f"Комната {id}",
                    "power": int(r["power"] or 70),
                    "temp": float(r["temp"] or 22.0),
                    "status": r["status"] or "auto",
                }
            )

        # Пример глобального режима
        global_mode = "auto"

        return jsonify({"global_mode": global_mode, "zones": zones})

    except Exception:
        logger.exception("Ошибка /api/control")
        return jsonify({"error": "server_error"}), 500


# Второй эндпоинт для старницы управления
@bp.route("/api/control", methods=["PUT", "POST"])
def update_control_state():
    """
    Принимает:
        {
            "action": "all_on|all_off|all_auto",
            "target_temp": 22.0,
            "fan_speed": 80
        }

    Здесь — mock: просто пишет в лог или обновляет "целевые" значения в БД.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid_json"}), 400

    action = data.get("action")
    target_temp = data.get("target_temp")
    fan_speed = data.get("fan_speed")

    logger.info(
        "Control action: %s, target_temp: %s, fan_speed: %s", action, target_temp, fan_speed
    )

    # 1. обновляем целевое состояние в БД (пример таблицы `sensor.control_targets`)
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            if action == "all_on":
                cursor.execute(
                    """
                    INSERT INTO sensor.control_targets (room_id, target_status, target_temp)
                    SELECT DISTINCT room_id, 'on', %s
                    FROM sensor.data
                    ON CONFLICT (room_id) DO UPDATE
                    SET target_status = 'on', target_temp = %s
                """,
                    (target_temp, target_temp),
                )
            elif action == "all_off":
                cursor.execute(
                    """
                    INSERT INTO sensor.control_targets (room_id, target_status, target_temp)
                    SELECT DISTINCT room_id, 'off', %s
                    FROM sensor.data
                    ON CONFLICT (room_id) DO UPDATE
                    SET target_status = 'off', target_temp = %s
                """,
                    (target_temp, target_temp),
                )
            else:
                cursor.execute(
                    """
                    UPDATE sensor.control_targets SET target_temp = %s, fan_speed = %s
                    WHERE target_temp IS NOT NULL
                """,
                    (target_temp, fan_speed),
                )

            conn.commit()
            cursor.close()
        except Exception:
            logger.exception("Control update error")
            conn.rollback()
        finally:
            conn.close()

    # ответ: просто текущее состояние (чтобы фронт обновился)
    return get_control_state()


# эндпоинт для страницы логов
@bp.route("/api/logs", methods=["GET"])
def get_logs():
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "database_error"}), 501

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        from_dt = request.args.get("from") or (datetime.now() - timedelta(days=7)).strftime(
            "%Y-%m-%d"
        )
        to_dt = request.args.get("to") or datetime.now().strftime("%Y-%m-%d")
        log_type = request.args.get("type") or "all"
        level = request.args.get("level") or "all"
        limit = int(request.args.get("limit") or 1000)

        where_parts = []
        params = [from_dt, to_dt]

        if log_type != "all":
            where_parts.append("log_type = %s")
            params.append(log_type)

        if level != "all":
            where_parts.append("log_level = %s")
            params.append(level)

        where_clause = " AND ".join(where_parts) if where_parts else ""

        # ВАЖНО: %% вместо % в тексте SQL (экранирование для psycopg2)
        if where_clause:
            query = f"""
                WITH raw_logs AS (
                    SELECT
                        id,
                        to_char(created_at, 'HH24:MI:SS') as time_,
                        to_char(created_at::date, 'YYYY-MM-DD') as date_,
                        room_id,
                        case
                            when category = 'alert' then 'alert'
                            when category = 'warning' then 'alert'
                            when iaq_class is not null then 'ml'
                            else 'system'
                        end as log_type,
                        case
                            when iaq_class >= 4 then 'Critical'
                            when iaq_class >= 3 then 'Warning'
                            else 'Info'
                        end as log_level,
                        'S-' || room_id::text as sensor,
                        room_id::text as room,
                        case
                            when category = 'alert' and ens_co2 > 1000 then 'CO2 превышен'
                            when category = 'alert' and hum > 70 then 'Влажность превысила порог'
                            when iaq_class is not null then 'ML IAQ=' || iaq_class::text
                            else 'Системное событие'
                        end as message,
                        case
                            when ens_co2 > 0 then ens_co2::text || '%%' || 'ppm'
                            when hum > 0 then hum::text || '%%'
                            else null
                        end as value_text
                    FROM (
                        SELECT
                            id,
                            created_at,
                            room_id,
                            temp, hum, ens_iaq, ens_tvoc, ens_co2,
                            iaq_class,
                            case
                                when iaq_class >= 4 then 'alert'
                                when iaq_class = 3 then 'warning'
                                else 'info'
                            end as category
                        FROM sensor.data
                        WHERE date BETWEEN %s AND %s
                    ) inner_src
                )
                SELECT * FROM raw_logs WHERE {where_clause}
                ORDER BY id DESC LIMIT %s
            """
        else:
            query = """
                WITH raw_logs AS (
                    SELECT
                        id,
                        to_char(created_at, 'HH24:MI:SS') as time_,
                        to_char(created_at::date, 'YYYY-MM-DD') as date_,
                        room_id,
                        case
                            when category = 'alert' then 'alert'
                            when category = 'warning' then 'alert'
                            when iaq_class is not null then 'ml'
                            else 'system'
                        end as log_type,
                        case
                            when iaq_class >= 4 then 'Critical'
                            when iaq_class >= 3 then 'Warning'
                            else 'Info'
                        end as log_level,
                        'S-' || room_id::text as sensor,
                        room_id::text as room,
                        case
                            when category = 'alert' and ens_co2 > 1000 then 'CO2 превышен'
                            when category = 'alert' and hum > 70 then 'Влажность превысила порог'
                            when iaq_class is not null then 'ML IAQ=' || iaq_class::text
                            else 'Системное событие'
                        end as message,
                        case
                            when ens_co2 > 0 then ens_co2::text || '%%' || 'ppm'
                            when hum > 0 then hum::text || '%%'
                            else null
                        end as value_text
                    FROM (
                        SELECT
                            id,
                            created_at,
                            room_id,
                            temp, hum, ens_iaq, ens_tvoc, ens_co2,
                            iaq_class,
                            case
                                when iaq_class >= 4 then 'alert'
                                when iaq_class = 3 then 'warning'
                                else 'info'
                            end as category
                        FROM sensor.data
                        WHERE date BETWEEN %s AND %s
                    ) inner_src
                )
                SELECT * FROM raw_logs
                ORDER BY id DESC LIMIT %s
            """

        params.append(limit)

        logger.debug("QUERY: %s", query)
        logger.debug("PARAMS: %s", params)
        logger.debug("PARAMS COUNT: %d", len(params))
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        logs = []
        for r in rows:
            logs.append(
                {
                    "id": str(r["id"]),
                    "time": r["date_"] + " " + r["time_"],
                    "type": r["log_type"],
                    "sensor": r["sensor"],
                    "room": r["room"],
                    "message": r["message"],
                    "value": r["value_text"],
                }
            )

        return jsonify(
            {
                "logs": logs,
                "from": from_dt,
                "to": to_dt,
                "filter": {"type": log_type, "level": level},
            }
        )

    except Exception:
        logger.exception("Ошибка /api/logs")
        return jsonify({"error": "server_error"}), 500

    finally:
        cursor.close()
        conn.close()


# Эндпоинт для страницы мониторинга
@bp.route("/api/monitoring", methods=["GET"])
def get_monitoring_data():
    """
    Возвращает данные для страницы MonitoringPage:
        - общее число датчиков,
        - сколько онлайн/внимание/критично,
        - список датчиков с CO2, темп, влажность, AQI.

    Результат:
        {
            "summary": { "total": 12, "online": 7, "warning": 3, "critical": 1 },
            "sensors": [
                { "id": "S-01", "room": "Зал", "co2": 412, "temp": 22.4, "hum": 58, "aqi": 94, "status": "online|warning|critical" }
            ]
        }
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "database_error"}), 501

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        now = datetime.now()
        one_hour = now - timedelta(hours=1)

        # 1. агрегируем по room_id (каждый room_id = один "датчик")
        cursor.execute(
            """
            SELECT
                room_id,
                MAX(created_at) as last_seen,
                COUNT(*) as count_all,
                AVG(ens_co2) as avg_co2,
                AVG(temp) as avg_temp,
                AVG(hum) as avg_hum,
                AVG(iaq_class) as avg_iaq
            FROM sensor.data
            WHERE created_at > %s AND ens_co2 IS NOT NULL
            GROUP BY room_id
        """,
            (one_hour,),
        )

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        total = len(rows)
        online = 0
        warning = 0
        critical = 0

        # пороги IAQ для статуса
        def classify_iaq(v):
            if v is None:
                return "offline"
            v = float(v or 0)
            if v >= 4:
                # critical += 1
                return "critical"
            if v >= 2:
                # warning += 1
                return "warning"
            # online += 1
            return "online"

        sensors = []
        for r in rows:
            iaq = r["avg_iaq"]
            status = classify_iaq(iaq)

            # Обновляем счётчики здесь
            if status == "online":
                online += 1
            elif status == "warning":
                warning += 1
            elif status == "critical":
                critical += 1

            sensors.append(
                {
                    "id": f"S-{r['room_id']}",
                    "room": f"room_{r['room_id']}",
                    "co2": int(r["avg_co2"] or 400),
                    "temp": float(r["avg_temp"] or 22.0),
                    "hum": float(r["avg_hum"] or 50.0),
                    "aqi": int(iaq or 0),  # max(1, int(iaq or 0) * 25),  # 0..100
                    "status": status,
                }
            )

        summary = {"total": total, "online": online, "warning": warning, "critical": critical}

        return jsonify({"summary": summary, "sensors": sensors})

    except Exception:
        logger.exception("Ошибка /api/monitoring")
        return jsonify({"error": "server_error"}), 500


# Эндпоинт для страницы настроек
@bp.route("/api/settings", methods=["GET"])
def get_settings():
    """
    Возвращает настройки системы и ML-модели.
    Формат:
        {
            "ml": { ... },
            "system": { ... },
            "notifications": { ... },
            "model_info": { ... }
        }
    """
    # 1. stereotype ML settings (или читать из файла)
    settings = {
        "ml": {
            "enabled": True,
            "auto_retrain": True,
            "forecast_horizon_hours": 6,
            "update_interval_minutes": 15,
            "threshold_good": 80,
            "threshold_moderate": 60,
        },
        "system": {"log_level": "INFO", "db_host": "localhost", "db_port": 5432},
        "notifications": {"email_enabled": True, "tg_enabled": False, "web_push": True},
    }

    # 2. модельная метаинформация
    ml_model_info = {
        "name": "AeroML v2.4",
        "version": "2.4.1",
        "accuracy": 97.8,
        "train_date": "2026-04-10",
        "dataset_size": 128400,
        "features": ["CO2", "Temperature", "Humidity", "PM2.5", "VOC", "TimeOfDay", "Occupancy"],
    }

    return jsonify({"settings": settings, "model_info": ml_model_info})


# Второй эндпоинт для страницы настроек
@bp.route("/api/settings", methods=["PUT", "POST"])
def update_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid_json"}), 400

    # 1. обновляем в памяти / в файл (пример)
    ml_settings = data.get("ml", {})
    system_settings = data.get("system", {})
    notifications_settings = data.get("notifications", {})

    # допустимо: запись в файл или БД

    # 2. здесь можно обновлять FLASK‑переменные, перезагружать модель и т.п.
    # пример логгирования
    logger.info("ML settings updated: %s", ml_settings)
    logger.info("System settings updated: %s", system_settings)
    logger.info("Notifications settings updated: %s", notifications_settings)

    # ответ — полный актуальный объект
    ml_model_info = {
        "name": "AeroML v2.4",
        "version": "2.4.1",
        "accuracy": 97.8,
        "train_date": "2026-04-10",
        "dataset_size": 128400,
        "features": ["CO2", "Temperature", "Humidity", "PM2.5", "VOC", "TimeOfDay", "Occupancy"],
    }

    return jsonify({"settings": data, "model_info": ml_model_info})
