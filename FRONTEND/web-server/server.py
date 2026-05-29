import sys
import os
import json  #Для JSON в БД!
from datetime import datetime, timedelta  # <-- ДОБАВИТЬ timedelta

# Настройка путей
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ml_dir = os.path.join(project_root, 'ML')
sys.path.insert(0, ml_dir)

# Проверка ML файлов
print(f"Корень проекта: {project_root}")
print(f"ML папка: {ml_dir}")
print(f"Модель: {os.path.exists(os.path.join(ml_dir, 'air_quality_model.pkl'))}")
print(f"Scaler: {os.path.exists(os.path.join(ml_dir, 'scaler.pkl'))}")

try:
    from air_quality_ml import AirQualityPredictor
    predictor = AirQualityPredictor(
        model_path=os.path.join(ml_dir, 'air_quality_model.pkl'),
        scaler_path=os.path.join(ml_dir, 'scaler.pkl')
    )
    print("✅ ML модель загружена!")
except Exception as e:
    print(f"Ошибка ML: {e}")
    predictor = None

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from datetime import datetime
import socket
import psycopg2
from psycopg2.extras import RealDictCursor

# Файл с настройками ML
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'ml_settings.json')

app = Flask(__name__)
CORS(app)

# Конфигурация БД
DB_CONFIG = {
    'host': 'localhost', 'database': 'AirQualityMLDB',
    'user': 'postgres', 'password': '1187', 'port': 5432
}

def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Ошибка БД: {e}")
        return None

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Неизвестно"

# HTML шаблон (БЕЗ ИЗМЕНЕНИЙ)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html><head><title>Умная вентиляция</title><meta charset="utf-8">
<style>
body{font-family:Arial,sans-serif;margin:20px;background:#f5f5f5;}
.sensor-card{border:1px solid #ddd;padding:15px;margin:10px 0;border-radius:8px;background:white;box-shadow:0 2px 5px rgba(0,0,0,0.1);}
.value{font-size:1.5em;font-weight:bold;color:#333;}
.timestamp{color:#666;font-style:italic;}
h1{color:#2c3e50;text-align:center;}
.packet-counter{background:#e7f3ff;padding:15px;border-radius:10px;margin:20px 0;text-align:center;font-weight:bold;font-size:1.2em;border:2px solid #2196F3;}
.info-card{background:#e3f2fd;padding:15px;border-radius:10px;margin:20px 0;border-left:5px solid #2196F3;}
.iaq-card{background:linear-gradient(135deg, #4CAF50, #8BC34A);color:white;padding:20px;border-radius:15px;margin:20px 0;text-align:center;box-shadow:0 4px 15px rgba(76,175,80,0.3);}
.iaq-bad{background:linear-gradient(135deg, #FF9800, #F57C00);box-shadow:0 4px 15px rgba(255,152,0,0.3);}
.iaq-worst{background:linear-gradient(135deg, #F44336, #D32F2F);box-shadow:0 4px 15px rgba(244,67,54,0.3);}
</style>
</head><body>
<h1>📊 Умная вентиляция</h1>
<div class="packet-counter">📦 Пакетов: <span class="value">{{ packet_count }}</span> | Последние данные: {{ timestamp }}</div>

<div class="info-card">
<h3>🏠 Комната</h3>
<p>№: {{ room_id }} | {{ date }} {{ time }} | {{ season_name }}({{ season }}) | {{ weekday_name }}({{ weekday }})</p>
</div>

<!-- ✅ IAQ КАРТОЧКА -->
<div class="iaq-card {% if iaq_class == 0 %}iaq-good{% elif iaq_class >= 4 %}iaq-worst{% elif iaq_class >= 3 %}iaq-bad{% endif %}">
<h2>🌬️ IAQ: <span class="value">{{ iaq_class }}</span>/5</h2>
<p><strong>{{ recommendation }}</strong></p>
{% if iaq_proba %}
<p>Уверенность: {{ "{:.0%}".format((iaq_proba|max)|float) }}</p>
{% endif %}
</div>

<div class="sensor-card"><h4>🌡️ Темп:</h4><p class="value">{{ temp }}°C</p></div>
<div class="sensor-card"><h4>💧 Влаж:</h4><p class="value">{{ hum }}%</p></div>
<div class="sensor-card"><h4>💨 CO MQ7:</h4><p class="value">{{ mq7 }}</p></div>
<div class="sensor-card"><h4>🌫 MQ135:</h4><p class="value">{{ mq135 }}</p></div>
<div class="sensor-card"><h4>☀️ KY028 A:</h4><p class="value">{{ ky028_analog }}</p></div>
<div class="sensor-card"><h4>💡 KY028 D:</h4><p class="value">{{ ky028_digital }}</p></div>
<div class="sensor-card"><h4>🌡️ BMP T:</h4><p class="value">{{ bmp_temp }}°C</p></div>
<div class="sensor-card"><h4>🌀 Давл:</h4><p class="value">{{ pressure }}</p></div>
<div class="sensor-card"><h4>🪂 Выс:</h4><p class="value">{{ altitude }}</p></div>
<div class="sensor-card"><h4>🌡️ AHT21 T:</h4><p class="value">{{ aht21_temp }}°C</p></div>
<div class="sensor-card"><h4>💦 AHT21 H:</h4><p class="value">{{ aht21_hum }}%</p></div>
<div class="sensor-card"><h4>ENS IAQ:</h4><p class="value">{{ ens_iaq }}</p></div>
<div class="sensor-card"><h4>TVOC:</h4><p class="value">{{ ens_tvoc }}</p></div>
<div class="sensor-card"><h4>CO₂:</h4><p class="value">{{ ens_co2 }}</p></div>

<!-- График последних 24ч IAQ -->
<div style="margin:20px 0;padding:20px;background:white;border-radius:10px;">
<h3>📈 IAQ за 24ч</h3>
<canvas id="iaqChart" width="400" height="200"></canvas>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
const ctx = document.getElementById('iaqChart').getContext('2d');
new Chart(ctx, {
  type: 'line', data: {
    labels: {{ recent_iaq_labels|tojson }},
    datasets: [{label: 'IAQ класс', data: {{ recent_iaq_data|tojson }}, 
                borderColor: '#2196F3', backgroundColor: 'rgba(33,150,244,0.1)', fill: true,
                tension: 0.4}]
  }, options: {
    scales: {y: {min: 0, max: 5, ticks: {stepSize: 1}}},
    plugins: {title: {display: true, text: 'Последние измерения'}}
  }
});
</script>
</div>
</body></html>
'''

last_data = {
    'mq7': 0, 'mq135': 0, 'temp': 0.0, 'hum': 0.0, 'ky028_analog': 0, 'ky028_digital': 0,
    'bmp_temp': 0.0, 'pressure': 0.0, 'altitude': 0.0, 'aht21_temp': 0.0, 'aht21_hum': 0.0,
    'ens_iaq': 0, 'ens_tvoc': 0, 'ens_co2': 0, 'room_id': 0, 'date': '', 'time': '', 'season': 0, 'weekday': 0,
    'iaq_class': -1, 'recommendation': 'Нет данных'
}

SEASON_NAMES = {1: 'Зима', 2: 'Весна', 3: 'Лето', 4: 'Осень'}
WEEKDAY_NAMES = {1: 'Пн', 2: 'Вт', 3: 'Ср', 4: 'Чт', 5: 'Пт', 6: 'Сб', 7: 'Вс'}
packet_count = 0

@app.route('/')
def index():
    # Имена сезонов/дней
    season_name = SEASON_NAMES.get(last_data['season'], '?')
    weekday_name = WEEKDAY_NAMES.get(last_data['weekday'], '?')
    
    # Подготовка данных для шаблона (без конфликтов)
    page_data = dict(last_data)  # Копия
    page_data.update({
        'iaq_class': last_data.get('iaq_class', -1),
        'recommendation': last_data.get('recommendation', 'Нет данных'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'packet_count': packet_count,
        'season_name': season_name,
        'weekday_name': weekday_name
    })
    
    # График из БД
    recent_labels, recent_data = get_recent_iaq(10)
    page_data['recent_iaq_labels'] = recent_labels
    page_data['recent_iaq_data'] = recent_data
    
    return render_template_string(HTML_TEMPLATE, **page_data)

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({**last_data, 'timestamp': datetime.now().isoformat(), 'packet_count': packet_count})

def get_season_by_month(month):
    if month in [12,1,2]: return 1
    elif month in [3,4,5]: return 2
    elif month in [6,7,8]: return 3
    return 4

def get_recent_iaq(limit=10):
    """Последние 10 IAQ из БД"""
    conn = get_db_connection()
    if not conn: 
        return ['?']*limit, [1]*limit
    
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT iaq_class, created_at 
            FROM sensor.data 
            WHERE iaq_class IS NOT NULL 
            ORDER BY id DESC LIMIT {limit}
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Сортируем по времени (хронологически)
        rows.sort(key=lambda x: x[1])
        
        labels = [row[1].strftime('%H:%M') for row in rows]
        data = [row[0] if row[0] is not None else 0 for row in rows]
        
        return labels, data
    except Exception as e:
        print(f"Ошибка графика: {e}")
        return ['?']*limit, [1]*limit

@app.route('/data', methods=['POST'])
def receive_data():
    global last_data, packet_count
    try:
        data = request.get_json(force=True)
        if not data: return jsonify({'error': 'Invalid JSON'}), 400

        # Обновление данных
        for k, v in data.items():
            if k in last_data: last_data[k] = v

        # Season/Weekday
        if 'date' in data:
            try:
                date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
                last_data['season'] = get_season_by_month(date_obj.month)
                last_data['weekday'] = date_obj.isoweekday()
            except: pass

        packet_count += 1

        # ML
        if predictor:
            iaq_result = predictor.predict(last_data)
            last_data['iaq_class'] = iaq_result['iaq_class']
            last_data['recommendation'] = iaq_result['recommendation']
            print(f"ML IAQ: {iaq_result['iaq_class']}")
        else:
            iaq_result = {'iaq_class': -1, 'recommendation': 'ML off'}

        save_to_database(last_data, iaq_result)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Пакет #{packet_count}")
        return jsonify({
            'status': 'success', 'packet_number': packet_count,
            'iaq_class': iaq_result['iaq_class'],
            'recommendation': iaq_result['recommendation']
        }), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400

def save_to_database(data, iaq_result):
    conn = get_db_connection()
    if not conn: return

    try:
        cursor = conn.cursor()
        now = datetime.now()
        record_date = data.get('date', now.strftime('%d.%m.%Y'))
        record_time = data.get('time', now.strftime('%H:%M:%S'))

        def safe_convert(value, typ, default=-1):
            if value is None or str(value).lower() in ('null', 'none', ''): return default
            try:
                return int(value) if typ == 'int' else float(value)
            except: return default

        insert_data = {
            'room_id': safe_convert(data.get('room_id'), 'int'),
            'date': record_date, 'time': record_time,
            'season': safe_convert(data.get('season'), 'int'),
            'weekday': safe_convert(data.get('weekday'), 'int'),
            'temp': safe_convert(data.get('temp'), 'float'),
            'hum': safe_convert(data.get('hum'), 'float'),
            'mq7': safe_convert(data.get('mq7'), 'float'),
            'mq135': safe_convert(data.get('mq135'), 'float'),
            'ky028_analog': safe_convert(data.get('ky028_analog'), 'int'),
            'ky028_digital': safe_convert(data.get('ky028_digital'), 'int'),
            'bmp_temp': safe_convert(data.get('bmp_temp'), 'float'),
            'pressure': safe_convert(data.get('pressure'), 'float'),
            'altitude': safe_convert(data.get('altitude'), 'float'),
            'aht21_temp': safe_convert(data.get('aht21_temp'), 'float'),
            'aht21_hum': safe_convert(data.get('aht21_hum'), 'float'),
            'ens_iaq': safe_convert(data.get('ens_iaq'), 'int'),
            'ens_tvoc': safe_convert(data.get('ens_tvoc'), 'int'),
            'ens_co2': safe_convert(data.get('ens_co2'), 'int'),
            'packet_count': packet_count,
            'iaq_class': iaq_result['iaq_class'],
            'iaq_proba': json.dumps(iaq_result['probabilities'])  # ✅ JSON!
        }

        query = """
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

        cursor.execute(query, insert_data)
        conn.commit()
        print(f"Сохранено IAQ={iaq_result['iaq_class']}")

    except Exception as e:
        print(f"Ошибка БД: {e}")
        if conn: conn.rollback()
    finally:
        if 'cursor' in locals(): cursor.close()
        conn.close()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# Новый эндпоинт для реального времени
'''@app.route('/api/dashboard/realtime', methods=['GET'])
def get_realtime_data():
    """
    Возвращает последние данные сенсоров, статус по этажам и события.
    Ответ: JSON с ключами "sensors", "floors_status", "events".
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "database_error"}), 501

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # 1. Последние значения для каждой комнаты (room_id)
        cursor.execute("""
            SELECT room_id, temp, hum, mq7, mq135, bmp_temp, aht21_temp, 
                   aht21_hum, ens_iaq, ens_tvoc, ens_co2, pressure, iaq_class, packet_count
            FROM sensor.data
            WHERE id IN (
                SELECT MAX(id) FROM sensor.data GROUP BY room_id
            )
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # 2. Структура для фронта
        #floor_names = {1: "Цоколь", 2: "Этаж 1", 3: "Этаж 2", 4: "Этаж 3", 5: "Этаж 4"}
        sensors = {}
        floors = []
        for r in rows:
            key = f"room_{r['room_id']}"
            sensors[key] = dict(r)
            floor_name =f"Помещение {r['room_id']}" # floor_names.get(r['room_id'], f"Помещение {r['room_id']}")
            value = r['iaq_class'] #max(0, min(100, (r['iaq_class'] or 0) ))  # IAQ -> 0-100
            floors.append({"name": floor_name, "value": value})

        # 3. События (пример, можно адаптировать)
        events = [
            {"time": "14:23", "text": "Высокая влажность", "type": "warning"},
            {"time": "13:55", "text": "VOC превышен", "type": "error"},
            {"time": "12:10", "text": "Датчик восстановлен", "type": "success"},
        ]

        return jsonify({
            "sensors": sensors,
            "floors_status": floors,
            "events": events
        })

    except Exception as e:
        print(f"Ошибка /api/dashboard/realtime: {e}")
        return jsonify({"error": "server_error"}), 500'''

# Новый эндпоинт для реального времени
@app.route('/api/dashboard/realtime', methods=['GET'])
def get_realtime_data():
    """
    Возвращает последние данные сенсоров, статус по этажам и события.
    Ответ: JSON с ключами "sensors", "floors_status", "events".
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "database_error"}), 501

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # 1. Последние значения для каждой комнаты + имя из sensor.rooms
        cursor.execute("""
            SELECT d.room_id, r.name_room as name, d.temp, d.hum, d.mq7, d.mq135, 
                   d.bmp_temp, d.aht21_temp, d.aht21_hum, d.ens_iaq, d.ens_tvoc, 
                   d.ens_co2, d.pressure, d.iaq_class, d.packet_count
            FROM sensor.data d
            INNER JOIN sensor.rooms r ON r.room_id = d.room_id
            WHERE d.id IN (
                SELECT MAX(id) FROM sensor.data GROUP BY room_id
            )
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # 2. Структура для фронта
        sensors = {}
        floors = []
        for r in rows:
            key = f"room_{r['room_id']}"
            sensors[key] = dict(r)
            # Вместо "Помещение 1" — реальное название из БД
            room_name = r['name'] or f"Помещение {r['room_id']}"
            value = r['iaq_class']  # IAQ -> 0-100
            floors.append({"name": room_name, "value": value})

        # 3. События (пример, можно адаптировать)
        events = [
            {"time": "14:23", "text": "Высокая влажность", "type": "warning"},
            {"time": "13:55", "text": "VOC превышен", "type": "error"},
            {"time": "12:10", "text": "Датчик восстановлен", "type": "success"},
        ]

        return jsonify({
            "sensors": sensors,
            "floors_status": floors,
            "events": events
        })

    except Exception as e:
        print(f"Ошибка /api/dashboard/realtime: {e}")
        return jsonify({"error": "server_error"}), 500


#эндпоинт для страницы аналитики
@app.route('/api/analytics', methods=['GET'])
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
        cursor.execute("""
            SELECT
                SUM(packet_count) as today_kwh,
                AVG(packet_count) as now_kw
            FROM sensor.data
            WHERE date = %s
        """, (today,))
        row_today = cursor.fetchone()
        today_kwh = float(row_today['today_kwh'] or 0) / 1000  # пример нормализации
        now_kw = float(row_today['now_kw'] or 0) / 100   # пример «сейчас»

        # 2. Потребление за месяц
        cursor.execute("""
            SELECT SUM(packet_count) as month_kwh
            FROM sensor.data
            WHERE date BETWEEN %s AND %s
        """, (month_start, today))
        row_month = cursor.fetchone()
        month_kwh = float(row_month['month_kwh'] or 0) / 1000

        # 3. Часовой график за сегодня
        cursor.execute("""
            SELECT
                EXTRACT(HOUR FROM created_at) as hour,
                SUM(packet_count) as count
            FROM sensor.data
            WHERE date = %s
            GROUP BY hour
            ORDER BY hour
        """, (today,))
        rows_hour = cursor.fetchall()

        # нормализуем 0–100%
        hours = [0]*24
        for r in rows_hour:
            h = int(r['hour'])
            if 0 <= h < 24:
                hours[h] = float(r['count'] or 0)

        max_count = max(hours) if max(hours) > 0 else 100
        hourly = [int(h / max_count * 100) for h in hours]

        # 4. AQI по дням за неделю (средний IAQ)
        cursor.execute("""
            SELECT
                date,
                AVG(iaq_class) as avg_iaq
            FROM sensor.data
            WHERE date BETWEEN %s AND %s
              AND iaq_class IS NOT NULL
            GROUP BY date
            ORDER BY date
        """, (week_start, today))
        rows_week = cursor.fetchall()

        # IAQ точки 0–120 (для SVG)
        points = []
        for _ in range(7):
            points.append(0)
        for i, r in enumerate(rows_week[-7:]):
            idx = 6 - (len(rows_week) - 1 - i)
            if 0 <= idx < 7:
                points[idx] = max(0, min(120, int(r['avg_iaq'] or 0) * 15))

        # 5. Потребление по зонам (room_id → zone)
        zone_names = [
            'Серверная', 'Открытый офис',
            'Конференц-зал', 'Лаборатория', 'Остальные'
        ]
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
        total_all = sum([r['total'] or 0 for r in rows_zone]) + 1
        percents = [0] * 5
        for r in rows_zone:
            idx = (r['room_id'] - 1) % 5
            percents[idx] = (r['total'] or 0) / total_all * 100

        # 6. Ответ клиенту
        return jsonify({
            # AnalyticsStats
            "today_kwh": today_kwh,
            "today_delta_pct": -8,       # пример: «↓ 8% vs вчера»
            "now_kw": now_kw,
            "month_kwh": month_kwh,
            "budget_kwh": 1500,
            "save_pct": 18,              # ↑ 18%

            # ConsumptionChart
            "hourly_consumption": hourly,

            # AQIChart
            "weekly_aqi": points,

            # ZonesConsumption
            "zones": [
                {"label": zone_names[i], "percent": percents[i]}
                for i in range(5)
            ]
        })

    except Exception as e:
        print(f"Ошибка /api/analytics: {e}")
        return jsonify({"error": "server_error"}), 500

#эндпоинт для страниц управления
'''@app.route('/api/control', methods=['GET'])
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

        # Пример: последние показания по каждой комнате → зоны
        cursor.execute("""
            SELECT
                room_id,
                aht21_temp as temp,
                packet_count % 100 as power,
                CASE
                    WHEN iaq_class >= 3 THEN 'on'
                    WHEN iaq_class = 2 THEN 'auto'
                    ELSE 'off'
                END as status
            FROM sensor.data
            WHERE id IN (
                SELECT MAX(id) FROM sensor.data GROUP BY room_id
            )
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Имя зон (можно позже хранить в таблице `sensor.zones`)
        zone_names = [
            "Конференц-зал",
            "Открытый офис",
            "Серверная",
            "Приёмная",
            "Лаборатория",
            "Столовая",
        ]

        zones = []
        for i, r in enumerate(rows):
            if i >= 6:
                break  # всего 6 зон в UI
            id = str(r['room_id'])
            zones.append({
                "id": id,
                "name": zone_names[i],
                "power": int(r['power'] or 70),
                "temp": float(r['temp'] or 22.0),
                "status": r['status'] or 'auto',
            })

        # Пример глобального режима
        global_mode = "auto"

        return jsonify({
            "global_mode": global_mode,
            "zones": zones
        })

    except Exception as e:
        print(f"Ошибка /api/control: {e}")
        return jsonify({"error": "server_error"}), 500'''

# эндпоинт для страниц управления
@app.route('/api/control', methods=['GET'])
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
            id = str(r['room_id'])
            zones.append({
                "id": id,
                "name": r['name'] or f"Комната {id}",
                "power": int(r['power'] or 70),
                "temp": float(r['temp'] or 22.0),
                "status": r['status'] or 'auto',
            })

        # Пример глобального режима
        global_mode = "auto"

        return jsonify({
            "global_mode": global_mode,
            "zones": zones
        })

    except Exception as e:
        print(f"Ошибка /api/control: {e}")
        return jsonify({"error": "server_error"}), 500
    
#Второй эндпоинт для старницы управления
@app.route('/api/control', methods=['PUT', 'POST'])
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

    print(f"Control action: {action}, target_temp: {target_temp}, fan_speed: {fan_speed}")

    # 1. обновляем целевое состояние в БД (пример таблицы `control_targets`)
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            if action == "all_on":
                cursor.execute("""
                    INSERT INTO control_targets (room_id, target_status, target_temp)
                    SELECT DISTINCT room_id, 'on', %s
                    FROM sensor.data
                    ON CONFLICT (room_id) DO UPDATE
                    SET target_status = 'on', target_temp = %s
                """, (target_temp, target_temp))
            elif action == "all_off":
                cursor.execute("""
                    INSERT INTO control_targets (room_id, target_status, target_temp)
                    SELECT DISTINCT room_id, 'off', %s
                    FROM sensor.data
                    ON CONFLICT (room_id) DO UPDATE
                    SET target_status = 'off', target_temp = %s
                """, (target_temp, target_temp))
            else:
                cursor.execute("""
                    UPDATE control_targets SET target_temp = %s, fan_speed = %s
                    WHERE target_temp IS NOT NULL
                """, (target_temp, fan_speed))

            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Control update error: {e}")
            conn.rollback()
        finally:
            conn.close()

    # ответ: просто текущее состояние (чтобы фронт обновился)
    return get_control_state()

'''
#эндпоинт для страницы логов
@app.route('/api/logs', methods=['GET'])
def get_logs():
    """
    Возвращает историю событий/логов:
    [
        { id, time, type, sensor, room, message, value }
    ]
    Параметры:
      ?from=YYYY-MM-DD
      ?to=YYYY-MM-DD
      ?type=alert|action|system|ml|all
      ?level=critical|warning|info|all
      ?limit=1000
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "database_error"}), 501

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # Параметры фильтра
        from_dt = request.args.get('from') or (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        to_dt   = request.args.get('to') or datetime.now().strftime('%Y-%m-%d')
        log_type = request.args.get('type') or 'all'
        level    = request.args.get('level') or 'all'
        limit    = int(request.args.get('limit') or 1000)

        # базовый запрос: логи из sensor.data по IAQ/ошибкам/действиям
        # здесь пример — логи из поля iaq_class/remark/кастом‑таблицы:
        query = """
            WITH raw_logs AS (
                SELECT
                    id,
                    created_at as created_at,
                    room_id,
                    temp, hum, ens_iaq, ens_tvoc, ens_co2,
                    iaq_class,
                    case
                        when iaq_class >= 4 then 'alert'
                        when iaq_class = 3 then 'warning'
                        when iaq_class = 2 then 'info'
                        else 'info'
                    end as category
                FROM sensor.data
                WHERE date BETWEEN %s AND %s
            )
            SELECT
                id,
                to_char(created_at, 'HH24:MI:SS') as time_,
                to_char(created_at::date, 'YYYY-MM-DD') as date_,
                room_id,

                -- тип события
                case
                    when category = 'alert' then 'alert'
                    when category = 'warning' then 'alert'
                    when iaq_class is not null then 'ml'
                    else 'system'
                end as log_type,

                -- уровень (уровень серьёзности)
                case
                    when iaq_class >= 4 then 'Critical'
                    when iaq_class >= 3 then 'Warning'
                    else 'Info'
                end as log_level,

                -- sensor и room (примерно)
                'S-' || room_id::text as sensor,
                room_id::text as room,

                -- сообщение
                case
                    when category = 'alert' and ens_co2 > 1000 then 'CO2 превышен'
                    when category = 'alert' and hum > 70 then 'Влажность превысила порог'
                    when iaq_class is not null then 'ML IAQ=' || iaq_class::text
                    else 'Системное событие'
                end as message,

                -- значение
                case
                    when ens_co2 > 0 then ens_co2::text || ' ppm'
                    when hum > 0 then hum::text || '%'
                    else null
                end as value_text
            FROM raw_logs
        """

        # фильтр по типу
        where_parts = []
        if log_type != 'all':
            where_parts.append(f" log_type = '{log_type}' ")
        if level != 'all':
            where_parts.append(f" log_level = '{level}' ")

        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)

        query += " ORDER BY id DESC LIMIT %s"

        cursor.execute(query, (from_dt, to_dt, limit))
        rows = cursor.fetchall()

        # форматируем под фронт‑логи
        logs = []
        for r in rows:
            logs.append({
                "id": str(r['id']),
                "time": r['date_'] + " " + r['time_'],
                "type": r['log_type'],
                "sensor": r['sensor'],
                "room": r['room'],
                "message": r['message'],
                "value": r['value_text'],
            })

        return jsonify({
            "logs": logs,
            "from": from_dt,
            "to": to_dt,
            "filter": {
                "type": log_type,
                "level": level
            }
        })

    except Exception as e:
        print(f"Ошибка /api/logs: {e}")
        return jsonify({"error": "server_error"}), 500

    finally:
        cursor.close()
        conn.close()    
'''
# эндпоинт для страницы логов
@app.route('/api/logs', methods=['GET'])
def get_logs():
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "database_error"}), 501

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        from_dt = request.args.get('from') or (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        to_dt   = request.args.get('to') or datetime.now().strftime('%Y-%m-%d')
        log_type = request.args.get('type') or 'all'
        level    = request.args.get('level') or 'all'
        limit    = int(request.args.get('limit') or 1000)

        where_parts = []
        params = [from_dt, to_dt]

        if log_type != 'all':
            where_parts.append("log_type = %s")
            params.append(log_type)

        if level != 'all':
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

        print(f"QUERY: {query}")
        print(f"PARAMS: {params}")
        print(f"PARAMS COUNT: {len(params)}, PLACEHOLDERS in query: {query.count('%s')}")  
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        logs = []
        for r in rows:
            logs.append({
                "id": str(r['id']),
                "time": r['date_'] + " " + r['time_'],
                "type": r['log_type'],
                "sensor": r['sensor'],
                "room": r['room'],
                "message": r['message'],
                "value": r['value_text'],
            })

        return jsonify({
            "logs": logs,
            "from": from_dt,
            "to": to_dt,
            "filter": {
                "type": log_type,
                "level": level
            }
        })

    except Exception as e:
        print(f"Ошибка /api/logs: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "server_error"}), 500

    finally:
        cursor.close()
        conn.close()


#Эндпоинт для страницы мониторинга
@app.route('/api/monitoring', methods=['GET'])
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
        cursor.execute("""
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
        """, (one_hour,))

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
                #critical += 1
                return "critical"
            if v >= 2:
                #warning += 1
                return "warning"
            #online += 1
            return "online"

        sensors = []
        for r in rows:
            iaq = r['avg_iaq']
            status = classify_iaq(iaq)
            
            # Обновляем счётчики здесь
            if status == "online":
                online += 1
            elif status == "warning":
                warning += 1
            elif status == "critical":
                critical += 1

            sensors.append({
                "id": f"S-{r['room_id']}",
                "room": f"room_{r['room_id']}",
                "co2": int(r['avg_co2'] or 400),
                "temp": float(r['avg_temp'] or 22.0),
                "hum": float(r['avg_hum'] or 50.0),
                "aqi": int(iaq or 0),#max(1, int(iaq or 0) * 25),  # 0..100
                "status": status,
            })

        summary = {
            "total": total,
            "online": online,
            "warning": warning,
            "critical": critical
        }

        return jsonify({
            "summary": summary,
            "sensors": sensors
        })

    except Exception as e:
        print(f"Ошибка /api/monitoring: {e}")
        return jsonify({"error": "server_error"}), 500
    
#Эндпоинт для страницы настроек
@app.route('/api/settings', methods=['GET'])
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
            "threshold_moderate": 60
        },
        "system": {
            "log_level": "INFO",
            "db_host": "localhost",
            "db_port": 5432
        },
        "notifications": {
            "email_enabled": True,
            "tg_enabled": False,
            "web_push": True
        }
    }

    # 2. модельная метаинформация
    ml_model_info = {
        "name": "AeroML v2.4",
        "version": "2.4.1",
        "accuracy": 97.8,
        "train_date": "2026-04-10",
        "dataset_size": 128400,
        "features": [
            "CO2", "Temperature", "Humidity", "PM2.5", "VOC", "TimeOfDay", "Occupancy"
        ]
    }

    return jsonify({
        "settings": settings,
        "model_info": ml_model_info
    })

#Второй эндпоинт для страницы настроек
@app.route('/api/settings', methods=['PUT', 'POST'])
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
    print(f"ML settings updated: {ml_settings}")
    print(f"System settings updated: {system_settings}")
    print(f"Notifications settings updated: {notifications_settings}")

    # ответ — полный актуальный объект
    ml_model_info = {
        "name": "AeroML v2.4",
        "version": "2.4.1",
        "accuracy": 97.8,
        "train_date": "2026-04-10",
        "dataset_size": 128400,
        "features": [
            "CO2", "Temperature", "Humidity", "PM2.5", "VOC", "TimeOfDay", "Occupancy"
        ]
    }

    return jsonify({
        "settings": data,
        "model_info": ml_model_info
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)