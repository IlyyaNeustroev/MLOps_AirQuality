import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# 1. Константы и параметры
start_date = datetime(2026, 4, 27, 0, 0, 0)
end_date = datetime(2027, 4, 27, 23, 0, 0)
delta = timedelta(minutes=5)  # 1 запись каждые 5 минут
NUM_ROOMS = 10
df_rows = []

# Вероятности ошибок датчиков (в %)
SENSOR_FAILURES = {
    'temp': 0.02, 'hum': 0.02, 'mq7': 0.05, 'mq135': 0.05,
    'ky028_analog': 0.03, 'ky028_digital': 0.03,
    'bmp_temp': 0.04, 'pressure': 0.04, 'altitude': 0.04,
    'aht21_temp': 0.02, 'aht21_hum': 0.02,
    'ens_iaq': 0.06, 'ens_tvoc': 0.06, 'ens_co2': 0.06
}

# 2. Улучшенная функция IAQ 
def compute_iaq(mq7, mq135, temp, hum, ky028_analog, ky028_digital, 
                bmp_temp, pressure, altitude, aht21_temp, aht21_hum,
                ens_iaq, ens_tvoc, ens_co2, season, weekday, hour, room_id):
    # СРЕДНИЕ значения для сбоев (-1) вместо 1.0
    MEAN_VALUES = {
        'mq7': 120,        # Средний CO
        'mq135': 110,      # Средний NH3/CO2  
        'temp': 22.0,      # Комфортная t
        'hum': 45.0,       # Комфортная влажность
        'ens_co2': 800,    # Средний eCO2
        'ens_tvoc': 250,   # Средний TVOC
        'ens_iaq': 75,     # Средний IAQ
        'ky028_analog': 35 # Средняя температура KY
    }
    
    # Заменяем -1 на средние
    def safe_value(val, mean_key):
        return MEAN_VALUES.get(mean_key, 0.0) if val == -1 else val
    
    mq7_safe = safe_value(mq7, 'mq7')
    mq135_safe = safe_value(mq135, 'mq135')
    temp_safe = safe_value(temp, 'temp')
    hum_safe = safe_value(hum, 'hum')
    ens_co2_safe = safe_value(ens_co2, 'ens_co2')
    ens_tvoc_safe = safe_value(ens_tvoc, 'ens_tvoc')
    ens_iaq_safe = safe_value(ens_iaq, 'ens_iaq')
    ky028_safe = safe_value(ky028_analog, 'ky028_analog')
    
    # Нормализация с "средними" значениями
    co_norm = np.clip((mq7_safe - 100) / 60, 0, 1)      # ~0.33 (нормально)
    nh3_norm = np.clip(mq135_safe / 160, 0, 1)          # ~0.69 (средне)
    co2_norm = np.clip((ens_co2_safe - 400) / 3600, 0, 1)  # ~0.11 (хорошо)
    tvoc_norm = np.clip(ens_tvoc_safe / 1000, 0, 1)     # ~0.25 (нормально)
    
    # Комфорт
    t_norm = np.clip(abs(temp_safe - 22) / 10, 0, 1)     # 0.0 (идеально)
    rh_norm = np.clip(abs(hum_safe - 45) / 25, 0, 1)     # 0.0 (идеально)
    comfort_norm = max(t_norm, rh_norm)
    
    # Взвешивание с пиковыми нагрузками
    weights = {'co': 0.25, 'nh3': 0.15, 'co2': 0.25, 'tvoc': 0.20, 'comfort': 0.15}
    
    peak_multiplier = 1.0
    if weekday == 7: peak_multiplier = 1.8   # Воскресенье
    elif weekday == 6: peak_multiplier = 1.4 # Суббота
    if season in [2,3]: peak_multiplier *= 1.25  # Лето/весна
    if 11 <= hour <= 15: peak_multiplier *= 1.3  # Обед
    
    iaq_score = np.max([
        weights['co'] * co_norm * peak_multiplier,
        weights['nh3'] * nh3_norm,
        weights['co2'] * co2_norm * peak_multiplier,
        weights['tvoc'] * tvoc_norm * peak_multiplier,
        weights['comfort'] * comfort_norm
    ])
    
    # Класс IAQ 0-5
    if iaq_score < 0.15: return 0
    elif iaq_score < 0.30: return 1
    elif iaq_score < 0.50: return 2
    elif iaq_score < 0.70: return 3
    elif iaq_score < 0.90: return 4
    else: return 5

# 3. Генератор с реалистичными паттернами и сбоями
current = start_date
while current <= end_date:
    year, month, day = current.year, current.month, current.day
    weekday = current.isoweekday()  # 1-7 (понедельник=1)
    hour, minute = current.hour, current.minute
    
    # Сезон
    season = 2 if month in [3,4,5] else 3 if month in [6,7,8] else 4 if month in [9,10,11] else 1
    
    date_str = f"{day:02d}.{month:02d}.{year}"
    time_str = f"{hour:02d}:{minute:02d}:00"
    
    for room_id in range(1, 11):
        # Модификатор загруженности
        crowd_factor = 1.0
        if weekday >= 6:  # Выходные
            crowd_factor = 1.8 if weekday == 7 else 1.4
        if season in [2,3]:  # Весна-лето
            crowd_factor *= 1.25
        if 10 <= hour <= 16:  # Рабочие часы
            crowd_factor *= 1.2
        
        # Базовые значения с учетом загруженности
        temp = 22 + np.random.normal(0, 1.5) * crowd_factor * 0.1
        hum = 45 + np.random.normal(0, 8) * crowd_factor * 0.15
        
        # Газовые датчики растут с загруженностью
        mq7 = max(100, 120 + 40 * crowd_factor + np.random.normal(0, 10))
        mq135 = max(100, 110 + 30 * crowd_factor + np.random.normal(0, 8))
        
        # KY-028 (аналоговый термометр + цифровой)
        ky028_analog = int(25 + 40 * np.random.random() + crowd_factor * 5)
        ky028_digital = 1 if ky028_analog > 40 else 0
        
        # BMP280 (улица)
        t_outdoor = 5 + 20 * np.sin(2 * np.pi * (month-1) / 12)
        pressure = 1013 + np.random.normal(0, 8)
        altitude = 82 + np.random.normal(0, 2)
        
        # AHT21 (резервный)
        aht21_temp = temp + np.random.normal(0, 0.3)
        aht21_hum = hum + np.random.normal(0, 1.5)
        
        # ENS160
        ens_iaq = int(50 + 30 * crowd_factor + np.random.normal(0, 10))
        ens_tvoc = int(100 + 300 * crowd_factor + np.random.normal(0, 50))
        ens_co2 = int(450 + 800 * crowd_factor + np.random.normal(0, 100))
        
        # Добавляем сбои датчиков (-1)
        row_data = {
            'date': date_str,
            'time': time_str,
            'season': season,
            'weekday': weekday,
            'room_id': room_id,
            'temp': temp if random.random() > SENSOR_FAILURES['temp'] else -1,
            'hum': hum if random.random() > SENSOR_FAILURES['hum'] else -1,
            'mq7': int(mq7) if random.random() > SENSOR_FAILURES['mq7'] else -1,
            'mq135': int(mq135) if random.random() > SENSOR_FAILURES['mq135'] else -1,
            'ky028_analog': ky028_analog if random.random() > SENSOR_FAILURES['ky028_analog'] else -1,
            'ky028_digital': ky028_digital if random.random() > SENSOR_FAILURES['ky028_digital'] else -1,
            'bmp_temp': t_outdoor if random.random() > SENSOR_FAILURES['bmp_temp'] else -1,
            'pressure': pressure if random.random() > SENSOR_FAILURES['pressure'] else -1,
            'altitude': altitude if random.random() > SENSOR_FAILURES['altitude'] else -1,
            'aht21_temp': aht21_temp if random.random() > SENSOR_FAILURES['aht21_temp'] else -1,
            'aht21_hum': aht21_hum if random.random() > SENSOR_FAILURES['aht21_hum'] else -1,
            'ens_iaq': ens_iaq if random.random() > SENSOR_FAILURES['ens_iaq'] else -1,
            'ens_tvoc': ens_tvoc if random.random() > SENSOR_FAILURES['ens_tvoc'] else -1,
            'ens_co2': ens_co2 if random.random() > SENSOR_FAILURES['ens_co2'] else -1,
            'packet_count': 0  # Заполнится Flask
        }
        
        # Целевая переменная IAQ класс
        iaq_class = compute_iaq(
            row_data['mq7'], row_data['mq135'], row_data['temp'], row_data['hum'],
            row_data['ky028_analog'], row_data['ky028_digital'],
            row_data['bmp_temp'], row_data['pressure'], row_data['altitude'],
            row_data['aht21_temp'], row_data['aht21_hum'],
            row_data['ens_iaq'], row_data['ens_tvoc'], row_data['ens_co2'],
            season, weekday, hour, room_id
        )
        
        # Формируем строку для CSV в формате  БД
        row = [
            row_data['date'], row_data['weekday'], row_data['season'], row_data['time'],
            row_data['room_id'], row_data['mq7'], row_data['mq135'],
            round(row_data['temp'], 1), round(row_data['hum'], 1),
            row_data['ky028_analog'], row_data['ky028_digital'],
            round(row_data['bmp_temp'], 1), round(row_data['pressure'], 1), round(row_data['altitude'], 1),
            round(row_data['aht21_temp'], 1), round(row_data['aht21_hum'], 1),
            row_data['ens_iaq'], row_data['ens_tvoc'], row_data['ens_co2'],
            iaq_class  # Для ML обучения
        ]
        df_rows.append(row)
    
    current += delta

# 4. Сохраняем в формате БД
columns = [
    "date", "weekday", "season", "time", "room_id",
    "mq7", "mq135", "temp", "hum", 
    "ky028_analog", "ky028_digital",
    "bmp_temp", "pressure", "altitude",
    "aht21_temp", "aht21_hum",
    "ens_iaq", "ens_tvoc", "ens_co2",
    "iaq_class"  # Целевая переменная ML
]

df = pd.DataFrame(df_rows, columns=columns)
df.to_csv("air_quality_training_dataset_2.csv", index=False, float_format="%.1f")

print(f"Датасет сгенерирован: {len(df):,} записей")
print("\nСтатистика:")
print(df[['mq7', 'temp', 'ens_co2', 'iaq_class']].describe())
print(f"\n🔧 Сбои датчиков (-1):")
print((df == -1).sum())