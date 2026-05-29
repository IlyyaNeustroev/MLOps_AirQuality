

-- Создание таблицы для хранения данных с датчиков
CREATE TABLE IF NOT EXISTS sensor.data (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    room_id INTEGER NOT NULL,
    date DATE NOT NULL,
    time TIME NOT NULL,
    season INTEGER NOT NULL,
    weekday INTEGER NOT NULL,
    temp DECIMAL(5,2),
    hum DECIMAL(5,2),
    mq7 DECIMAL(8,2),
    mq135 DECIMAL(8,2),
    ky028_analog INTEGER,
    ky028_digital INTEGER,
    bmp_temp DECIMAL(5,2),
    pressure DECIMAL(8,2),
    altitude DECIMAL(8,2),
    aht21_temp DECIMAL(5,2),
    aht21_hum DECIMAL(5,2),
    ens_iaq INTEGER,
    ens_tvoc INTEGER,
    ens_co2 INTEGER,
    packet_count INTEGER NOT NULL,
    iaq_class INTEGER,
    iaq_proba JSONB
);

-- Индексы для ускорения запросов
CREATE INDEX IF NOT EXISTS idx_sensor_data_room_number ON sensor.data(room_number);
CREATE INDEX IF NOT EXISTS idx_sensor_data_date ON sensor.data(date);
CREATE INDEX IF NOT EXISTS idx_sensor_data_created_at ON sensor.data(created_at);
