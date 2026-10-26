-- Инициализация БД MLAIRQuality.
-- Выполняется контейнером postgres один раз при первом старте (пустой том).
-- Исправлено относительно DA/sql DB/*.sql:
--   * создаётся схема sensor (раньше её не было — CREATE TABLE sensor.data падал);
--   * все таблицы в схеме sensor (раньше rooms/control_targets создавались в public,
--     а INSERT шёл в sensor.rooms);
--   * индекс по room_id вместо несуществующего room_number;
--   * iaq_class/iaq_proba сразу в CREATE TABLE ("alter col.sql" больше не нужен).

CREATE SCHEMA IF NOT EXISTS sensor;

CREATE TABLE IF NOT EXISTS sensor.rooms (
    room_id    SERIAL PRIMARY KEY,
    name_room  VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sensor.data (
    id            SERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    room_id       INTEGER NOT NULL,
    date          DATE NOT NULL,
    time          TIME NOT NULL,
    season        INTEGER NOT NULL,
    weekday       INTEGER NOT NULL,
    temp          DECIMAL(5,2),
    hum           DECIMAL(5,2),
    mq7           DECIMAL(8,2),
    mq135         DECIMAL(8,2),
    ky028_analog  INTEGER,
    ky028_digital INTEGER,
    bmp_temp      DECIMAL(5,2),
    pressure      DECIMAL(8,2),
    altitude      DECIMAL(8,2),
    aht21_temp    DECIMAL(5,2),
    aht21_hum     DECIMAL(5,2),
    ens_iaq       INTEGER,
    ens_tvoc      INTEGER,
    ens_co2       INTEGER,
    packet_count  INTEGER NOT NULL,
    iaq_class     INTEGER,
    iaq_proba     JSONB
);

CREATE INDEX IF NOT EXISTS idx_sensor_data_room_id    ON sensor.data(room_id);
CREATE INDEX IF NOT EXISTS idx_sensor_data_date       ON sensor.data(date);
CREATE INDEX IF NOT EXISTS idx_sensor_data_created_at ON sensor.data(created_at);

CREATE TABLE IF NOT EXISTS sensor.control_targets (
    id            SERIAL PRIMARY KEY,
    room_id       INTEGER NOT NULL UNIQUE,
    target_status VARCHAR(10) NOT NULL CHECK (target_status IN ('on', 'off', 'auto')),
    target_temp   DECIMAL(5,2) NOT NULL DEFAULT 22.0,
    fan_speed     INTEGER DEFAULT 1,
    created_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO sensor.rooms (name_room) VALUES
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq'),
    ('sensor.rooms_room_id_seq')
ON CONFLICT DO NOTHING;
