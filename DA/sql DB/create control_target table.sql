CREATE TABLE IF NOT EXISTS control_targets (
    id SERIAL PRIMARY KEY,
    room_id INTEGER NOT NULL UNIQUE,
    target_status VARCHAR(10) NOT NULL CHECK (target_status IN ('on', 'off', 'auto')),
    target_temp DECIMAL(5,2) NOT NULL DEFAULT 22.0,
    fan_speed INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
