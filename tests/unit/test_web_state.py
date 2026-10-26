from airquality.web.state import SensorState


def test_rooms_do_not_mix() -> None:
    state = SensorState()
    state.update(1, {"temp": 20.0, "hum": 40.0})
    room2, packet = state.update(2, {"temp": 25.0})
    assert packet == 2
    assert room2["temp"] == 25.0
    assert room2["hum"] == 0.0  # влажность комнаты 1 не «протекла» в комнату 2


def test_unknown_keys_ignored() -> None:
    room, _ = SensorState().update(1, {"hack": 1, "temp": 21.0})
    assert "hack" not in room
