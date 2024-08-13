from typing import NamedTuple
from datetime import datetime

class TemperaturePoint(NamedTuple):
    time: datetime
    temperature: float

class HouseData(NamedTuple):
    outside_temperatures: list[TemperaturePoint]
    room_temperatures: dict[str, dict[datetime, float]]
    room_setpoints: dict[str, dict[datetime, float]]
    gas_readings: dict[datetime, float]
