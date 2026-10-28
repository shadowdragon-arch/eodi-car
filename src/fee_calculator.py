"""
EodiCar ANPR Prototype - Samsan Tech Sdn. Bhd.
Milestone 6: Parking Timer and Configurable Fee Engine
"""

import json
import os
import math
from datetime import datetime

CONFIG_PATH = os.path.join("config", "tariff_config.json")

DEFAULT_TARIFF = {
    "currency": "RM",
    "weekday_first_hour": 2.00,
    "weekday_subsequent_hour": 1.00,
    "weekend_first_hour": 3.00,
    "weekend_subsequent_hour": 2.00,
}


def load_tariff() -> dict:
    """Loads parking rates from JSON config, auto-creating default config if missing."""
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_TARIFF, f, indent=4)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_fee(entry_str: str, exit_str: str) -> tuple:
    """
    Calculates the parking duration and the estimated fee.
    Partial hours are rounded up (e.g., 61 minutes = 2 hours).
    """
    fmt = "%Y-%m-%d %H:%M:%S"
    entry_dt = datetime.strptime(entry_str, fmt)
    exit_dt = datetime.strptime(exit_str, fmt)

    duration_delta = exit_dt - entry_dt
    duration_hours = math.ceil(duration_delta.total_seconds() / 3600.0)

    if duration_hours <= 0:
        duration_hours = 1

    tariff = load_tariff()
    is_weekend = entry_dt.weekday() >= 5

    if is_weekend:
        first_hr_rate = float(tariff.get("weekend_first_hour", 3.00))
        sub_hr_rate = float(tariff.get("weekend_subsequent_hour", 2.00))
        day_type = "Weekend/Holiday"
    else:
        first_hr_rate = float(tariff.get("weekday_first_hour", 2.00))
        sub_hr_rate = float(tariff.get("weekday_subsequent_hour", 1.00))
        day_type = "Weekday"

    total_fee = first_hr_rate + (sub_hr_rate * (duration_hours - 1))
    return duration_hours, total_fee, day_type