"""Generate the synthetic sensor dataset used by the demo.

The data is fully synthetic (fixed random seed) and only meant to give the
agent something realistic to reason over. Two faults are injected on purpose:

* P-102: gradually increasing vibration over the last 5 days (bearing wear pattern)
* C-201: bearing temperature excursion on day 20 lasting ~10 hours

Run:  python data/generate_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DAYS = 30
START = "2026-09-01"

EQUIPMENT = {
    # id: (type, baseline per sensor)
    "P-101": ("centrifugal pump", {"vibration_mm_s": 2.1, "bearing_temp_c": 58.0, "discharge_pressure_bar": 12.5, "flow_m3_h": 180.0}),
    "P-102": ("centrifugal pump", {"vibration_mm_s": 2.3, "bearing_temp_c": 60.0, "discharge_pressure_bar": 12.2, "flow_m3_h": 175.0}),
    "C-201": ("gas compressor", {"vibration_mm_s": 3.0, "bearing_temp_c": 72.0, "discharge_pressure_bar": 45.0, "flow_m3_h": 950.0}),
    "HX-301": ("heat exchanger", {"vibration_mm_s": 0.6, "bearing_temp_c": 40.0, "discharge_pressure_bar": 8.0, "flow_m3_h": 220.0}),
}

NOISE = {"vibration_mm_s": 0.12, "bearing_temp_c": 0.8, "discharge_pressure_bar": 0.15, "flow_m3_h": 2.5}


def generate(out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    ts = pd.date_range(START, periods=DAYS * 24, freq="h")
    rows = []
    for eq_id, (_, base) in EQUIPMENT.items():
        frame = {"timestamp": ts, "equipment_id": eq_id}
        for sensor, mu in base.items():
            values = mu + rng.normal(0, NOISE[sensor], len(ts))
            # mild daily cycle on temperature
            if sensor == "bearing_temp_c":
                values += 1.0 * np.sin(2 * np.pi * np.arange(len(ts)) / 24)
            frame[sensor] = values
        df = pd.DataFrame(frame)

        if eq_id == "P-102":
            # vibration ramps from 2.3 to ~7.3 mm/s over the last 5 days
            ramp_start = (DAYS - 5) * 24
            n = len(ts) - ramp_start
            df.loc[ramp_start:, "vibration_mm_s"] += np.linspace(0, 5.0, n)
            df.loc[ramp_start:, "bearing_temp_c"] += np.linspace(0, 6.0, n)

        if eq_id == "C-201":
            # bearing temperature excursion on day 20, 10 hours long
            s = 19 * 24 + 8
            df.loc[s : s + 9, "bearing_temp_c"] += 18.0

        rows.append(df)

    data = pd.concat(rows, ignore_index=True)
    sensors = list(NOISE)
    data[sensors] = data[sensors].round(3)
    out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_dir / "sensors.csv", index=False)
    return data


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    d = generate(here)
    print(f"Wrote {len(d)} rows to {here / 'sensors.csv'}")
