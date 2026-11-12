"""Deterministic, read-only tools the agent can call.

Design principle: numbers and rules that must be exact (statistics, thresholds,
anomaly detection) are computed in code, not by the language model. Every tool
returns a `source` id so the final answer can cite where each fact came from.
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .retrieval import Retriever

SENSORS = ("vibration_mm_s", "bearing_temp_c", "discharge_pressure_bar", "flow_m3_h")

# Zones from data/docs/vibration_limits.md (kept in sync by tests/test_tools.py)
VIBRATION_ZONES = ((2.8, "A"), (4.5, "B"), (7.1, "C"), (float("inf"), "D"))


def vibration_zone(value_mm_s: float) -> str:
    for upper, zone in VIBRATION_ZONES:
        if value_mm_s < upper:
            return zone
    return "D"


class ToolError(ValueError):
    """Raised for invalid tool input; the message is shown to the agent."""


class MaintenanceToolkit:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    # ---------- data loading (lazy) ----------
    @cached_property
    def sensors(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "sensors.csv", parse_dates=["timestamp"])

    @cached_property
    def work_orders(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "work_orders.csv", parse_dates=["date"])

    @cached_property
    def retriever(self) -> Retriever:
        return Retriever.from_dir(self.data_dir / "docs")

    # ---------- validation ----------
    def _check_equipment(self, equipment_id: str) -> str:
        eq = equipment_id.strip().upper()
        known = set(self.sensors["equipment_id"].unique())
        if eq not in known:
            raise ToolError(f"Unknown equipment_id '{equipment_id}'. Known: {sorted(known)}")
        return eq

    @staticmethod
    def _check_sensor(sensor: str) -> str:
        if sensor not in SENSORS:
            raise ToolError(f"Unknown sensor '{sensor}'. Known: {list(SENSORS)}")
        return sensor

    def _window(self, eq: str, last_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not 1 <= last_days <= 28:
            raise ToolError("last_days must be between 1 and 28")
        df = self.sensors[self.sensors["equipment_id"] == eq]
        cutoff = df["timestamp"].max() - pd.Timedelta(days=last_days)
        return df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff]

    # ---------- tools ----------
    def list_equipment(self) -> dict:
        eq = sorted(self.sensors["equipment_id"].unique())
        return {
            "source": "tool:list_equipment",
            "equipment": eq,
            "sensors": list(SENSORS),
            "data_period": [str(self.sensors["timestamp"].min()), str(self.sensors["timestamp"].max())],
        }

    def sensor_summary(self, equipment_id: str, sensor: str, last_days: int = 7) -> dict:
        eq = self._check_equipment(equipment_id)
        sensor = self._check_sensor(sensor)
        _, win = self._window(eq, last_days)
        s = win[sensor].to_numpy()
        days = (win["timestamp"] - win["timestamp"].min()).dt.total_seconds().to_numpy() / 86400
        slope = float(np.polyfit(days, s, 1)[0]) if len(s) > 1 else 0.0
        first_day = win[win["timestamp"] < win["timestamp"].min() + pd.Timedelta(days=1)][sensor].mean()
        last_day = win[win["timestamp"] > win["timestamp"].max() - pd.Timedelta(days=1)][sensor].mean()
        out = {
            "source": f"tool:sensor_summary/{eq}/{sensor}/{last_days}d",
            "equipment_id": eq,
            "sensor": sensor,
            "window_days": last_days,
            "latest": round(float(s[-1]), 2),
            "mean": round(float(s.mean()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "max_at": str(win["timestamp"].iloc[int(s.argmax())]),
            "trend_per_day": round(slope, 3),
            "change_pct_first_vs_last_day": round(float((last_day - first_day) / first_day * 100), 1),
        }
        if sensor == "vibration_mm_s":
            out["latest_vibration_zone"] = vibration_zone(out["latest"])
        return out

    def detect_anomalies(
        self,
        equipment_id: str,
        last_days: int = 14,
        z_threshold: float = 3.0,
        min_hours: int = 3,
        max_gap_hours: int = 6,
    ) -> dict:
        """Find sustained deviations from the baseline in the recent window.

        The baseline is the data *before* the window. Readings with |z| above the
        threshold are grouped into episodes (gaps of up to `max_gap_hours` are
        bridged, e.g. dips caused by a daily temperature cycle). Episodes shorter than `min_hours` are treated as noise and only
        counted, which keeps isolated spikes from producing false alarms.
        """
        eq = self._check_equipment(equipment_id)
        base, win = self._window(eq, last_days)
        if len(base) < 48:
            raise ToolError("Not enough baseline data; use a smaller last_days")
        findings, ignored_spikes = [], 0
        for sensor in SENSORS:
            mu, sd = base[sensor].mean(), base[sensor].std(ddof=1)
            z = (win[sensor] - mu) / sd
            flagged = win.loc[z.abs() > z_threshold, "timestamp"]
            if flagged.empty:
                continue
            # new episode whenever the gap to the previous flagged hour is too large
            episode_id = (flagged.diff() > pd.Timedelta(hours=max_gap_hours)).cumsum()
            for _, ts in flagged.groupby(episode_id):
                if len(ts) < min_hours:
                    ignored_spikes += len(ts)
                    continue
                zs = z.loc[ts.index]
                peak = zs.abs().idxmax()
                findings.append({
                    "sensor": sensor,
                    "start": str(ts.min()),
                    "end": str(ts.max()),
                    "anomalous_hours": int(len(ts)),
                    "direction": "high" if zs.loc[peak] > 0 else "low",
                    "peak_value": round(float(win.loc[peak, sensor]), 2),
                    "peak_z": round(float(zs.loc[peak]), 1),
                    "baseline_mean": round(float(mu), 2),
                })
        return {
            "source": f"tool:detect_anomalies/{eq}/{last_days}d",
            "equipment_id": eq,
            "window_days": last_days,
            "z_threshold": z_threshold,
            "anomalies": findings,
            "isolated_spikes_ignored": ignored_spikes,
        }

    def fleet_overview(self, last_days: int = 14) -> dict:
        """Run anomaly detection on every piece of equipment and rank them by severity.

        One call answers "which equipment has problems?", which is easier for small
        models than planning one detect_anomalies call per equipment.
        """
        rows = []
        for eq in sorted(self.sensors["equipment_id"].unique()):
            found = self.detect_anomalies(eq, last_days)["anomalies"]
            rows.append({
                "equipment_id": eq,
                "status": "abnormal" if found else "normal",
                "max_abs_z": max((abs(a["peak_z"]) for a in found), default=0.0),
                "anomalies": [
                    {k: a[k] for k in ("sensor", "direction", "start", "end", "peak_value", "baseline_mean")}
                    for a in found
                ],
            })
        rows.sort(key=lambda r: r["max_abs_z"], reverse=True)
        return {
            "source": f"tool:fleet_overview/{last_days}d",
            "window_days": last_days,
            "abnormal_equipment": [r["equipment_id"] for r in rows if r["status"] == "abnormal"],
            "equipment": rows,
        }

    def get_work_orders(self, equipment_id: str, status: str | None = None) -> dict:
        eq = self._check_equipment(equipment_id)
        wo = self.work_orders[self.work_orders["equipment_id"] == eq]
        if status:
            wo = wo[wo["status"] == status.lower()]
        records = wo.sort_values("date").assign(date=lambda d: d["date"].dt.date.astype(str)).to_dict("records")
        return {"source": f"tool:work_orders/{eq}", "equipment_id": eq, "work_orders": records}

    def search_procedures(self, query: str, k: int = 3) -> dict:
        if not query.strip():
            raise ToolError("query must not be empty")
        hits = self.retriever.search(query, k=k)
        return {"source": "tool:search_procedures", "query": query, "results": [h.to_dict() for h in hits]}
