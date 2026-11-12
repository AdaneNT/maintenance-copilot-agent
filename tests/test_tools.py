import re
from pathlib import Path

import pytest

from maint_agent.tools import VIBRATION_ZONES, ToolError, vibration_zone


def test_list_equipment(toolkit):
    out = toolkit.list_equipment()
    assert out["equipment"] == ["C-201", "HX-301", "P-101", "P-102"]


def test_sensor_summary_detects_p102_degradation(toolkit):
    out = toolkit.sensor_summary("P-102", "vibration_mm_s", last_days=7)
    assert out["latest_vibration_zone"] == "D"
    assert out["trend_per_day"] > 0.5
    assert out["source"] == "tool:sensor_summary/P-102/vibration_mm_s/7d"


def test_sensor_summary_reports_when_the_maximum_occurred(toolkit):
    out = toolkit.sensor_summary("C-201", "bearing_temp_c", last_days=14)
    assert out["max"] > 90
    assert out["max_at"].startswith("2026-09-20")


def test_healthy_pump_is_zone_a(toolkit):
    out = toolkit.sensor_summary("p-101", "vibration_mm_s")  # case-insensitive id
    assert out["latest_vibration_zone"] == "A"
    assert abs(out["trend_per_day"]) < 0.05


def test_anomalies_find_injected_faults_only(toolkit):
    p102 = toolkit.detect_anomalies("P-102")
    assert any(a["sensor"] == "vibration_mm_s" and a["direction"] == "high" for a in p102["anomalies"])

    c201 = toolkit.detect_anomalies("C-201")
    temp = [a for a in c201["anomalies"] if a["sensor"] == "bearing_temp_c"]
    assert len(temp) == 1 and temp[0]["start"].startswith("2026-09-20")

    # healthy equipment: noise spikes are filtered out
    assert toolkit.detect_anomalies("P-101")["anomalies"] == []
    assert toolkit.detect_anomalies("HX-301")["anomalies"] == []


def test_fleet_overview_ranks_faulty_equipment_first(toolkit):
    out = toolkit.fleet_overview()
    assert out["source"] == "tool:fleet_overview/14d"
    assert set(out["abnormal_equipment"]) == {"P-102", "C-201"}
    assert [r["equipment_id"] for r in out["equipment"]][:2] == ["P-102", "C-201"]
    assert {r["equipment_id"]: r["status"] for r in out["equipment"]}["P-101"] == "normal"


def test_work_orders_filter(toolkit):
    out = toolkit.get_work_orders("C-201", status="open")
    assert [w["work_order_id"] for w in out["work_orders"]] == ["WO-1005"]


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.sensor_summary("X-999", "vibration_mm_s"),
        lambda t: t.sensor_summary("P-101", "rpm"),
        lambda t: t.sensor_summary("P-101", "vibration_mm_s", last_days=0),
        lambda t: t.search_procedures("   "),
    ],
)
def test_invalid_input_raises_tool_error(toolkit, call):
    with pytest.raises(ToolError):
        call(toolkit)


def test_vibration_zones_match_guideline_document():
    """The hard-coded zone limits must stay in sync with the procedure document."""
    doc = (Path(__file__).resolve().parents[1] / "data/docs/vibration_limits.md").read_text()
    limits = sorted({float(x) for x in re.findall(r"(\d+\.\d+) mm/s", doc)})
    assert limits == [u for u, _ in VIBRATION_ZONES if u != float("inf")]
    assert [vibration_zone(v) for v in (1.0, 3.0, 5.0, 8.0)] == ["A", "B", "C", "D"]
