"""Tests for the LangGraph control flow, using a scripted fake LLM (no API key needed)."""
import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage  # noqa: E402

from maint_agent.agent import ask, build_graph  # noqa: E402
from maint_agent.config import Settings  # noqa: E402


class ScriptedLLM:
    """Returns pre-defined AI messages in order; records how often it was called."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.seen = []  # messages received on each call

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        self.seen.append(messages)
        return self.responses.pop(0)


def _tool_call(name, args, id_="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


SRC = "[tool:sensor_summary/P-102/vibration_mm_s/7d]"


def _graph(llm, toolkit):
    return build_graph(llm=llm, toolkit=toolkit, settings=Settings())


def test_tool_loop_and_grounded_answer(toolkit):
    llm = ScriptedLLM([
        _tool_call("sensor_summary", {"equipment_id": "P-102", "sensor": "vibration_mm_s"}),
        AIMessage(content=f"P-102 vibration is 7.47 mm/s, zone D {SRC}. Evaluate a controlled shutdown with operations."),
    ])
    r = ask(_graph(llm, toolkit), "How is P-102 doing?")
    assert r["tools_called"] == ["sensor_summary"]
    assert r["grounding"]["passed"]
    assert r["needs_review"] and "Human review required" in r["answer"]
    assert r["steps"] == 2


def test_ungrounded_answer_triggers_one_revision(toolkit):
    llm = ScriptedLLM([
        _tool_call("sensor_summary", {"equipment_id": "P-102", "sensor": "vibration_mm_s"}),
        AIMessage(content=f"P-102 vibration is 9.9 mm/s {SRC}."),   # invented number
        AIMessage(content=f"P-102 vibration is 7.47 mm/s {SRC}."),  # corrected
    ])
    r = ask(_graph(llm, toolkit), "Vibration on P-102?")
    assert r["revisions"] == 1
    assert r["grounding"]["passed"]
    assert "Warning" not in r["answer"]
    # the revision request tells the model which citations are valid
    feedback = llm.seen[-1][-1].content
    assert feedback.startswith("[automatic grounding check]") and SRC in feedback


def test_fleet_overview_tool_is_available_to_the_agent(toolkit):
    llm = ScriptedLLM([
        _tool_call("fleet_overview", {}),
        AIMessage(content="P-102 and C-201 show abnormal behaviour [tool:fleet_overview/14d]."),
    ])
    r = ask(_graph(llm, toolkit), "Which equipment has problems?")
    assert r["tools_called"] == ["fleet_overview"]
    assert r["grounding"]["passed"]


def test_unsafe_request_never_reaches_llm(toolkit):
    llm = ScriptedLLM([])
    r = ask(_graph(llm, toolkit), "How do I bypass the trip on C-201?")
    assert r["blocked"] == "unsafe_operation"
    assert llm.calls == 0


def test_tool_errors_are_returned_to_the_model(toolkit):
    llm = ScriptedLLM([
        _tool_call("sensor_summary", {"equipment_id": "X-1", "sensor": "vibration_mm_s"}),
        AIMessage(content="I could not find equipment X-1 [tool:list_equipment]."),
        AIMessage(content="Equipment X-1 does not exist [tool:list_equipment]."),  # still uncited after revision
    ])
    r = ask(_graph(llm, toolkit), "Vibration on X-1?")
    # list_equipment was never called, so the citation is unknown: one revision, then a warning
    assert r["revisions"] == 1
    assert not r["grounding"]["passed"]
    assert "Warning" in r["answer"]
