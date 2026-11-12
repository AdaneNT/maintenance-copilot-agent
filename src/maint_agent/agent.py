"""LangGraph agent: input guard -> reason/act loop -> grounding check -> finalize.

    START
      |
  input_guard --(unsafe / injection)--> END (refusal)
      |
    agent <------------------+----------------+
      |                      |                |
      +--(tool calls)--> tools                |
      |                                       |
  grounding_check --(not grounded, retry)-----+
      |
   finalize --> END

The LLM decides which tools to call; the tools are deterministic and read-only.
The grounding check verifies that every cited source was actually retrieved and
that every number in the answer appears in a tool output. If not, the agent gets
one chance to revise; otherwise the answer is returned with a warning.
"""
from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import Settings, get_settings
from .guardrails import check_grounding, check_input, collect_sources, needs_human_review
from .tools import MaintenanceToolkit, ToolError

SYSTEM_PROMPT = """You are Maintenance Copilot, an assistant for maintenance engineers at an industrial facility.

How to work:
- Use the tools to look up equipment data, anomalies, work orders and procedures. Do not rely on memory.
- For questions about which equipment has problems, or the overall situation, call fleet_overview first.
- For questions about what happened or about abnormal events, call detect_anomalies.
- For questions about the current condition of equipment, call sensor_summary or detect_anomalies, even when
  the question also asks about history.
- For questions about what to do, what should be done, limits or next steps, call search_procedures and base
  the recommendations on the procedures it returns.
- Call several tools when a question needs both data and procedures. Always call at least one tool before answering.
- Every factual statement must cite its source in square brackets, using the exact `source` value
  returned by the tool, e.g. [tool:sensor_summary/P-102/vibration_mm_s/7d] or [doc:vibration_limits#trend-rules].
- Only cite sources that a tool has returned in this conversation. Never cite a tool you have not called,
  and never write tool names or arguments inside the brackets. Square brackets are only for evidence:
  in recommended next steps, name procedures or checks in plain words, without brackets.
- Only state numbers that appear in tool outputs. Never estimate or invent values.
- Summarise the evidence in your own words. Do not copy raw tool output field by field.
- If the data is insufficient, say so.

Safety:
- You are read-only. You never operate equipment and never help bypass or change trips, alarms or interlocks.
- You may recommend next steps, but decisions on shutdown or intrusive work belong to operations and qualified personnel.

Answer format (concise):
**Finding** - one or two sentences.
**Evidence** - bullet points with citations.
**Recommended next steps** - numbered, referencing procedures.
"""

REVIEW_NOTICE = (
    "\n\n> **Human review required:** this answer recommends actions with operational or safety "
    "consequences. Verify with the responsible technician and operations before acting."
)
UNGROUNDED_NOTICE = (
    "\n\n> **Warning:** the automatic grounding check could not verify every statement in this answer "
    "against retrieved data. Treat unverified details with caution."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_outputs: Annotated[list[dict], operator.add]
    tools_called: Annotated[list[str], operator.add]
    steps: int
    revisions: int
    blocked: str | None
    grounding: dict | None
    needs_review: bool
    final_answer: str


# ------------------------------------------------------------------ tools


def build_tools(toolkit: MaintenanceToolkit) -> tuple[list, dict[str, Any]]:
    """LangChain tool schemas for the LLM plus a dispatch table to the toolkit."""

    @tool
    def list_equipment() -> str:
        """List the monitored equipment, available sensors and the data period."""
        return json.dumps(toolkit.list_equipment())

    @tool
    def fleet_overview() -> str:
        """Check all equipment for sustained abnormal behaviour over the last 14 days in one call,
        ranked by severity. Use this first for questions like "which equipment has problems?"."""
        # Fixed screening window: small models tended to pick windows too short to see recent events.
        return json.dumps(toolkit.fleet_overview(last_days=14))

    @tool
    def sensor_summary(equipment_id: str, sensor: str, last_days: int = 7) -> str:
        """Statistics for one sensor over the last N days: latest, mean, min, max, trend per day and
        percent change. For vibration also the severity zone of the latest value.

        Args:
            equipment_id: e.g. "P-102".
            sensor: one of vibration_mm_s, bearing_temp_c, discharge_pressure_bar, flow_m3_h.
            last_days: window length in days (1-28).
        """
        return json.dumps(toolkit.sensor_summary(equipment_id, sensor, last_days))

    @tool
    def detect_anomalies(equipment_id: str, last_days: int = 14) -> str:
        """Detect sustained abnormal behaviour on all sensors of one equipment, compared with the
        baseline before the window. Returns anomaly episodes with start, end, peak value and z-score.

        Args:
            equipment_id: e.g. "C-201".
            last_days: window length in days (1-28).
        """
        return json.dumps(toolkit.detect_anomalies(equipment_id, last_days))

    @tool
    def get_work_orders(equipment_id: str, status: str | None = None) -> str:
        """Maintenance work order history for one equipment.

        Args:
            equipment_id: e.g. "P-101".
            status: optional filter, "open" or "closed".
        """
        return json.dumps(toolkit.get_work_orders(equipment_id, status))

    @tool
    def search_procedures(query: str) -> str:
        """Search maintenance procedures and guidelines (vibration limits, bearing inspection,
        compressor temperature response, heat exchanger fouling, work permits and safety).

        Args:
            query: natural-language search query.
        """
        return json.dumps(toolkit.search_procedures(query))

    tools = [fleet_overview, list_equipment, sensor_summary, detect_anomalies, get_work_orders, search_procedures]
    dispatch = {
        "fleet_overview": toolkit.fleet_overview,
        "list_equipment": toolkit.list_equipment,
        "sensor_summary": toolkit.sensor_summary,
        "detect_anomalies": toolkit.detect_anomalies,
        "get_work_orders": toolkit.get_work_orders,
        "search_procedures": toolkit.search_procedures,
    }
    return tools, dispatch


# ------------------------------------------------------------------ LLM


def make_llm(settings: Settings):
    if settings.llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI

        # reads AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY from the environment
        return AzureChatOpenAI(
            azure_deployment=settings.azure_deployment,
            api_version=settings.azure_api_version,
            temperature=settings.temperature,
        )
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.model, temperature=settings.temperature)
    if settings.llm_provider == "ollama":
        # free, runs locally; needs a tool-calling model, e.g. `ollama pull qwen2.5:7b`
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.model, temperature=settings.temperature, base_url=settings.ollama_base_url)
    if settings.llm_provider == "groq":
        # free tier available; reads GROQ_API_KEY from the environment
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.model, temperature=settings.temperature)
    raise ValueError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}' (use ollama, groq, openai or azure)")


# ------------------------------------------------------------------ graph


def _last_ai(messages: list[AnyMessage]) -> AIMessage | None:
    return next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)


def build_graph(llm=None, toolkit: MaintenanceToolkit | None = None, settings: Settings | None = None, checkpointer=None):
    settings = settings or get_settings()
    toolkit = toolkit or MaintenanceToolkit(settings.data_dir)
    llm = llm or make_llm(settings)
    tools, dispatch = build_tools(toolkit)
    llm_with_tools = llm.bind_tools(tools)

    def input_guard(state: AgentState) -> dict:
        question = state["messages"][-1].content
        check = check_input(question)
        reset = {"steps": 0, "revisions": 0, "grounding": None, "needs_review": False, "final_answer": ""}
        if check.allowed:
            return {**reset, "blocked": None}
        return {**reset, "blocked": check.category, "final_answer": check.message,
                "messages": [AIMessage(content=check.message)]}

    def agent(state: AgentState) -> dict:
        response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def run_tools(state: AgentState) -> dict:
        last = _last_ai(state["messages"])
        messages, outputs, names = [], [], []
        for call in last.tool_calls:
            names.append(call["name"])
            try:
                fn = dispatch[call["name"]]
                result = fn(**call["args"])
                outputs.append(result)
            except (ToolError, TypeError, KeyError) as exc:  # report errors back to the model
                result = {"error": str(exc)}
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"], name=call["name"]))
        return {"messages": messages, "tool_outputs": outputs, "tools_called": names}

    def grounding_check(state: AgentState) -> dict:
        last = _last_ai(state["messages"])
        if last.tool_calls:  # step limit reached while the model still wanted tools
            msg = "I could not complete the analysis within the allowed number of steps."
            return {"final_answer": msg, "grounding": {"passed": False, "reason": "step_limit"}}
        outputs = state.get("tool_outputs", [])
        report = check_grounding(last.content, outputs)
        result = {"grounding": {"passed": report.passed, **report.__dict__}, "final_answer": last.content}
        if not report.passed and state.get("revisions", 0) < settings.max_revisions:
            feedback = report.feedback(valid_sources=collect_sources(outputs))
            result["messages"] = [HumanMessage(content=f"[automatic grounding check] {feedback}")]
            result["revisions"] = state.get("revisions", 0) + 1
        return result

    def finalize(state: AgentState) -> dict:
        answer = state["final_answer"]
        review = needs_human_review(answer)
        if not (state.get("grounding") or {}).get("passed", False):
            answer += UNGROUNDED_NOTICE
        if review:
            answer += REVIEW_NOTICE
        return {"final_answer": answer, "needs_review": review}

    # routing
    def after_guard(state: AgentState) -> str:
        return "end" if state.get("blocked") else "agent"

    def after_agent(state: AgentState) -> str:
        last = _last_ai(state["messages"])
        if last.tool_calls and state["steps"] < settings.max_agent_steps:
            return "tools"
        return "check"

    def after_check(state: AgentState) -> str:
        last = state["messages"][-1]
        revising = isinstance(last, HumanMessage) and last.content.startswith("[automatic grounding check]")
        return "revise" if revising else "finalize"

    g = StateGraph(AgentState)
    g.add_node("input_guard", input_guard)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_node("grounding_check", grounding_check)
    g.add_node("finalize", finalize)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", after_guard, {"agent": "agent", "end": END})
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", "check": "grounding_check"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("grounding_check", after_check, {"revise": "agent", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def ask(graph, question: str, thread_id: str | None = None) -> dict:
    """Run one question through the graph and return a flat result for CLI, API and evals."""
    config: dict = {"recursion_limit": 50}
    n_prev_calls = 0
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
        # tool history accumulates over a conversation; report only this turn's calls
        n_prev_calls = len(graph.get_state(config).values.get("tools_called", []))
    t0 = time.perf_counter()
    state = graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    return {
        "question": question,
        "answer": state.get("final_answer", ""),
        "blocked": state.get("blocked"),
        "tools_called": state.get("tools_called", [])[n_prev_calls:],
        "grounding": state.get("grounding"),
        "needs_review": state.get("needs_review", False),
        "steps": state.get("steps", 0),
        "revisions": state.get("revisions", 0),
        "latency_s": round(time.perf_counter() - t0, 2),
    }
