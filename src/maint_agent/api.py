"""Minimal HTTP service so the agent can be deployed as a container.

    uvicorn maint_agent.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import uuid
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from . import __version__
from .agent import ask, build_graph

load_dotenv()
app = FastAPI(title="Maintenance Copilot", version=__version__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    thread_id: str | None = Field(default=None, description="reuse to continue a conversation")


class AskResponse(BaseModel):
    answer: str
    blocked: str | None
    tools_called: list[str]
    grounded: bool | None
    needs_review: bool
    steps: int
    latency_s: float


@lru_cache
def _graph():
    # In-memory conversation state; use a persistent checkpointer (e.g. Postgres) in production.
    return build_graph(checkpointer=MemorySaver())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    r = ask(_graph(), req.question, thread_id=req.thread_id or str(uuid.uuid4()))
    return AskResponse(
        answer=r["answer"],
        blocked=r["blocked"],
        tools_called=r["tools_called"],
        grounded=(r["grounding"] or {}).get("passed"),
        needs_review=r["needs_review"],
        steps=r["steps"],
        latency_s=r["latency_s"],
    )
