"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("MAINT_DATA_DIR", PROJECT_ROOT / "data")))
    # "ollama" (local, free), "groq" (free tier), "openai" or "azure"
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5:7b"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    azure_deployment: str | None = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    azure_api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0")))
    max_agent_steps: int = field(default_factory=lambda: int(os.getenv("MAX_AGENT_STEPS", "8")))
    max_revisions: int = field(default_factory=lambda: int(os.getenv("MAX_REVISIONS", "1")))


def get_settings() -> Settings:
    return Settings()
