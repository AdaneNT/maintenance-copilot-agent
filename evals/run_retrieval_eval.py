"""Offline retrieval evaluation (no LLM or API key needed).

    python evals/run_retrieval_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from maint_agent.retrieval import Retriever  # noqa: E402
from metrics import retrieval_metrics  # noqa: E402


def unique_docs(hits) -> list[str]:
    seen: list[str] = []
    for h in hits:
        if h.chunk.doc_id not in seen:
            seen.append(h.chunk.doc_id)
    return seen


def main(k: int = 3) -> dict:
    retriever = Retriever.from_dir(ROOT / "data" / "docs")
    items = [json.loads(line) for line in (ROOT / "evals" / "retrieval_eval_set.jsonl").read_text().splitlines() if line.strip()]
    rankings = [unique_docs(retriever.search(it["query"], k=len(retriever.chunks), min_score=0.0)) for it in items]
    result = retrieval_metrics(rankings, [it["relevant_doc"] for it in items], k=k)
    for it, rank in zip(items, rankings):
        mark = "ok " if it["relevant_doc"] in rank[:k] else "MISS"
        print(f"{mark}  {it['query']!r:55} -> {rank[:k]}")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
