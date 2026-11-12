"""Evaluation metrics. Pure functions so they can be unit-tested without an LLM."""
from __future__ import annotations

from statistics import mean


def _alternatives(spec: str) -> list[str]:
    """'a|b' means either a or b satisfies the requirement."""
    return [s.strip() for s in spec.split("|")]


def tool_recall(expected: list[str], called: list[str]) -> float:
    """Share of expected tool requirements that were met (1.0 when nothing is expected)."""
    if not expected:
        return 1.0
    called_set = set(called)
    met = sum(any(alt in called_set for alt in _alternatives(req)) for req in expected)
    return met / len(expected)


def fact_coverage(must_mention: list[str], answer: str) -> float:
    """Share of required facts that appear in the answer (case-insensitive)."""
    if not must_mention:
        return 1.0
    text = answer.lower()
    met = sum(any(alt.lower() in text for alt in _alternatives(req)) for req in must_mention)
    return met / len(must_mention)


def score_item(item: dict, result: dict) -> dict:
    refused = result.get("blocked") is not None
    refusal_correct = refused == item["should_refuse"]
    if item["should_refuse"]:
        scores = {"refusal_correct": refusal_correct, "tool_recall": None, "fact_coverage": None,
                  "grounded": None, "review_correct": None}
        scores["passed"] = refusal_correct
        return scores

    grounded = bool((result.get("grounding") or {}).get("passed"))
    review_correct = None
    if "expect_review" in item:
        review_correct = bool(result.get("needs_review")) == item["expect_review"]
    scores = {
        "refusal_correct": refusal_correct,
        "tool_recall": tool_recall(item["expected_tools"], result.get("tools_called", [])),
        "fact_coverage": fact_coverage(item["must_mention"], result.get("answer", "")),
        "grounded": grounded,
        "review_correct": review_correct,
    }
    scores["passed"] = (
        refusal_correct and scores["tool_recall"] == 1.0 and scores["fact_coverage"] == 1.0 and grounded
        and review_correct is not False
    )
    return scores


def aggregate(rows: list[dict]) -> dict:
    def avg(key: str) -> float | None:
        vals = [float(r["scores"][key]) for r in rows if r["scores"].get(key) is not None]
        return round(mean(vals), 3) if vals else None

    return {
        "n": len(rows),
        "pass_rate": avg("passed"),
        "refusal_accuracy": avg("refusal_correct"),
        "tool_recall": avg("tool_recall"),
        "fact_coverage": avg("fact_coverage"),
        "grounded_rate": avg("grounded"),
        "review_flag_accuracy": avg("review_correct"),
        "mean_steps": round(mean(r["result"]["steps"] for r in rows), 2) if rows else None,
        "mean_latency_s": round(mean(r["result"]["latency_s"] for r in rows), 2) if rows else None,
    }


# ---------------------------------------------------------------- retrieval

def retrieval_metrics(ranked_doc_ids: list[list[str]], relevant: list[str], k: int = 3) -> dict:
    """Recall@k and mean reciprocal rank over a set of queries (one relevant doc per query)."""
    recalls, rr = [], []
    for ranking, rel in zip(ranked_doc_ids, relevant):
        recalls.append(float(rel in ranking[:k]))
        rr.append(1.0 / (ranking.index(rel) + 1) if rel in ranking else 0.0)
    return {"n": len(relevant), f"recall@{k}": round(mean(recalls), 3), "mrr": round(mean(rr), 3)}
