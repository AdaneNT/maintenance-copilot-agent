"""End-to-end agent evaluation (requires an LLM API key, see .env.example).

    python evals/run_agent_eval.py            # writes evals/results/<timestamp>.{json,md}
    python evals/run_agent_eval.py --repeats 3

Metrics per question: refusal correctness, tool recall, fact coverage, grounding,
human-review flag correctness, steps and latency. Repeats estimate run-to-run variance.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from maint_agent.agent import ask, build_graph  # noqa: E402
from maint_agent.config import get_settings  # noqa: E402
from metrics import aggregate, score_item  # noqa: E402


def to_markdown(summary: dict, rows: list[dict], model: str) -> str:
    lines = [f"# Agent evaluation - {model}", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary.items()]
    lines += ["", "| id | category | passed | tool recall | fact coverage | grounded | steps |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["scores"]
        lines.append(f"| {r['id']} | {r['category']} | {s['passed']} | {s['tool_recall']} | "
                     f"{s['fact_coverage']} | {s['grounded']} | {r['result']['steps']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", help="comma-separated item ids")
    args = parser.parse_args()

    items = [json.loads(line) for line in (ROOT / "evals" / "agent_eval_set.jsonl").read_text().splitlines() if line.strip()]
    if args.only:
        wanted = set(args.only.split(","))
        items = [it for it in items if it["id"] in wanted]

    settings = get_settings()
    graph = build_graph(settings=settings)
    rows = []
    for rep in range(args.repeats):
        for it in items:
            result = ask(graph, it["question"])
            scores = score_item(it, result)
            rows.append({"id": it["id"], "category": it["category"], "repeat": rep, "result": result, "scores": scores})
            print(f"[{'PASS' if scores['passed'] else 'FAIL'}] {it['id']:9} tools={result['tools_called']}")

    summary = aggregate(rows)
    model = settings.azure_deployment if settings.llm_provider == "azure" else settings.model
    print(json.dumps(summary, indent=2))

    out_dir = ROOT / "evals" / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"{stamp}.json").write_text(json.dumps({"model": model, "summary": summary, "rows": rows}, indent=2))
    (out_dir / f"{stamp}.md").write_text(to_markdown(summary, rows, model))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
