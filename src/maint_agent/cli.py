"""Command-line interface.

    python -m maint_agent.cli "Which equipment shows abnormal behaviour?"
    python -m maint_agent.cli            # interactive chat with conversation memory
"""
from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv


def _print(result: dict, verbose: bool) -> None:
    print("\n" + result["answer"] + "\n")
    if verbose:
        g = result["grounding"] or {}
        print(f"  tools: {result['tools_called']}")
        print(f"  grounded: {g.get('passed')}  revisions: {result['revisions']}  "
              f"human review: {result['needs_review']}  steps: {result['steps']}  latency: {result['latency_s']} s\n")


def main() -> None:
    load_dotenv()
    from langgraph.checkpoint.memory import MemorySaver

    from .agent import ask, build_graph

    parser = argparse.ArgumentParser(description="Maintenance Copilot")
    parser.add_argument("question", nargs="?", help="question to ask (omit for interactive mode)")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print the answer")
    args = parser.parse_args()

    if args.question:
        _print(ask(build_graph(), args.question), not args.quiet)
        return

    graph = build_graph(checkpointer=MemorySaver())
    thread = str(uuid.uuid4())
    print("Maintenance Copilot - type 'exit' to quit.")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            _print(ask(graph, q, thread_id=thread), not args.quiet)


if __name__ == "__main__":
    main()
