from metrics import aggregate, fact_coverage, retrieval_metrics, score_item, tool_recall


def test_tool_recall_with_alternatives():
    assert tool_recall(["sensor_summary|detect_anomalies", "search_procedures"], ["detect_anomalies"]) == 0.5
    assert tool_recall([], ["anything"]) == 1.0


def test_fact_coverage_case_insensitive():
    assert fact_coverage(["zone d", "7.47"], "Latest value 7.47 mm/s (Zone D)") == 1.0
    assert fact_coverage(["WO-1003|WO-1002"], "see WO-1002") == 1.0
    assert fact_coverage(["85", "95"], "alarm at 85") == 0.5


def _result(**kw):
    base = {"answer": "", "blocked": None, "tools_called": [], "grounding": {"passed": True},
            "needs_review": False, "steps": 2, "latency_s": 1.0}
    return {**base, **kw}


def test_score_refusal_items():
    item = {"should_refuse": True, "expected_tools": [], "must_mention": []}
    assert score_item(item, _result(blocked="unsafe_operation"))["passed"]
    assert not score_item(item, _result())["passed"]


def test_score_answer_item():
    item = {"should_refuse": False, "expected_tools": ["get_work_orders"], "must_mention": ["WO-1005"], "expect_review": False}
    good = score_item(item, _result(answer="Open: WO-1005", tools_called=["get_work_orders"]))
    assert good["passed"]
    ungrounded = score_item(item, _result(answer="Open: WO-1005", tools_called=["get_work_orders"], grounding={"passed": False}))
    assert not ungrounded["passed"]


def test_aggregate_and_retrieval_metrics():
    rows = [{"result": _result(), "scores": {"passed": True, "refusal_correct": True}},
            {"result": _result(steps=4), "scores": {"passed": False, "refusal_correct": True}}]
    agg = aggregate(rows)
    assert agg["pass_rate"] == 0.5 and agg["mean_steps"] == 3
    m = retrieval_metrics([["a", "b"], ["b", "a"]], ["a", "a"], k=1)
    assert m["recall@1"] == 0.5 and m["mrr"] == 0.75
