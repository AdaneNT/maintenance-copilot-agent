from run_retrieval_eval import main as run_retrieval_eval

from maint_agent.retrieval import chunk_markdown


def test_chunking_by_section():
    chunks = chunk_markdown("d", "# Title\n\nintro\n\n## A\ntext a\n\n## B\ntext b\n")
    assert [c.section for c in chunks] == ["A", "B"]
    assert chunks[0].source == "doc:d#a"


def test_search_returns_cited_sections(toolkit):
    out = toolkit.search_procedures("compressor bearing temperature alarm")
    assert out["results"][0]["source"].startswith("doc:compressor_high_temperature#")


def test_retrieval_quality_regression():
    """Guards against retrieval regressions when documents or features change."""
    result = run_retrieval_eval(k=3)
    assert result["recall@3"] >= 0.9
    assert result["mrr"] >= 0.8
