import json

import pytest

from src.evaluation.retrieval.comparison import evaluate_retrieval_comparison
from src.retrieval.hybrid import LlamaIndexRetrievalSuite
from tests.integration.evaluation.test_dense_evaluation import fixture, make_dataset
from tests.unit.retrieval.test_reranking import make_reranker


def test_hybrid_and_reranked_share_candidates_and_embedding_work(fixture, monkeypatch):
    _, dense = fixture
    reranker, predictor = make_reranker()
    calls = []
    original = dense.embedder.embed_query
    def encode(question):
        calls.append(question)
        return original(question)
    monkeypatch.setattr(dense.embedder, "embed_query", encode)
    suite = LlamaIndexRetrievalSuite(dense, reranker=reranker)
    results, diagnostics = suite.retrieve_all_with_diagnostics("flood", methods=("hybrid", "hybrid_reranked"))
    assert len(calls) == 1
    assert len(predictor.pairs) == 3
    assert {hit.chunk_id for hit in results["hybrid"]} == {hit.chunk_id for hit in results["hybrid_reranked"]}
    assert set(diagnostics["rerank_candidate_ids"]) == {hit.chunk_id for hit in results["hybrid"]}
    assert diagnostics["method_seconds"]["hybrid_reranked"] >= diagnostics["method_seconds"]["hybrid"]
    assert diagnostics["rerank_window_count"] == 3
    scoped = suite.retrieve("flood", method="hybrid_reranked", filters={"document_id": "doc-b"})
    assert len(scoped) == 1 and scoped[0].document_id == "doc-b"
    assert suite.retrieve("flood", method="hybrid_reranked", filters={"document_id": "missing"}) == []


def test_paired_evaluation_reports_quality_timing_and_review_status(fixture):
    root, retriever = fixture
    reranker, _ = make_reranker()
    output = root / "reranked"
    progress = []
    summary = evaluate_retrieval_comparison(
        make_dataset(root), retriever, output, methods=("hybrid", "hybrid_reranked"), reranker=reranker,
        progress_callback=progress.append, k_values=[1, 5, 20],
    )
    assert len(progress) == 2 and progress[-1]["completed"] == progress[-1]["total"] == 2
    assert summary["reranker"]["fingerprint"] == reranker.fingerprint
    for mode in ["unfiltered", "policy_scoped"]:
        paired = summary["paired_hybrid_comparison"][mode]
        assert paired["improved_questions"] + paired["regressed_questions"] + paired["tied_questions"] == 1
        assert paired["mean_rerank_seconds"] > 0
        assert paired["questions_missing_candidate_evidence"] == []
        assert paired["mean_candidate_evidence_coverage"] == 1
        for method in ("hybrid", "hybrid_reranked"):
            assert summary["by_method"][method][mode]["metrics"]["evidence_coverage@5"] == 1
        assert (summary["by_method"]["hybrid"][mode]["metrics"]["pooled_recall@20"]
                == summary["by_method"]["hybrid_reranked"][mode]["metrics"]["pooled_recall@20"])
    assert summary["by_method"]["hybrid_reranked"]["unfiltered"]["metrics"]["precision@5"] is None
    report = json.loads((output / "retrieval_comparison.json").read_text(encoding="utf-8"))
    assert len(report["queries"]) == 4
    row = next(row for row in report["queries"] if row["method"] == "hybrid_reranked")
    assert row["rerank_candidate_ids"]
    assert row["results"][0]["metadata"]["reranking"]["truncated"] is False
    assert "Paired hybrid comparison" in (output / "retrieval_comparison.md").read_text(encoding="utf-8")
    assert "Flood damage is covered." in (output / "ranking_changes.md").read_text(encoding="utf-8")


def test_reranker_cannot_promote_candidates_outside_the_fixed_budget(fixture):
    _, dense = fixture
    reranker, predictor = make_reranker()
    suite = LlamaIndexRetrievalSuite(dense, reranker=reranker, rerank_candidates=2)
    results, diagnostics = suite.retrieve_all_with_diagnostics("flood", top_k=2, methods=("hybrid", "hybrid_reranked"))
    assert len(predictor.pairs) == 2
    assert set(diagnostics["rerank_candidate_ids"]) == {hit.chunk_id for hit in results["hybrid_reranked"]}
    assert len(results["hybrid"]) == 2


def test_reranking_requires_explicit_configuration_and_fair_depth(fixture):
    root, dense = fixture
    suite = LlamaIndexRetrievalSuite(dense)
    with pytest.raises(ValueError, match="configured reranker"):
        suite.retrieve("flood", method="hybrid_reranked")
    reranker, _ = make_reranker()
    with pytest.raises(ValueError, match="pool_depth == rerank_candidates"):
        evaluate_retrieval_comparison(make_dataset(root), dense, root / "out",
                                     methods=("hybrid", "hybrid_reranked"), reranker=reranker, rerank_candidates=10)
    with pytest.raises(ValueError):
        LlamaIndexRetrievalSuite(dense, reranker=reranker, rerank_candidates=2).retrieve("flood", method="hybrid_reranked", top_k=5)
