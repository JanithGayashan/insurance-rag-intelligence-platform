import json

import pytest

from src.evaluation.retrieval.comparison import evaluate_retrieval_comparison
from src.evaluation.retrieval.models import JudgmentSet
from src.evaluation.retrieval.dataset import bind_evidence, corpus_fingerprint, load_dataset
from src.evaluation.retrieval.pooling import build_pool, merge_judgments
from src.indexing.configuration import json_sha256
from tests.integration.evaluation.test_dense_evaluation import fixture, make_dataset


def test_comparison_builds_deduplicated_pool_and_leaves_unknowns_unknown(fixture):
    root, retriever = fixture
    output = root / "pooled"
    summary = evaluate_retrieval_comparison(make_dataset(root), retriever, output)
    assert set(summary["by_method"]) == {"dense", "bm25", "hybrid"}
    assert summary["pool"]["question_chunk_pairs"] == 3
    assert summary["pool"]["accepted_relevant_pairs"] == 1
    assert summary["pool"]["accepted_irrelevant_pairs"] == 0
    assert summary["pool"]["unjudged_top5_pairs"] == 2
    values = summary["by_method"]["dense"]["unfiltered"]["metrics"]
    assert values["precision@5"] is None
    assert values["precision_lower_bound@5"] == 0.2
    assert values["precision_upper_bound@5"] == 0.6
    assert values["evidence_coverage@5"] == 1
    assert values["pooled_recall@5"] == 1
    assert (output / "pool_review.md").exists()
    pool = json.loads((output / "pool.json").read_text(encoding="utf-8"))
    assert len({(row["question_id"], row["chunk_id"]) for row in pool["candidates"]}) == 3
    assert all(row["origins"] for row in pool["candidates"])


def test_reviewed_negatives_enable_exact_precision_and_strict_gate(fixture):
    root, retriever = fixture
    path, output = make_dataset(root), root / "pooled"
    evaluate_retrieval_comparison(path, retriever, output)
    judgments = JudgmentSet.model_validate_json((output / "judgments.json").read_text(encoding="utf-8"))
    for row in judgments.judgments:
        if row.relevance is None:
            row.relevance, row.review_status, row.reviewer = 0, "approved", "Synthetic fixture reviewer"
            row.rationale = "This fixture passage does not answer the specified policy question."
    path_judgments = root / "reviewed.json"
    path_judgments.write_text(judgments.model_dump_json(), encoding="utf-8")
    summary = evaluate_retrieval_comparison(path, retriever, output, judgments_path=path_judgments, require_judged_top5=True)
    assert summary["status"] == "REVIEWED_TOP5_POOLED_BASELINE"
    assert summary["pool"]["unjudged_top5_pairs"] == 0
    assert summary["by_method"]["dense"]["unfiltered"]["metrics"]["precision@5"] == 0.2
    assert summary["by_method"]["dense"]["unfiltered"]["metrics"]["judged_fraction@5"] == 1
    # A later run without --judgments must not overwrite completed review.
    resumed = evaluate_retrieval_comparison(path, retriever, output, require_judged_top5=True)
    assert resumed["pool"]["accepted_irrelevant_pairs"] == 2


def test_strict_precision_refuses_unjudged_pairs_without_publication(fixture):
    root, retriever = fixture
    with pytest.raises(ValueError, match="unjudged pairs"):
        evaluate_retrieval_comparison(make_dataset(root), retriever, root / "strict", require_judged_top5=True)
    assert not (root / "strict").exists()


def test_provisional_question_gate_runs_before_retrieval(fixture):
    root, retriever = fixture
    path = make_dataset(root, status="source_checked")
    with pytest.raises(ValueError, match="no approved"):
        evaluate_retrieval_comparison(path, retriever, root / "pooled")
    result = evaluate_retrieval_comparison(path, retriever, root / "pooled", include_source_checked=True)
    assert result["status"] == "PROVISIONAL_BASELINE"


def test_stale_judgments_and_contradictions_are_rejected(fixture):
    root, retriever = fixture
    path, output = make_dataset(root), root / "pooled"
    evaluate_retrieval_comparison(path, retriever, output)
    original = json.loads((output / "judgments.json").read_text(encoding="utf-8"))
    stale = {**original, "corpus_sha256": "different"}
    judgment_path = root / "bad.json"
    judgment_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="different dataset or corpus"):
        evaluate_retrieval_comparison(path, retriever, output, judgments_path=judgment_path)
    positive = next(row for row in original["judgments"] if row["relevance"] == 1)
    positive.update(relevance=0, evidence_ids=[], supporting_quotes=[], rationale="Contradiction")
    judgment_path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="contradicts source evidence"):
        evaluate_retrieval_comparison(path, retriever, output, judgments_path=judgment_path)


@pytest.mark.parametrize("kwargs", [{"pool_depth": 4}, {"candidate_k": 5}, {"methods": ("bad",)}, {"modes": ("bad",)}, {"outside_pool_sample": -1}])
def test_invalid_comparison_configuration_is_rejected(fixture, kwargs):
    root, retriever = fixture
    with pytest.raises(ValueError):
        evaluate_retrieval_comparison(make_dataset(root), retriever, root / "pooled", **kwargs)


def test_index_change_during_comparison_aborts_publication(fixture, monkeypatch):
    from tests.unit.indexing.test_qdrant_index import _record
    root, retriever = fixture
    original_search = retriever.vector_store.search
    def mutate(*args, **kwargs):
        results = original_search(*args, **kwargs)
        record = _record(9, "New flood policy text")
        retriever.vector_store.upsert([record], retriever.embedder.embed_documents([record.text]))
        return results
    monkeypatch.setattr(retriever.vector_store, "search", mutate)
    with pytest.raises(RuntimeError, match="index changed"):
        evaluate_retrieval_comparison(make_dataset(root), retriever, root / "pooled")
    assert not (root / "pooled").exists()


@pytest.mark.parametrize("change,error", [
    ({"text_sha256": "changed"}, "chunk text changed"),
    ({"supporting_quotes": ["A quotation absent from the chunk"]}, "quote is absent"),
    ({"evidence_ids": ["unknown"]}, "unknown evidence"),
])
def test_review_import_validates_text_and_evidence(fixture, change, error):
    root, retriever = fixture
    path = make_dataset(root)
    dataset = load_dataset(path)
    snapshot = retriever.vector_store.snapshot()
    _, seed = build_pool(dataset.questions, snapshot, [], bind_evidence(dataset.questions, snapshot),
                         json_sha256(dataset.model_dump()), corpus_fingerprint(snapshot))
    value = seed.model_dump()
    value["judgments"][0].update(change)
    review = root / "review.json"
    review.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        merge_judgments(seed, review, dataset.questions, snapshot)


def test_review_cannot_label_same_text_from_wrong_policy_as_relevant(fixture):
    root, retriever = fixture
    path, output = make_dataset(root), root / "pooled"
    evaluate_retrieval_comparison(path, retriever, output)
    value = json.loads((output / "judgments.json").read_text(encoding="utf-8"))
    wrong = next(row for row in value["judgments"] if row["chunk_id"].startswith("doc-b"))
    wrong.update(relevance=1, review_status="approved", reviewer="Synthetic test reviewer",
                 rationale="Wrong policy despite matching text", supporting_quotes=["Flood damage is covered."], evidence_ids=["E1"])
    review = root / "wrong_policy.json"
    review.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong policy"):
        evaluate_retrieval_comparison(path, retriever, output, judgments_path=review)


def test_cutoff_five_is_always_reported_and_audit_sampling_is_reproducible(fixture):
    root, retriever = fixture
    path, output = make_dataset(root), root / "pooled"
    for index in range(3, 12):
        from tests.unit.indexing.test_qdrant_index import _record
        record = _record(index, f"Unrelated hospital invoice number {index}")
        retriever.vector_store.upsert([record], retriever.embedder.embed_documents([record.text]))
    summary = evaluate_retrieval_comparison(path, retriever, output, methods=("bm25",), pool_depth=5, candidate_k=5, k_values=[1])
    first = json.loads((output / "pool.json").read_text(encoding="utf-8"))
    assert summary["k_values"] == [1, 5]
    audit_ids = {row["chunk_id"] for row in first["candidates"] if row["origins"][0]["mode"] == "audit"}
    assert len(audit_ids) == 2
    evaluate_retrieval_comparison(path, retriever, output, methods=("bm25",), pool_depth=5, candidate_k=5, k_values=[1])
    assert first == json.loads((output / "pool.json").read_text(encoding="utf-8"))
