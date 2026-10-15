import pytest

from src.evaluation.retrieval.metrics import aggregate_judged_metrics, judged_retrieval_metrics
from src.evaluation.retrieval.models import JudgmentSet, PoolJudgment


def test_five_of_eight_relevant_chunks_has_perfect_precision_but_partial_recall():
    judgments = {f"c{index}": 1 for index in range(8)}
    groups = {"fact": list(judgments)}
    result = judged_retrieval_metrics(list(judgments)[:5], judgments, groups, [5])
    assert result["precision@5"] == 1
    assert result["pooled_recall@5"] == 5 / 8
    assert result["hit_rate@5"] == result["mrr@5"] == result["evidence_coverage@5"] == 1


def test_unknowns_get_bounds_not_fake_negative_labels():
    result = judged_retrieval_metrics(["unknown", "a", "negative"], {"a": 1, "negative": 0}, {"E": ["a"]}, [5])
    assert result["precision@5"] is None
    assert result["precision_lower_bound@5"] == 0.2
    assert result["precision_upper_bound@5"] == 0.4
    assert result["mrr@5"] is None
    assert result["mrr_lower_bound@5"] == 0.5
    assert result["mrr_upper_bound@5"] == 1
    assert result["hit_rate@5"] == 1
    assert result["judged_fraction@5"] == 2 / 3


def test_no_positive_hit_is_unresolved_until_unknown_candidates_are_judged():
    result = judged_retrieval_metrics(["unknown"], {"a": 1}, {"E": ["a"]}, [5])
    assert result["hit_rate@5"] is None
    assert result["mrr@5"] is None
    assert result["hit_rate_lower_bound@5"] == 0
    assert result["hit_rate_upper_bound@5"] == 1


def test_fact_coverage_does_not_count_duplicate_support_twice():
    result = judged_retrieval_metrics(["a", "b"], {"a": 1, "b": 1, "c": 1}, {"E1": ["a", "b"], "E2": ["c"]}, [5])
    assert result["evidence_coverage@5"] == 0.5
    assert result["complete_evidence_coverage@5"] == 0
    assert result["pooled_recall@5"] == 2 / 3


def test_empty_results_are_zero_with_no_unknown_judgments():
    result = judged_retrieval_metrics([], {"a": 1}, {"E": ["a"]}, [5])
    assert result["precision@5"] == result["hit_rate@5"] == result["mrr@5"] == 0
    assert result["judged_fraction@5"] == 1


def test_macro_average_does_not_drop_unresolved_questions():
    determined = judged_retrieval_metrics(["a"], {"a": 1}, {"E": ["a"]}, [5])
    unknown = judged_retrieval_metrics(["b"], {"a": 1}, {"E": ["a"]}, [5])
    result = aggregate_judged_metrics([determined, unknown])
    assert result["metrics"]["precision@5"] is None
    assert result["determined_question_counts"]["precision@5"] == 1
    assert result["metrics"]["precision_lower_bound@5"] == 0.1


@pytest.mark.parametrize("kwargs", [
    {"review_status": "approved", "relevance": 0, "rationale": "irrelevant"},
    {"relevance": 1, "review_status": "source_checked", "rationale": "relevant"},
    {"relevance": None, "review_status": "approved", "reviewer": "Human"},
    {"relevance": 0, "review_status": "source_checked", "rationale": "wrong", "evidence_ids": ["E"]},
])
def test_invalid_judgments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PoolJudgment(question_id="Q", chunk_id="C", text_sha256="hash", **kwargs)


def test_duplicate_question_chunk_judgments_are_rejected():
    item = PoolJudgment(question_id="Q", chunk_id="C", text_sha256="hash")
    with pytest.raises(ValueError, match="duplicate"):
        JudgmentSet(dataset_sha256="dataset", corpus_sha256="corpus", judgments=[item, item])
