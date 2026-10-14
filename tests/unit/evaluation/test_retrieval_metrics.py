import pytest

from src.evaluation.retrieval.metrics import aggregate_metrics, retrieval_metrics


def test_metrics_use_rank_and_all_labeled_relevant_chunks():
    result = retrieval_metrics(["irrelevant", "a", "irrelevant-2", "b"], {"a", "b"}, [1, 3, 5])
    assert result["hit_rate@1"] == result["recall@1"] == result["mrr@1"] == 0
    assert result["hit_rate@3"] == 1
    assert result["recall@3"] == 0.5
    assert result["mrr@3"] == 0.5
    assert result["recall@5"] == 1
    assert result["mrr@5"] == 0.5


def test_relevant_result_beyond_cutoff_has_zero_reciprocal_rank():
    result = retrieval_metrics(["x", "y", "a"], {"a"}, [1, 2])
    assert all(value == 0 for value in result.values())


def test_empty_results_have_zero_metrics():
    assert all(value == 0 for value in retrieval_metrics([], {"a"}, [5]).values())


def test_macro_average_weights_questions_equally():
    rows = [retrieval_metrics(["a"], {"a"}, [5]), retrieval_metrics(["x", "b"], {"b", "c", "d"}, [5])]
    metrics = aggregate_metrics(rows)
    assert metrics == {"hit_rate@5": 1, "mrr@5": 0.75, "recall@5": 0.666667}


@pytest.mark.parametrize("ranked,relevant,ks", [(["a", "a"], {"a"}, [5]), (["a"], set(), [5]), (["a"], {"a"}, [0]), (["a"], {"a"}, [])])
def test_invalid_metric_inputs_are_rejected(ranked, relevant, ks):
    with pytest.raises(ValueError):
        retrieval_metrics(ranked, relevant, ks)
