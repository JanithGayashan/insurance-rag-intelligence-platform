from __future__ import annotations

from collections.abc import Iterable, Sequence


def retrieval_metrics(
    ranked_chunk_ids: Sequence[str], relevant_chunk_ids: Iterable[str], k_values: Sequence[int],
) -> dict[str, float]:
    """Binary chunk relevance; reciprocal rank is explicitly truncated at K."""
    relevant = set(relevant_chunk_ids)
    if not relevant:
        raise ValueError("answerable questions require nonempty relevance labels")
    if len(set(ranked_chunk_ids)) != len(ranked_chunk_ids):
        raise ValueError("retrieved chunk IDs must be unique")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("K values must be positive")
    result = {}
    for k in sorted(set(k_values)):
        top = ranked_chunk_ids[:k]
        ranks = [rank for rank, chunk_id in enumerate(top, 1) if chunk_id in relevant]
        result[f"hit_rate@{k}"] = float(bool(ranks))
        result[f"recall@{k}"] = len(ranks) / len(relevant)
        result[f"mrr@{k}"] = 1 / ranks[0] if ranks else 0.0
    return result


def aggregate_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average an empty evaluation")
    names = set(rows[0])
    if any(set(row) != names for row in rows):
        raise ValueError("metric rows must use the same K values")
    return {name: round(sum(row[name] for row in rows) / len(rows), 6) for name in sorted(names)}


def judged_retrieval_metrics(ranked_chunk_ids, judgments, evidence_groups, k_values):
    """Exact top-K metrics only when determined; otherwise expose honest bounds.

    `judgments` contains accepted 0/1 judgments only. Missing keys are unknown.
    Pooled recall is explicitly relative to the currently accepted positive set.
    """
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("K values must be positive")
    if len(set(ranked_chunk_ids)) != len(ranked_chunk_ids):
        raise ValueError("retrieved chunk IDs must be unique")
    if any(value not in (0, 1) for value in judgments.values()):
        raise ValueError("accepted judgments must be binary")
    if not evidence_groups or any(not ids for ids in evidence_groups.values()):
        raise ValueError("every evidence unit must have supporting chunks")
    positives = {chunk for chunk, value in judgments.items() if value == 1}
    result = {}
    for k in sorted(set(k_values)):
        top = ranked_chunk_ids[:k]
        positive_ranks = [rank for rank, chunk in enumerate(top, 1) if judgments.get(chunk) == 1]
        possible_ranks = [rank for rank, chunk in enumerate(top, 1) if judgments.get(chunk) != 0]
        unknown = sum(chunk not in judgments for chunk in top)
        p_lower, p_upper = len(positive_ranks) / k, (len(positive_ranks) + unknown) / k
        hit_lower, hit_upper = float(bool(positive_ranks)), float(bool(possible_ranks))
        rr_lower = 1 / positive_ranks[0] if positive_ranks else 0.0
        rr_upper = 1 / possible_ranks[0] if possible_ranks else 0.0
        for metric, lower, upper in (("precision", p_lower, p_upper),
                                     ("hit_rate", hit_lower, hit_upper), ("mrr", rr_lower, rr_upper)):
            result[f"{metric}@{k}"] = lower if lower == upper else None
            result[f"{metric}_lower_bound@{k}"] = lower
            result[f"{metric}_upper_bound@{k}"] = upper
        result[f"judged_fraction@{k}"] = (len(top) - unknown) / len(top) if top else 1.0
        result[f"pooled_recall@{k}"] = len(positive_ranks) / len(positives) if positives else None
        covered = sum(bool(set(top) & set(ids) & positives) for ids in evidence_groups.values())
        result[f"evidence_coverage@{k}"] = covered / len(evidence_groups)
        result[f"complete_evidence_coverage@{k}"] = float(covered == len(evidence_groups))
    return result


def aggregate_judged_metrics(rows):
    """Never average only the easier, fully judged subset without saying so."""
    if not rows or any(set(row) != set(rows[0]) for row in rows):
        raise ValueError("nonempty rows with consistent metric keys are required")
    counts = {name: sum(row[name] is not None for row in rows) for name in rows[0]}
    metrics = {name: round(sum(row[name] for row in rows) / len(rows), 6)
               if counts[name] == len(rows) else None for name in sorted(rows[0])}
    return {"metrics": metrics, "determined_question_counts": counts, "questions": len(rows)}
