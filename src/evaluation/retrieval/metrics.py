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
