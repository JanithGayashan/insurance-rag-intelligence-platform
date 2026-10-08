from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from src.chunking.parent_child.models import (
    HierarchicalChunk,
    ParentChildHierarchy,
)


_TERM = re.compile(r"[A-Za-z0-9]+")


def evaluate_section_retrieval(
    hierarchy: ParentChildHierarchy,
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Run a labeled structural retrieval probe using section titles.

    This is an automatic diagnostic rather than a substitute for business
    questions written and reviewed by insurance-domain experts.
    """

    children = hierarchy.children
    if not children:
        return {"status": "NOT_RUN", "reason": "hierarchy has no children"}

    children_by_section: dict[tuple[str, ...], set[str]] = {}
    for child in children:
        path = tuple(child.section_path)
        if not path or path == ("Document introduction",):
            continue
        children_by_section.setdefault(path, set()).add(child.chunk_id)
    probes = [
        (" > ".join(path), relevant_ids)
        for path, relevant_ids in children_by_section.items()
    ]
    if not probes:
        return {"status": "NOT_RUN", "reason": "no labeled sections found"}

    rankings = _rank_queries(children, [query for query, _ in probes])
    recall_totals = {k: 0.0 for k in k_values}
    reciprocal_ranks: list[float] = []
    for ranking, (_, relevant_ids) in zip(rankings, probes):
        for k in k_values:
            hits = len(set(ranking[:k]) & relevant_ids)
            recall_totals[k] += hits / len(relevant_ids)
        first_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(ranking, start=1)
                if chunk_id in relevant_ids
            ),
            None,
        )
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)

    return {
        "status": "COMPLETED",
        "evaluation_type": "synthetic-section-title-probe",
        "query_count": len(probes),
        "recall_at_k": {
            str(k): round(recall_totals[k] / len(probes), 6)
            for k in k_values
        },
        "mean_reciprocal_rank": round(
            sum(reciprocal_ranks) / len(reciprocal_ranks), 6
        ),
        "limitation": (
            "Section-title probes verify structural retrievability; they do not "
            "replace a reviewed insurance question set."
        ),
    }


def evaluate_question_retrieval(
    hierarchy: ParentChildHierarchy,
    questions: Sequence[Mapping[str, Any]],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Evaluate labeled questions when relevant sections or refs are supplied."""

    labeled: list[tuple[str, set[str]]] = []
    for item in questions:
        if str(item.get("document_id", "")) != hierarchy.document_id:
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        relevant_refs = {
            str(value) for value in item.get("relevant_source_refs", []) if value
        }
        relevant_path = tuple(
            str(value) for value in item.get("relevant_section_path", []) if value
        )
        relevant_ids = {
            child.chunk_id
            for child in hierarchy.children
            if (relevant_refs and relevant_refs.intersection(child.source_refs))
            or (relevant_path and tuple(child.section_path) == relevant_path)
        }
        if relevant_ids:
            labeled.append((question, relevant_ids))

    if not labeled:
        return {
            "status": "NOT_RUN",
            "reason": "no labeled questions matched this document",
        }

    rankings = _rank_queries(
        hierarchy.children, [question for question, _ in labeled]
    )
    recall_totals = {k: 0.0 for k in k_values}
    reciprocal_ranks: list[float] = []
    for ranking, (_, relevant_ids) in zip(rankings, labeled):
        for k in k_values:
            recall_totals[k] += len(set(ranking[:k]) & relevant_ids) / len(
                relevant_ids
            )
        first_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(ranking, start=1)
                if chunk_id in relevant_ids
            ),
            None,
        )
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)

    return {
        "status": "COMPLETED",
        "evaluation_type": "labeled-question-bm25-baseline",
        "query_count": len(labeled),
        "recall_at_k": {
            str(k): round(recall_totals[k] / len(labeled), 6)
            for k in k_values
        },
        "mean_reciprocal_rank": round(
            sum(reciprocal_ranks) / len(reciprocal_ranks), 6
        ),
    }


def unavailable_answer_evaluation() -> dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "answer_accuracy": None,
        "source_faithfulness": None,
        "reason": (
            "Answer evaluation requires a RAG answer generator plus reviewed "
            "reference answers and supporting-source labels."
        ),
    }


def _rank_queries(
    chunks: Sequence[HierarchicalChunk],
    queries: Sequence[str],
) -> list[list[str]]:
    documents = [_terms(chunk.text) for chunk in chunks]
    document_frequencies = Counter(
        term for document in documents for term in set(document)
    )
    document_count = len(documents)
    average_length = (
        sum(len(document) for document in documents) / document_count
        if document_count
        else 1.0
    )
    rankings: list[list[str]] = []
    for query in queries:
        query_terms = set(_terms(query))
        scored: list[tuple[float, int, str]] = []
        for index, (chunk, document) in enumerate(zip(chunks, documents)):
            frequencies = Counter(document)
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                document_frequency = document_frequencies[term]
                inverse_frequency = math.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                length_factor = 1.5 * (
                    1 - 0.75 + 0.75 * len(document) / average_length
                )
                score += inverse_frequency * (
                    frequency * 2.5 / (frequency + length_factor)
                )
            scored.append((score, -index, chunk.chunk_id))
        scored.sort(reverse=True)
        rankings.append([chunk_id for _, _, chunk_id in scored])
    return rankings


def _terms(text: str) -> list[str]:
    return [term.casefold() for term in _TERM.findall(text)]
