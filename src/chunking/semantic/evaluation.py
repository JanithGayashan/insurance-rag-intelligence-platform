from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from statistics import mean, median
from typing import Any

import numpy as np

from src.chunking.models import Chunk
from src.chunking.semantic.embeddings import TextEmbedder, cosine_similarities


_WORD = re.compile(r"\w+", re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]*[A-Z0-9])")


def evaluate_semantic_chunks(
    chunks: Sequence[Chunk],
    source_text: str,
    embedder: TextEmbedder,
    max_chunk_size: int,
    tiny_chunk_threshold: int = 20,
) -> dict[str, Any]:
    """Measure structural, textual, size, and embedding quality."""

    token_counts = [chunk.token_count or 0 for chunk in chunks]
    reconstructed = " ".join(_chunk_body(chunk.text) for chunk in chunks)
    source_words = Counter(_words(source_text))
    output_words = Counter(_words(reconstructed))
    missing_words = sum((source_words - output_words).values())
    duplicated_words = sum((output_words - source_words).values())
    source_word_count = sum(source_words.values())

    semantic = _semantic_separation(chunks, embedder)
    retrieval = evaluate_section_retrieval(chunks, embedder)
    return {
        "chunk_count": len(chunks),
        "token_statistics": {
            "total": sum(token_counts),
            "average": round(mean(token_counts), 2) if token_counts else 0,
            "median": round(median(token_counts), 2) if token_counts else 0,
            "minimum": min(token_counts, default=0),
            "maximum": max(token_counts, default=0),
        },
        "oversized_chunks": sum(count > max_chunk_size for count in token_counts),
        "tiny_chunks": sum(count < tiny_chunk_threshold for count in token_counts),
        "children_crossing_sections": sum(
            not bool(chunk.metadata.get("section_boundary_enforced")) for chunk in chunks
        ),
        "fallback_token_split_chunks": sum(
            bool(chunk.metadata.get("fallback_token_split")) for chunk in chunks
        ),
        "missing_word_rate": round(missing_words / source_word_count, 6)
        if source_word_count
        else 0,
        "duplicate_word_rate": round(duplicated_words / source_word_count, 6)
        if source_word_count
        else 0,
        "sentence_boundary_break_rate": round(
            sum(bool(chunk.metadata.get("fallback_token_split")) for chunk in chunks)
            / len(chunks),
            6,
        )
        if chunks
        else 0,
        **semantic,
        "section_retrieval_evaluation": retrieval,
        "answer_evaluation": {
            "status": "NOT_RUN",
            "answer_accuracy": None,
            "source_faithfulness": None,
            "reason": (
                "Requires the later answer-generation pipeline and a reviewed "
                "insurance question-and-answer set."
            ),
        },
    }


def evaluate_section_retrieval(
    chunks: Sequence[Chunk],
    embedder: TextEmbedder,
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    sections: dict[tuple[str, ...], set[str]] = {}
    for chunk in chunks:
        path = tuple(chunk.section_path)
        if path:
            sections.setdefault(path, set()).add(chunk.chunk_id)
    if not sections or not chunks:
        return {"status": "NOT_RUN", "reason": "no labeled sections found"}

    queries = [" > ".join(path) for path in sections]
    vectors = embedder.encode([*queries, *(chunk.text for chunk in chunks)])
    query_vectors = vectors[: len(queries)]
    chunk_vectors = vectors[len(queries) :]
    scores = query_vectors @ chunk_vectors.T
    recall_totals = {k: 0.0 for k in k_values}
    reciprocal_ranks: list[float] = []
    for row, relevant_ids in zip(scores, sections.values()):
        order = np.argsort(-row)
        ranking = [chunks[index].chunk_id for index in order]
        for k in k_values:
            recall_totals[k] += len(set(ranking[:k]) & relevant_ids) / len(relevant_ids)
        first = next(
            (rank for rank, chunk_id in enumerate(ranking, 1) if chunk_id in relevant_ids),
            None,
        )
        reciprocal_ranks.append(1 / first if first else 0.0)
    return {
        "status": "COMPLETED",
        "evaluation_type": "embedding-section-title-probe",
        "query_count": len(queries),
        "recall_at_k": {
            str(k): round(recall_totals[k] / len(queries), 6) for k in k_values
        },
        "mean_reciprocal_rank": round(mean(reciprocal_ranks), 6),
        "limitation": (
            "Section-title probes are automatic diagnostics and do not replace "
            "reviewed insurance questions."
        ),
    }


def _semantic_separation(
    chunks: Sequence[Chunk], embedder: TextEmbedder
) -> dict[str, float | None]:
    sentence_groups: list[list[str]] = []
    flat_sentences: list[str] = []
    for chunk in chunks:
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(_chunk_body(chunk.text))
            if sentence.strip()
        ]
        sentence_groups.append(sentences)
        flat_sentences.extend(sentences)
    sentence_vectors = embedder.encode(flat_sentences)
    cursor = 0
    within: list[float] = []
    starts: list[np.ndarray | None] = []
    ends: list[np.ndarray | None] = []
    for sentences in sentence_groups:
        group_vectors = sentence_vectors[cursor : cursor + len(sentences)]
        cursor += len(sentences)
        if len(group_vectors):
            starts.append(group_vectors[0])
            ends.append(group_vectors[-1])
        else:
            starts.append(None)
            ends.append(None)
        if len(group_vectors) > 1:
            within.extend(
                float(value)
                for value in cosine_similarities(group_vectors[:-1], group_vectors[1:])
            )

    boundary: list[float] = []
    for index in range(len(chunks) - 1):
        if tuple(chunks[index].section_path) != tuple(chunks[index + 1].section_path):
            continue
        if ends[index] is None or starts[index + 1] is None:
            continue
        boundary.append(
            float(
                cosine_similarities(
                    np.asarray([ends[index]]), np.asarray([starts[index + 1]])
                )[0]
            )
        )
    within_mean = mean(within) if within else None
    boundary_mean = mean(boundary) if boundary else None
    return {
        "mean_within_chunk_adjacent_similarity": round(within_mean, 6)
        if within_mean is not None
        else None,
        "mean_across_boundary_similarity": round(boundary_mean, 6)
        if boundary_mean is not None
        else None,
        "semantic_separation_margin": round(within_mean - boundary_mean, 6)
        if within_mean is not None and boundary_mean is not None
        else None,
    }


def _chunk_body(text: str) -> str:
    if text.startswith("Section: ") and "\n\n" in text:
        return text.split("\n\n", 1)[1]
    return text


def _words(text: str) -> list[str]:
    return [word.casefold() for word in _WORD.findall(text)]
