from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

import tiktoken

from src.indexing.embeddings import DenseEmbedder
from src.indexing.models import IndexRecord, point_id_for_chunk


def prepare_embedding_records(
    records: Sequence[IndexRecord], embedder: DenseEmbedder,
) -> tuple[list[IndexRecord], dict[str, Any]]:
    """Preserve source artifacts while splitting inputs that exceed the model limit.

    Character offsets refer to the original chunk, and successive offsets are
    contiguous. Only fallback parts after the first receive a repeated section
    heading. No child can cross the source structure-aware chunk.
    """
    prepared: list[IndexRecord] = []
    fallbacks = []
    encoding = tiktoken.get_encoding("cl100k_base")
    limit = embedder.configuration.max_sequence_length
    for record in records:
        original_count = embedder.count_tokens(record.text)
        if original_count <= limit:
            prepared.append(record.model_copy(update={"embedding_token_count": original_count}))
            continue
        heading = re.match(r"^(Section:[^\n]*\n\n)", record.text)
        prefix = heading.group(0) if heading else (
            "Section: " + " > ".join(record.section_path) + "\n\n" if record.section_path else ""
        )
        if embedder.count_tokens(prefix) >= limit:
            raise ValueError(f"section heading for {record.chunk_id} exceeds the model token budget")
        slices = _bounded_slices(record.text, prefix, embedder)
        for index, (start, end, text) in enumerate(slices):
            chunk_id = f"{record.chunk_id}-embedding-{index:04d}"
            metadata = {
                **record.metadata,
                "embedding_fallback": True,
                "embedding_fallback_reason": "model_token_limit",
                "original_embedding_token_count": original_count,
                "fallback_part_index": index,
                "fallback_part_count": len(slices),
                "source_character_start": start,
                "source_character_end": end,
            }
            prepared.append(record.model_copy(update={
                "point_id": point_id_for_chunk(chunk_id),
                "chunk_id": chunk_id,
                "parent_chunk_id": record.chunk_id,
                "text": text,
                "token_count": len(encoding.encode(text)),
                "embedding_token_count": embedder.count_tokens(text),
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "metadata": metadata,
            }))
        fallbacks.append({
            "chunk_id": record.chunk_id, "source_file": record.source_file,
            "original_tokens": original_count, "parts": len(slices),
        })
    return prepared, {
        "model_token_limit": limit,
        "source_chunks": len(records),
        "oversized_source_chunks": len(fallbacks),
        "fallback_parts": sum(item["parts"] for item in fallbacks),
        "prepared_chunks": len(prepared),
        "maximum_prepared_tokens": max((record.embedding_token_count or 0 for record in prepared), default=0),
        "truncated_inputs": 0,
        "fallbacks": fallbacks,
    }


def _bounded_slices(text: str, prefix: str, embedder: DenseEmbedder) -> list[tuple[int, int, str]]:
    parts = []
    start = 0
    limit = embedder.configuration.max_sequence_length
    while start < len(text):
        context = prefix if start else ""
        if embedder.count_tokens(context + text[start:]) <= limit:
            end = len(text)
        else:
            low, high = start, len(text)
            while low < high:
                middle = (low + high + 1) // 2
                if embedder.count_tokens(context + text[start:middle]) <= limit:
                    low = middle
                else:
                    high = middle - 1
            end = low
            if end <= start:
                raise ValueError("embedding token budget cannot fit even one source character")
            # Prefer paragraphs, then sentence/newline/word boundaries, when
            # they do not leave most of the available space unused.
            floor = start + max(1, (end - start) // 2)
            window = text[floor:end]
            for pattern in (r"\n\s*\n", r"(?<=[.!?])\s+", r"\n", r"\s+"):
                boundaries = list(re.finditer(pattern, window))
                if boundaries:
                    end = floor + boundaries[-1].end()
                    break
        rendered = context + text[start:end]
        # Token counts need not be monotonic in character length. Verify the
        # actual chosen boundary after the binary search and natural split.
        while embedder.count_tokens(rendered) > limit and end > start:
            end -= 1
            rendered = context + text[start:end]
        if end <= start:
            raise ValueError("unable to prepare a lossless embedding fallback")
        parts.append((start, end, rendered))
        start = end
    return parts
