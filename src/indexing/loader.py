from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from src.chunking.models import Chunk
from src.indexing.models import IndexRecord


def discover_structure_chunk_files(processed_directory: Path) -> list[Path]:
    """Discover generated structure-aware chunk files deterministically."""

    return sorted(Path(processed_directory).rglob("structure_chunks.json"))


def load_structure_records(
    processed_directory: Path,
) -> tuple[list[IndexRecord], dict[str, int]]:
    """Load and validate every structure-aware chunk in the corpus."""

    files = discover_structure_chunk_files(processed_directory)
    if not files:
        raise FileNotFoundError(
            f"no structure_chunks.json files found under {processed_directory}"
        )
    records: list[IndexRecord] = []
    document_counts: dict[str, int] = {}
    seen_chunk_ids: set[str] = set()
    seen_point_ids: set[str] = set()
    for path in files:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"{path} must contain a JSON array")
        for position, item in enumerate(value):
            try:
                chunk = Chunk.model_validate(item)
                record = IndexRecord.from_chunk(chunk)
            except (ValidationError, ValueError) as error:
                raise ValueError(f"invalid chunk at {path}:{position}: {error}") from error
            if record.chunk_id in seen_chunk_ids:
                raise ValueError(f"duplicate chunk_id found: {record.chunk_id}")
            if record.point_id in seen_point_ids:
                raise ValueError(f"duplicate Qdrant point_id found: {record.point_id}")
            seen_chunk_ids.add(record.chunk_id)
            seen_point_ids.add(record.point_id)
            records.append(record)
            key = record.source_file or record.document_id
            document_counts[key] = document_counts.get(key, 0) + 1
    return records, document_counts
