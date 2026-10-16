from __future__ import annotations

import json
import hashlib
from pathlib import Path

from pydantic import ValidationError

from src.chunking.models import Chunk
from src.indexing.models import IndexRecord


def discover_structure_chunk_files(processed_directory: Path) -> list[Path]:
    """Discover generated structure-aware chunk files deterministically."""

    return sorted(path for path in Path(processed_directory).rglob("structure_chunks.json") if not any(part.startswith(".") for part in path.relative_to(processed_directory).parts))


def load_structure_records(
    processed_directory: Path,
    *, allow_empty: bool = False,
) -> tuple[list[IndexRecord], dict[str, int]]:
    """Load and validate every structure-aware chunk in the corpus."""

    files = discover_structure_chunk_files(processed_directory)
    if not Path(processed_directory).is_dir():
        raise FileNotFoundError(f"processed corpus directory does not exist: {processed_directory}")
    if not files and not allow_empty:
        raise FileNotFoundError(
            f"no structure_chunks.json files found under {processed_directory}"
        )
    records: list[IndexRecord] = []
    document_counts: dict[str, int] = {}
    seen_chunk_ids: set[str] = set()
    seen_point_ids: set[str] = set()
    for path in files:
        if (path.parent / "chunking_invalidated.json").exists():
            raise ValueError(f"chunk artifacts invalidated for {path.parent}; rerun chunking")
        validation_path = path.parent / "validation_report.json"
        current_validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else None
        if current_validation is not None and current_validation.get("status", "MISSING").upper() not in {"PASS", "REVIEW"}:
            raise ValueError(f"current document validation rejects {path}; rerun chunking after review")
        metadata_path = path.parent / "metadata.json"
        current_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        parsed_path = path.parent / "document.json"
        parsed_hash = hashlib.sha256(parsed_path.read_bytes()).hexdigest() if parsed_path.exists() else None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"{path} must contain a JSON array")
        if not value and not allow_empty:
            raise ValueError(f"{path} contains no chunks; explicit allow_empty is required")
        for position, item in enumerate(value):
            try:
                chunk = Chunk.model_validate(item)
                record = IndexRecord.from_chunk(chunk)
                current_source = current_metadata.get("source", {}).get("sha256")
                if current_source and current_source != record.document_id:
                    raise ValueError("source document changed; regenerate chunks")
                if chunk.metadata.get("parsed_document_sha256") and parsed_hash != chunk.metadata["parsed_document_sha256"]:
                    raise ValueError("parsed document changed; regenerate chunks")
                if current_validation is not None:
                    record.validation_status = current_validation["status"].upper()
                    record.metadata = {**record.metadata, "validation_status": record.validation_status}
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
