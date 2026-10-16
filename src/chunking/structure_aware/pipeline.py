from __future__ import annotations

import json
import hashlib
from collections.abc import Collection
from pathlib import Path
from typing import Any

from src.configuration.dataset import get_dataset_configuration
from src.chunking.structure_aware.artifacts import clear_invalidation, invalidate_chunks
from src.chunking.structure_aware.chunker import (
    StructureAwareChunker,
    write_chunks,
)


PROCESSED_DATA_DIR = get_dataset_configuration().processed_root
DEFAULT_ALLOWED_STATUSES = frozenset({"PASS", "REVIEW"})


def discover_documents(processed_directory: Path) -> list[Path]:
    """Find native Docling documents in deterministic path order."""

    return sorted(path for path in Path(processed_directory).rglob("document.json") if not any(part.startswith(".") for part in path.relative_to(processed_directory).parts))


def run_structure_chunking_pipeline(
    processed_directory: Path = PROCESSED_DATA_DIR,
    allowed_statuses: Collection[str] = DEFAULT_ALLOWED_STATUSES,
    chunk_size: int = 500,
    overlap: int = 50,
) -> dict[str, Any]:
    """Chunk validated Docling documents and write a corpus summary.

    Documents with a missing validation report or a status outside the
    allowed set are skipped. Each accepted document receives a
    ``structure_chunks.json`` artifact beside its ingestion outputs.
    """

    processed_directory = Path(processed_directory)
    normalized_statuses = {status.upper() for status in allowed_statuses}
    chunker = StructureAwareChunker(
        chunk_size=chunk_size,
        overlap=overlap,
    )
    summary: dict[str, Any] = {
        "strategy": "structure-aware",
        "chunk_size": chunk_size,
        "overlap": overlap,
        "allowed_validation_statuses": sorted(normalized_statuses),
        "total_documents": 0,
        "processed_documents": 0,
        "skipped_documents": 0,
        "failed_documents": 0,
        "total_chunks": 0,
        "documents": [],
    }

    document_paths = discover_documents(processed_directory)
    summary["total_documents"] = len(document_paths)

    for document_path in document_paths:
        document_directory = document_path.parent
        relative_document = document_path.relative_to(processed_directory)
        validation = _load_json(document_directory / "validation_report.json")
        validation_status = str(validation.get("status", "MISSING")).upper()

        if validation_status not in normalized_statuses:
            invalidate_chunks(document_directory, f"validation-status:{validation_status}")
            summary["skipped_documents"] += 1
            summary["documents"].append(
                {
                    "document": str(relative_document),
                    "status": "SKIPPED",
                    "validation_status": validation_status,
                    "parsed_document_sha256": hashlib.sha256(document_path.read_bytes()).hexdigest(),
                }
            )
            continue

        ingestion_metadata = _load_json(document_directory / "metadata.json")
        document_id = _document_id(
            ingestion_metadata=ingestion_metadata,
            document_path=document_path,
            processed_directory=processed_directory,
        )
        source_metadata = ingestion_metadata.get("source", {})
        if not isinstance(source_metadata, dict):
            source_metadata = {}

        try:
            document_bytes = document_path.read_bytes()
            chunks = chunker.chunk_document(
                document=json.loads(document_bytes),
                document_id=document_id,
                document_metadata={
                    **get_dataset_configuration().source_metadata(str(source_metadata.get("relative_path") or relative_document.as_posix())),
                    "source_file": source_metadata.get("relative_path"),
                    "source_sha256": source_metadata.get("sha256"),
                    "validation_status": validation_status,
                    "parsed_document_sha256": hashlib.sha256(document_bytes).hexdigest(),
                },
            )
            if document_path.read_bytes() != document_bytes:
                raise RuntimeError("parsed document changed during chunking; rerun")
            output_path = document_directory / "structure_chunks.json"
            write_chunks(chunks, output_path)
            clear_invalidation(document_directory)
        except Exception as error:
            invalidate_chunks(document_directory, f"chunking-failed:{type(error).__name__}")
            summary["failed_documents"] += 1
            summary["documents"].append(
                {
                    "document": str(relative_document),
                    "status": "FAILED",
                    "validation_status": validation_status,
                    "error": str(error),
                }
            )
            continue

        summary["processed_documents"] += 1
        summary["total_chunks"] += len(chunks)
        summary["documents"].append(
            {
                "document": str(relative_document),
                "document_id": document_id,
                "status": "COMPLETED",
                "validation_status": validation_status,
                "chunks": len(chunks),
                "output": str(output_path.relative_to(processed_directory)),
            }
        )

    summary_path = processed_directory / "structure_chunking_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as json_file:
        value = json.load(json_file)
    return value if isinstance(value, dict) else {}


def _document_id(
    ingestion_metadata: dict[str, Any],
    document_path: Path,
    processed_directory: Path,
) -> str:
    source = ingestion_metadata.get("source", {})
    if isinstance(source, dict):
        fingerprint = source.get("sha256")
        if isinstance(fingerprint, str) and fingerprint.strip():
            return fingerprint
    return document_path.parent.relative_to(processed_directory).as_posix()


if __name__ == "__main__":
    result = run_structure_chunking_pipeline()
    print(
        "Structure-aware chunking completed: "
        f"{result['processed_documents']} processed, "
        f"{result['skipped_documents']} skipped, "
        f"{result['failed_documents']} failed, "
        f"{result['total_chunks']} chunks"
    )
