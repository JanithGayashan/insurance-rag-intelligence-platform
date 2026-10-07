from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.chunking.models import Chunk
from src.chunking.structure_aware.chunker import (
    StructureAwareChunker,
    write_chunks,
)
from src.chunking.structure_aware.docling_hybrid import (
    DoclingHybridStructureChunker,
)
from src.chunking.structure_aware.pipeline import discover_documents


PROCESSED_DATA_DIR = Path("data/processed/allianz")
DEFAULT_ALLOWED_STATUSES = frozenset({"PASS", "REVIEW"})


def compare_document(
    document_path: Path,
    output_directory: Path | None = None,
    chunk_size: int = 500,
    overlap: int = 50,
    document_id: str | None = None,
    document_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run both structure-aware chunkers on exactly the same document."""

    document_path = Path(document_path)
    output_directory = output_directory or (
        document_path.parent / "structure_aware_comparison"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    resolved_document_id = document_id or document_path.parent.name

    custom_chunks = StructureAwareChunker(
        chunk_size=chunk_size,
        overlap=overlap,
    ).chunk_file(
        document_path,
        document_id=resolved_document_id,
        document_metadata=document_metadata,
    )
    hybrid_chunks = DoclingHybridStructureChunker(
        chunk_size=chunk_size,
    ).chunk_file(
        document_path,
        document_id=resolved_document_id,
        document_metadata=document_metadata,
    )

    custom_output = output_directory / "custom_chunks.json"
    hybrid_output = output_directory / "docling_hybrid_chunks.json"
    write_chunks(custom_chunks, custom_output)
    write_chunks(hybrid_chunks, hybrid_output)

    custom_metrics = chunk_metrics(custom_chunks, chunk_size)
    hybrid_metrics = chunk_metrics(hybrid_chunks, chunk_size)
    comparison = {
        "document": str(document_path),
        "document_id": resolved_document_id,
        "configuration": {
            "tokenizer": "cl100k_base",
            "chunk_size": chunk_size,
            "custom_overlap": overlap,
            "docling_merge_peers": True,
            "docling_repeat_table_header": True,
        },
        "custom": {
            "implementation": "custom StructureAwareChunker",
            "output": str(custom_output),
            **custom_metrics,
        },
        "docling_hybrid": {
            "implementation": "docling.HybridChunker",
            "output": str(hybrid_output),
            **hybrid_metrics,
        },
        "difference": {
            "chunk_count": (
                hybrid_metrics["chunk_count"]
                - custom_metrics["chunk_count"]
            ),
            "average_tokens": round(
                hybrid_metrics["average_tokens"]
                - custom_metrics["average_tokens"],
                2,
            ),
            "table_chunks": (
                hybrid_metrics["table_chunks"]
                - custom_metrics["table_chunks"]
            ),
        },
    }

    comparison_path = output_directory / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path = output_directory / "comparison.md"
    report_path.write_text(
        render_document_report(comparison, custom_chunks, hybrid_chunks),
        encoding="utf-8",
    )
    comparison["comparison_output"] = str(comparison_path)
    comparison["report_output"] = str(report_path)
    return comparison


def run_comparison_pipeline(
    processed_directory: Path = PROCESSED_DATA_DIR,
    allowed_statuses: Collection[str] = DEFAULT_ALLOWED_STATUSES,
    chunk_size: int = 500,
    overlap: int = 50,
) -> dict[str, Any]:
    """Compare both chunkers across every validated processed document."""

    processed_directory = Path(processed_directory)
    allowed = {status.upper() for status in allowed_statuses}
    summary: dict[str, Any] = {
        "chunk_size": chunk_size,
        "custom_overlap": overlap,
        "total_documents": 0,
        "processed_documents": 0,
        "skipped_documents": 0,
        "failed_documents": 0,
        "documents": [],
    }

    document_paths = discover_documents(processed_directory)
    summary["total_documents"] = len(document_paths)
    for document_path in document_paths:
        validation = _load_json(document_path.parent / "validation_report.json")
        validation_status = str(validation.get("status", "MISSING")).upper()
        relative_document = document_path.relative_to(processed_directory)
        if validation_status not in allowed:
            summary["skipped_documents"] += 1
            summary["documents"].append(
                {
                    "document": str(relative_document),
                    "status": "SKIPPED",
                    "validation_status": validation_status,
                }
            )
            continue

        ingestion_metadata = _load_json(document_path.parent / "metadata.json")
        source = ingestion_metadata.get("source", {})
        if not isinstance(source, dict):
            source = {}
        document_id = str(
            source.get("sha256")
            or document_path.parent.relative_to(processed_directory).as_posix()
        )
        try:
            result = compare_document(
                document_path=document_path,
                chunk_size=chunk_size,
                overlap=overlap,
                document_id=document_id,
                document_metadata={
                    "source_file": source.get("relative_path"),
                    "source_sha256": source.get("sha256"),
                    "validation_status": validation_status,
                },
            )
        except Exception as error:
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
        summary["documents"].append(
            {
                "document": str(relative_document),
                "status": "COMPLETED",
                "validation_status": validation_status,
                "custom": _summary_metrics(result["custom"]),
                "docling_hybrid": _summary_metrics(result["docling_hybrid"]),
                "report": result["report_output"],
            }
        )

    summary["corpus"] = _corpus_metrics(summary["documents"])
    json_path = processed_directory / "structure_aware_comparison_summary.json"
    markdown_path = processed_directory / "structure_aware_comparison_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_path.write_text(render_corpus_report(summary), encoding="utf-8")
    summary["summary_output"] = str(json_path)
    summary["report_output"] = str(markdown_path)
    return summary


def chunk_metrics(chunks: Sequence[Chunk], chunk_size: int) -> dict[str, Any]:
    token_counts = [chunk.token_count or 0 for chunk in chunks]
    pages = {
        page_number for chunk in chunks for page_number in chunk.page_numbers
    }
    source_refs = {
        source_ref for chunk in chunks for source_ref in chunk.source_refs
    }
    return {
        "chunk_count": len(chunks),
        "total_tokens": sum(token_counts),
        "average_tokens": round(mean(token_counts), 2) if token_counts else 0,
        "median_tokens": round(median(token_counts), 2) if token_counts else 0,
        "minimum_tokens": min(token_counts, default=0),
        "maximum_tokens": max(token_counts, default=0),
        "chunks_over_configured_limit": sum(
            token_count > chunk_size for token_count in token_counts
        ),
        "table_chunks": sum(
            bool(chunk.metadata.get("contains_table")) for chunk in chunks
        ),
        "multi_page_chunks": sum(
            len(chunk.page_numbers) > 1 for chunk in chunks
        ),
        "unique_pages": len(pages),
        "unique_source_refs": len(source_refs),
    }


def render_document_report(
    comparison: dict[str, Any],
    custom_chunks: Sequence[Chunk],
    hybrid_chunks: Sequence[Chunk],
) -> str:
    configuration = comparison["configuration"]
    lines = [
        "# Structure-aware chunking comparison",
        "",
        f"Document: `{comparison['document']}`",
        "",
        (
            f"Both methods use `{configuration['tokenizer']}` with a "
            f"configured limit of {configuration['chunk_size']} tokens. "
            f"The custom method uses {configuration['custom_overlap']} "
            "tokens of overlap only when an oversized block is split."
        ),
        "",
        "| Metric | Custom | Docling HybridChunker |",
        "| --- | ---: | ---: |",
    ]
    for key, label in (
        ("chunk_count", "Chunks"),
        ("average_tokens", "Average tokens"),
        ("median_tokens", "Median tokens"),
        ("maximum_tokens", "Maximum tokens"),
        ("chunks_over_configured_limit", "Chunks over configured limit"),
        ("table_chunks", "Table chunks"),
        ("multi_page_chunks", "Multi-page chunks"),
        ("unique_pages", "Pages represented"),
        ("unique_source_refs", "Unique source references"),
    ):
        lines.append(
            f"| {label} | {comparison['custom'][key]} | "
            f"{comparison['docling_hybrid'][key]} |"
        )

    lines.extend(
        _sample_section("Custom implementation", custom_chunks)
        + _sample_section("Docling HybridChunker", hybrid_chunks)
        + [
            "## Complete outputs",
            "",
            "- `custom_chunks.json` contains every custom chunk.",
            "- `docling_hybrid_chunks.json` contains every native hybrid chunk.",
            "- `comparison.json` contains the complete metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def render_corpus_report(summary: dict[str, Any]) -> str:
    corpus = summary["corpus"]
    lines = [
        "# Structure-aware chunking corpus comparison",
        "",
        (
            f"Processed {summary['processed_documents']} documents; "
            f"skipped {summary['skipped_documents']}; "
            f"failed {summary['failed_documents']}."
        ),
        "",
        "| Metric | Custom | Docling HybridChunker |",
        "| --- | ---: | ---: |",
        f"| Total chunks | {corpus['custom']['chunk_count']} | {corpus['docling_hybrid']['chunk_count']} |",
        f"| Total tokens | {corpus['custom']['total_tokens']} | {corpus['docling_hybrid']['total_tokens']} |",
        f"| Average tokens | {corpus['custom']['average_tokens']} | {corpus['docling_hybrid']['average_tokens']} |",
        f"| Maximum tokens | {corpus['custom']['maximum_tokens']} | {corpus['docling_hybrid']['maximum_tokens']} |",
        f"| Chunks over configured limit | {corpus['custom']['chunks_over_configured_limit']} | {corpus['docling_hybrid']['chunks_over_configured_limit']} |",
        f"| Table chunks | {corpus['custom']['table_chunks']} | {corpus['docling_hybrid']['table_chunks']} |",
        "",
        "## Per-document results",
        "",
        "| Document | Custom chunks | Hybrid chunks | Custom avg | Hybrid avg |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for document in summary["documents"]:
        if document["status"] != "COMPLETED":
            continue
        lines.append(
            f"| {document['document']} | "
            f"{document['custom']['chunk_count']} | "
            f"{document['docling_hybrid']['chunk_count']} | "
            f"{document['custom']['average_tokens']} | "
            f"{document['docling_hybrid']['average_tokens']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _sample_section(title: str, chunks: Sequence[Chunk]) -> list[str]:
    selected = list(chunks[:3])
    table_chunk = next(
        (
            chunk
            for chunk in chunks
            if chunk.metadata.get("contains_table") and chunk not in selected
        ),
        None,
    )
    if table_chunk is not None:
        selected.append(table_chunk)

    lines = ["", f"## {title}: actual chunk samples", ""]
    for chunk in selected:
        lines.extend(
            [
                f"### Chunk {chunk.chunk_index}",
                "",
                (
                    f"Tokens: {chunk.token_count}; pages: "
                    f"{chunk.page_numbers}; section: {chunk.section_path}"
                ),
                "",
                "```text",
                chunk.text.replace("```", "'''"),
                "```",
                "",
            ]
        )
    return lines


def _summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    excluded = {"implementation", "output"}
    return {key: value for key, value in metrics.items() if key not in excluded}


def _corpus_metrics(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [document for document in documents if document["status"] == "COMPLETED"]
    result: dict[str, Any] = {}
    for method in ("custom", "docling_hybrid"):
        total_chunks = sum(document[method]["chunk_count"] for document in completed)
        total_tokens = sum(document[method]["total_tokens"] for document in completed)
        result[method] = {
            "chunk_count": total_chunks,
            "total_tokens": total_tokens,
            "average_tokens": (
                round(total_tokens / total_chunks, 2) if total_chunks else 0
            ),
            "maximum_tokens": max(
                (document[method]["maximum_tokens"] for document in completed),
                default=0,
            ),
            "chunks_over_configured_limit": sum(
                document[method]["chunks_over_configured_limit"]
                for document in completed
            ),
            "table_chunks": sum(
                document[method]["table_chunks"] for document in completed
            ),
        }
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as json_file:
        value = json.load(json_file)
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare custom and Docling hybrid structure-aware chunking."
    )
    parser.add_argument(
        "processed_directory",
        nargs="?",
        type=Path,
        default=PROCESSED_DATA_DIR,
    )
    args = parser.parse_args()
    result = run_comparison_pipeline(args.processed_directory)
    print(
        "Structure-aware comparison completed: "
        f"{result['processed_documents']} processed, "
        f"{result['failed_documents']} failed"
    )
