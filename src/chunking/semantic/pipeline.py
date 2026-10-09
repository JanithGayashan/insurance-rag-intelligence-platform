from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from statistics import mean
from typing import Any

from src.chunking.semantic.comparison import compare_semantic_chunkers
from src.chunking.semantic.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
    TextEmbedder,
)
from src.chunking.structure_aware.pipeline import discover_documents


PROCESSED_DATA_DIR = Path("data/processed/allianz")
DEFAULT_ALLOWED_STATUSES = frozenset({"PASS", "REVIEW"})


def run_semantic_chunking_pipeline(
    processed_directory: Path = PROCESSED_DATA_DIR,
    allowed_statuses: Collection[str] = DEFAULT_ALLOWED_STATUSES,
    embedder: TextEmbedder | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    min_chunk_size: int = 80,
    target_chunk_size: int = 250,
    max_chunk_size: int = 450,
    breakpoint_percentile: float = 90.0,
    buffer_size: int = 1,
) -> dict[str, Any]:
    """Run both semantic strategies over every validated Docling document."""

    processed_directory = Path(processed_directory)
    resolved_embedder = embedder or SentenceTransformerEmbedder(embedding_model)
    allowed = {status.upper() for status in allowed_statuses}
    summary: dict[str, Any] = {
        "strategy": "semantic-structure-constrained-comparison",
        "embedding_model": resolved_embedder.model_name,
        "configuration": {
            "min_chunk_size": min_chunk_size,
            "target_chunk_size": target_chunk_size,
            "max_chunk_size": max_chunk_size,
            "breakpoint_percentile": breakpoint_percentile,
            "buffer_size": buffer_size,
        },
        "total_documents": 0,
        "processed_documents": 0,
        "skipped_documents": 0,
        "failed_documents": 0,
        "documents": [],
    }
    document_paths = discover_documents(processed_directory)
    summary["total_documents"] = len(document_paths)
    for document_path in document_paths:
        directory = document_path.parent
        validation = _load_json(directory / "validation_report.json")
        status = str(validation.get("status", "MISSING")).upper()
        relative_document = document_path.relative_to(processed_directory)
        if status not in allowed:
            summary["skipped_documents"] += 1
            summary["documents"].append(
                {
                    "document": str(relative_document),
                    "status": "SKIPPED",
                    "validation_status": status,
                }
            )
            continue
        metadata = _load_json(directory / "metadata.json")
        source = metadata.get("source", {})
        source = source if isinstance(source, dict) else {}
        document_id = str(source.get("sha256") or relative_document.parent.as_posix())
        try:
            comparison = compare_semantic_chunkers(
                document_path,
                directory / "semantic_comparison",
                document_id,
                resolved_embedder,
                {
                    "source_file": source.get("relative_path"),
                    "source_sha256": source.get("sha256"),
                    "validation_status": status,
                },
                min_chunk_size,
                target_chunk_size,
                max_chunk_size,
                breakpoint_percentile,
                buffer_size,
            )
        except Exception as error:
            summary["failed_documents"] += 1
            summary["documents"].append(
                {
                    "document": str(relative_document),
                    "status": "FAILED",
                    "validation_status": status,
                    "error": str(error),
                }
            )
            continue
        summary["processed_documents"] += 1
        summary["documents"].append(
            {
                "document": str(relative_document),
                "document_id": document_id,
                "status": "COMPLETED",
                "validation_status": status,
                "custom": comparison["custom"],
                "llamaindex": comparison["llamaindex"],
                "report": str(
                    (directory / "semantic_comparison" / "comparison.md").relative_to(
                        processed_directory
                    )
                ),
            }
        )

    summary["corpus"] = _corpus_metrics(summary["documents"])
    json_path = processed_directory / "semantic_chunking_summary.json"
    markdown_path = processed_directory / "semantic_chunking_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(_render_summary(summary), encoding="utf-8")
    return summary


def _corpus_metrics(documents: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in documents if item["status"] == "COMPLETED"]
    result: dict[str, Any] = {}
    for method in ("custom", "llamaindex"):
        metrics = [item[method] for item in completed]
        retrieval = [
            item["section_retrieval_evaluation"]
            for item in metrics
            if item["section_retrieval_evaluation"].get("status") == "COMPLETED"
        ]
        result[method] = {
            "total_chunks": sum(item["chunk_count"] for item in metrics),
            "oversized_chunks": sum(item["oversized_chunks"] for item in metrics),
            "tiny_chunks": sum(item["tiny_chunks"] for item in metrics),
            "children_crossing_sections": sum(
                item["children_crossing_sections"] for item in metrics
            ),
            "fallback_token_split_chunks": sum(
                item["fallback_token_split_chunks"] for item in metrics
            ),
            "mean_missing_word_rate": _mean(metrics, "missing_word_rate"),
            "mean_duplicate_word_rate": _mean(metrics, "duplicate_word_rate"),
            "mean_semantic_separation_margin": _mean(
                metrics, "semantic_separation_margin"
            ),
            "mean_section_retrieval_mrr": round(
                mean(item["mean_reciprocal_rank"] for item in retrieval), 6
            )
            if retrieval
            else None,
            "mean_section_retrieval_recall_at_5": round(
                mean(item["recall_at_k"]["5"] for item in retrieval), 6
            )
            if retrieval
            else None,
        }
    return result


def _mean(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return round(mean(values), 6) if values else None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Semantic chunking corpus summary",
        "",
        f"Embedding model: `{summary['embedding_model']}`",
        "",
        f"Processed documents: {summary['processed_documents']} / {summary['total_documents']}",
        "",
        "| Metric | Custom | LlamaIndex |",
        "| --- | ---: | ---: |",
    ]
    custom = summary["corpus"]["custom"]
    llama = summary["corpus"]["llamaindex"]
    for key, label in (
        ("total_chunks", "Chunks"),
        ("oversized_chunks", "Oversized chunks"),
        ("tiny_chunks", "Tiny chunks"),
        ("children_crossing_sections", "Section crossings"),
        ("fallback_token_split_chunks", "Fallback token splits"),
        ("mean_missing_word_rate", "Mean missing-word rate"),
        ("mean_duplicate_word_rate", "Mean duplicate-word rate"),
        ("mean_semantic_separation_margin", "Mean semantic separation margin"),
        ("mean_section_retrieval_mrr", "Mean section retrieval MRR"),
        ("mean_section_retrieval_recall_at_5", "Mean section retrieval Recall@5"),
    ):
        lines.append(f"| {label} | {custom[key]} | {llama[key]} |")
    lines.extend(
        [
            "",
            "The retrieval values use section titles as automatic probes. A reviewed insurance question set is still required before selecting the production strategy.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_semantic_chunking_pipeline()
    print(
        "Semantic chunking completed: "
        f"{result['processed_documents']} processed, "
        f"{result['skipped_documents']} skipped, "
        f"{result['failed_documents']} failed"
    )
