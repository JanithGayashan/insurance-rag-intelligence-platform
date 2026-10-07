from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.chunking.models import Chunk
from src.chunking.sentence_paragraph.custom import (
    CustomSentenceParagraphChunker,
)
from src.chunking.sentence_paragraph.spacy import (
    SpacySentenceParagraphChunker,
)


PROCESSED_DATA_DIR = Path("data/processed/allianz")
ALLOWED_VALIDATION_STATUSES = frozenset({"PASS", "REVIEW"})


def compare_document(
    document_path: Path,
    output_directory: Path | None = None,
    chunk_size: int = 500,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Compare custom and spaCy natural-boundary chunking."""

    document_path = Path(document_path)
    output_directory = output_directory or (
        document_path.parent / "sentence_paragraph_comparison"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    resolved_document_id = document_id or document_path.parent.name

    custom_chunks = CustomSentenceParagraphChunker(
        chunk_size=chunk_size
    ).chunk_file(document_path, document_id=resolved_document_id)
    spacy_chunks = SpacySentenceParagraphChunker(
        chunk_size=chunk_size
    ).chunk_file(document_path, document_id=resolved_document_id)

    custom_path = output_directory / "custom_chunks.json"
    spacy_path = output_directory / "spacy_chunks.json"
    _write_chunks(custom_chunks, custom_path)
    _write_chunks(spacy_chunks, spacy_path)

    comparison = {
        "document": str(document_path),
        "document_id": resolved_document_id,
        "configuration": {
            "tokenizer": "cl100k_base",
            "chunk_size": chunk_size,
            "paragraphs_are_hard_boundaries": True,
            "overlap": 0,
        },
        "custom": {
            "implementation": "custom regex sentence detector",
            "output": str(custom_path),
            **chunk_metrics(custom_chunks, chunk_size),
        },
        "spacy": {
            "implementation": "spaCy rule-based Sentencizer",
            "output": str(spacy_path),
            **chunk_metrics(spacy_chunks, chunk_size),
        },
    }

    comparison_path = output_directory / "comparison.json"
    report_path = output_directory / "comparison.md"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(
        render_document_report(comparison, custom_chunks, spacy_chunks),
        encoding="utf-8",
    )
    comparison["comparison_output"] = str(comparison_path)
    comparison["report_output"] = str(report_path)
    return comparison


def run_comparison_pipeline(
    processed_directory: Path = PROCESSED_DATA_DIR,
    chunk_size: int = 500,
) -> dict[str, Any]:
    """Run both natural-boundary implementations on validated documents."""

    processed_directory = Path(processed_directory)
    document_paths = sorted(processed_directory.rglob("document.txt"))
    summary: dict[str, Any] = {
        "chunk_size": chunk_size,
        "total_documents": len(document_paths),
        "processed_documents": 0,
        "skipped_documents": 0,
        "failed_documents": 0,
        "documents": [],
    }

    for document_path in document_paths:
        validation = _load_json(document_path.parent / "validation_report.json")
        validation_status = str(validation.get("status", "MISSING")).upper()
        relative_document = document_path.relative_to(processed_directory)
        if validation_status not in ALLOWED_VALIDATION_STATUSES:
            summary["skipped_documents"] += 1
            summary["documents"].append(
                {
                    "document": str(relative_document),
                    "status": "SKIPPED",
                    "validation_status": validation_status,
                }
            )
            continue

        metadata = _load_json(document_path.parent / "metadata.json")
        source = metadata.get("source", {})
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
                document_id=document_id,
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
                "spacy": _summary_metrics(result["spacy"]),
                "report": result["report_output"],
            }
        )

    summary["corpus"] = _corpus_metrics(summary["documents"])
    json_path = processed_directory / "sentence_paragraph_summary.json"
    markdown_path = processed_directory / "sentence_paragraph_summary.md"
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
        "tiny_chunks_under_20_tokens": sum(
            token_count < 20 for token_count in token_counts
        ),
        "oversized_paragraph_chunks": sum(
            bool(chunk.metadata.get("oversized_paragraph")) for chunk in chunks
        ),
        "token_fallback_chunks": sum(
            bool(chunk.metadata.get("oversized_sentence_split"))
            for chunk in chunks
        ),
    }


def render_document_report(
    comparison: dict[str, Any],
    custom_chunks: Sequence[Chunk],
    spacy_chunks: Sequence[Chunk],
) -> str:
    lines = [
        "# Sentence/paragraph chunking comparison",
        "",
        f"Document: `{comparison['document']}`",
        "",
        (
            "Blank-line paragraphs are hard boundaries. A paragraph that fits "
            "is one chunk. An oversized paragraph is divided into sentences and "
            "packed up to the token limit. There is no overlap."
        ),
        "",
        "| Metric | Custom regex | spaCy Sentencizer |",
        "| --- | ---: | ---: |",
    ]
    for key, label in (
        ("chunk_count", "Chunks"),
        ("average_tokens", "Average tokens"),
        ("median_tokens", "Median tokens"),
        ("minimum_tokens", "Minimum tokens"),
        ("maximum_tokens", "Maximum tokens"),
        ("tiny_chunks_under_20_tokens", "Chunks under 20 tokens"),
        ("oversized_paragraph_chunks", "Chunks from oversized paragraphs"),
        ("token_fallback_chunks", "Token-fallback chunks"),
    ):
        lines.append(
            f"| {label} | {comparison['custom'][key]} | "
            f"{comparison['spacy'][key]} |"
        )

    lines.extend(
        _sample_section("Custom regex", custom_chunks)
        + _sample_section("spaCy Sentencizer", spacy_chunks)
        + [
            "## Complete outputs",
            "",
            "- `custom_chunks.json` contains all custom chunks.",
            "- `spacy_chunks.json` contains all spaCy chunks.",
            "- `comparison.json` contains the metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def render_corpus_report(summary: dict[str, Any]) -> str:
    corpus = summary["corpus"]
    lines = [
        "# Sentence/paragraph chunking corpus comparison",
        "",
        (
            f"Processed {summary['processed_documents']} documents; "
            f"skipped {summary['skipped_documents']}; "
            f"failed {summary['failed_documents']}."
        ),
        "",
        "| Metric | Custom regex | spaCy Sentencizer |",
        "| --- | ---: | ---: |",
    ]
    for key, label in (
        ("chunk_count", "Total chunks"),
        ("total_tokens", "Total tokens"),
        ("average_tokens", "Average tokens"),
        ("maximum_tokens", "Maximum tokens"),
        ("tiny_chunks_under_20_tokens", "Chunks under 20 tokens"),
        ("token_fallback_chunks", "Token-fallback chunks"),
    ):
        lines.append(
            f"| {label} | {corpus['custom'][key]} | "
            f"{corpus['spacy'][key]} |"
        )

    lines.extend(
        [
            "",
            "## Per-document results",
            "",
            "| Document | Custom chunks | spaCy chunks | Custom avg | spaCy avg |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for document in summary["documents"]:
        if document["status"] != "COMPLETED":
            continue
        lines.append(
            f"| {document['document']} | "
            f"{document['custom']['chunk_count']} | "
            f"{document['spacy']['chunk_count']} | "
            f"{document['custom']['average_tokens']} | "
            f"{document['spacy']['average_tokens']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _sample_section(title: str, chunks: Sequence[Chunk]) -> list[str]:
    selected = list(chunks[:5])
    oversized = next(
        (
            chunk
            for chunk in chunks
            if chunk.metadata.get("oversized_paragraph")
            and chunk not in selected
        ),
        None,
    )
    if oversized is not None:
        selected.append(oversized)

    lines = ["", f"## {title}: actual chunk samples", ""]
    for chunk in selected:
        lines.extend(
            [
                f"### Chunk {chunk.chunk_index}",
                "",
                (
                    f"Tokens: {chunk.token_count}; paragraph: "
                    f"{chunk.metadata['paragraph_index']}; sentences: "
                    f"{chunk.metadata['sentence_count']}"
                ),
                "",
                "```text",
                chunk.text.replace("```", "'''"),
                "```",
                "",
            ]
        )
    return lines


def _write_chunks(chunks: Sequence[Chunk], path: Path) -> None:
    path.write_text(
        json.dumps(
            [chunk.model_dump(mode="json") for chunk in chunks],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"implementation", "output"}
    }


def _corpus_metrics(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [document for document in documents if document["status"] == "COMPLETED"]
    result: dict[str, Any] = {}
    for method in ("custom", "spacy"):
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
            "tiny_chunks_under_20_tokens": sum(
                document[method]["tiny_chunks_under_20_tokens"]
                for document in completed
            ),
            "token_fallback_chunks": sum(
                document[method]["token_fallback_chunks"]
                for document in completed
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
        description="Compare custom and spaCy sentence/paragraph chunking."
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
        "Sentence/paragraph comparison completed: "
        f"{result['processed_documents']} processed, "
        f"{result['failed_documents']} failed"
    )
