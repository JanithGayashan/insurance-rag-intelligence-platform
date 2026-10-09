from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.chunking.models import Chunk
from src.chunking.semantic.custom import CustomSemanticChunker
from src.chunking.semantic.embeddings import TextEmbedder
from src.chunking.semantic.evaluation import evaluate_semantic_chunks
from src.chunking.semantic.llamaindex import LlamaIndexSemanticChunker


def compare_semantic_chunkers(
    document_path: Path,
    output_directory: Path,
    document_id: str,
    embedder: TextEmbedder,
    document_metadata: Mapping[str, Any] | None = None,
    min_chunk_size: int = 80,
    target_chunk_size: int = 250,
    max_chunk_size: int = 450,
    breakpoint_percentile: float = 90.0,
    buffer_size: int = 1,
) -> dict[str, Any]:
    document_path = Path(document_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    document = json.loads(document_path.read_text(encoding="utf-8"))

    custom_chunker = CustomSemanticChunker(
        embedder,
        min_chunk_size,
        target_chunk_size,
        max_chunk_size,
        breakpoint_percentile,
        buffer_size,
    )
    llamaindex_chunker = LlamaIndexSemanticChunker(
        embedder,
        min_chunk_size,
        target_chunk_size,
        max_chunk_size,
        breakpoint_percentile,
        buffer_size,
    )
    blocks = custom_chunker.extract_semantic_blocks(document)
    source_text = " ".join(block.text for block in blocks)
    custom = custom_chunker.chunk_document(document, document_id, document_metadata)
    llamaindex = llamaindex_chunker.chunk_document(
        document, document_id, document_metadata
    )

    custom_json = output_directory / "custom_semantic_chunks.json"
    llamaindex_json = output_directory / "llamaindex_semantic_chunks.json"
    custom_text = output_directory / "custom_semantic_chunks.txt"
    llamaindex_text = output_directory / "llamaindex_semantic_chunks.txt"
    _write_chunks(custom, custom_json)
    _write_chunks(llamaindex, llamaindex_json)
    _write_readable(custom, custom_text)
    _write_readable(llamaindex, llamaindex_text)

    comparison = {
        "document": str(document_path),
        "document_id": document_id,
        "configuration": {
            "embedding_model": embedder.model_name,
            "tokenizer": "cl100k_base",
            "min_chunk_size": min_chunk_size,
            "target_chunk_size": target_chunk_size,
            "max_chunk_size": max_chunk_size,
            "breakpoint_percentile": breakpoint_percentile,
            "buffer_size": buffer_size,
            "hard_boundary": "Docling section path",
        },
        "custom": {
            "implementation": "custom buffered embedding-distance breakpoints",
            "json_output": str(custom_json),
            "text_output": str(custom_text),
            **evaluate_semantic_chunks(custom, source_text, embedder, max_chunk_size),
        },
        "llamaindex": {
            "implementation": "LlamaIndex SemanticSplitterNodeParser per section",
            "json_output": str(llamaindex_json),
            "text_output": str(llamaindex_text),
            **evaluate_semantic_chunks(
                llamaindex, source_text, embedder, max_chunk_size
            ),
        },
    }
    comparison_path = output_directory / "comparison.json"
    report_path = output_directory / "comparison.md"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_path.write_text(_render_report(comparison), encoding="utf-8")
    comparison["comparison_output"] = str(comparison_path)
    comparison["report_output"] = str(report_path)
    return comparison


def _write_chunks(chunks: Sequence[Chunk], path: Path) -> None:
    path.write_text(
        json.dumps(
            [chunk.model_dump(mode="json") for chunk in chunks],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_readable(chunks: Sequence[Chunk], path: Path) -> None:
    sections: list[str] = []
    for number, chunk in enumerate(chunks, 1):
        sections.extend(
            [
                f"{'=' * 24} CHUNK {number} {'=' * 24}",
                f"Section: {' > '.join(chunk.section_path) or 'Document introduction'}",
                f"Tokens: {chunk.token_count}",
                f"Boundary: {chunk.metadata.get('boundary_reason', 'LlamaIndex semantic breakpoint')}",
                f"Fallback token split: {chunk.metadata.get('fallback_token_split', False)}",
                "",
                chunk.text,
            ]
        )
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def _render_report(comparison: Mapping[str, Any]) -> str:
    custom = comparison["custom"]
    llama = comparison["llamaindex"]
    lines = [
        "# Semantic chunking comparison",
        "",
        f"Document: `{comparison['document']}`",
        "",
        "Both implementations use the same embedding model and are prevented from crossing Docling sections.",
        "",
        "| Metric | Custom | LlamaIndex |",
        "| --- | ---: | ---: |",
    ]
    for key, label in (
        ("chunk_count", "Chunks"),
        ("oversized_chunks", "Oversized chunks"),
        ("tiny_chunks", "Tiny chunks"),
        ("children_crossing_sections", "Section crossings"),
        ("fallback_token_split_chunks", "Fallback token splits"),
        ("missing_word_rate", "Missing-word rate"),
        ("duplicate_word_rate", "Duplicate-word rate"),
        ("mean_within_chunk_adjacent_similarity", "Within-chunk similarity"),
        ("mean_across_boundary_similarity", "Across-boundary similarity"),
        ("semantic_separation_margin", "Separation margin"),
    ):
        lines.append(f"| {label} | {custom[key]} | {llama[key]} |")
    lines.extend(
        [
            "",
            "## Retrieval diagnostic",
            "",
            f"- Custom MRR: {custom['section_retrieval_evaluation'].get('mean_reciprocal_rank')}",
            f"- Custom Recall@5: {custom['section_retrieval_evaluation'].get('recall_at_k', {}).get('5')}",
            f"- LlamaIndex MRR: {llama['section_retrieval_evaluation'].get('mean_reciprocal_rank')}",
            f"- LlamaIndex Recall@5: {llama['section_retrieval_evaluation'].get('recall_at_k', {}).get('5')}",
            "",
            "These section-title probes test structural retrievability. Selecting a production winner still requires reviewed insurance questions.",
            "",
            "## Outputs",
            "",
            "- `custom_semantic_chunks.json` and `llamaindex_semantic_chunks.json` contain complete metadata.",
            "- `custom_semantic_chunks.txt` and `llamaindex_semantic_chunks.txt` show the actual text boundaries.",
            "- `comparison.json` contains all automatic measurements.",
            "",
        ]
    )
    return "\n".join(lines)
