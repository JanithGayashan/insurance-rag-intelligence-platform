from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

import tiktoken

from src.chunking.parent_child.custom import CustomParentChildChunker
from src.chunking.parent_child.evaluation import (
    evaluate_question_retrieval,
    evaluate_section_retrieval,
    unavailable_answer_evaluation,
)
from src.chunking.parent_child.llamaindex import LlamaIndexParentChildChunker
from src.chunking.parent_child.models import (
    HierarchicalChunk,
    ParentChildHierarchy,
)
from src.chunking.parent_child.structured import StructuredParentChildChunker


def compare_document(
    document_path: Path,
    output_directory: Path | None = None,
    parent_size: int = 1500,
    child_size: int = 300,
    child_overlap: int = 50,
    document_id: str | None = None,
    structured_document_path: Path | None = None,
    evaluation_questions: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build baseline and production-oriented parent-child hierarchies."""

    document_path = Path(document_path)
    output_directory = output_directory or (
        document_path.parent / "parent_child_comparison"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    resolved_document_id = document_id or document_path.parent.name
    text = document_path.read_text(encoding="utf-8")
    source_tokens = len(tiktoken.get_encoding("cl100k_base").encode(text))
    structured_document_path = structured_document_path or (
        document_path.parent / "document.json"
    )
    if not structured_document_path.exists():
        raise FileNotFoundError(
            f"Docling document JSON was not found: {structured_document_path}"
        )

    custom = CustomParentChildChunker(
        parent_size=parent_size,
        child_size=child_size,
        child_overlap=child_overlap,
    ).chunk_text(text, resolved_document_id)
    llamaindex = LlamaIndexParentChildChunker(
        parent_size=parent_size,
        child_size=child_size,
        child_overlap=child_overlap,
    ).chunk_text(text, resolved_document_id)
    industry = StructuredParentChildChunker(
        parent_size=parent_size,
        child_size=child_size,
    ).chunk_file(structured_document_path, resolved_document_id)

    custom_path = output_directory / "custom_hierarchy.json"
    llamaindex_path = output_directory / "llamaindex_hierarchy.json"
    industry_path = output_directory / "industry_hierarchy.json"
    _write_hierarchy(custom, custom_path)
    _write_hierarchy(llamaindex, llamaindex_path)
    _write_hierarchy(industry, industry_path)
    _write_text_hierarchy(custom, output_directory / "custom_chunks.txt")
    _write_text_hierarchy(llamaindex, output_directory / "llamaindex_chunks.txt")
    _write_text_hierarchy(industry, output_directory / "industry_chunks.txt")

    industry_metrics = hierarchy_metrics(industry, source_tokens)
    industry_metrics.update(
        {
            "structure_quality": industry.metadata.get("quality", {}),
            "section_retrieval_evaluation": evaluate_section_retrieval(industry),
            "question_retrieval_evaluation": evaluate_question_retrieval(
                industry, evaluation_questions or []
            ),
            "answer_evaluation": unavailable_answer_evaluation(),
        }
    )

    comparison = {
        "document": str(document_path),
        "document_id": resolved_document_id,
        "configuration": {
            "tokenizer": "cl100k_base",
            "parent_size": parent_size,
            "parent_overlap": 0,
            "child_size": child_size,
            "child_overlap": child_overlap,
            "source_tokens": source_tokens,
            "industry_child_overlap": 0,
        },
        "custom": {
            "implementation": "project recursive splitter",
            "output": str(custom_path),
            **hierarchy_metrics(custom, source_tokens),
        },
        "llamaindex": {
            "implementation": "LlamaIndex HierarchicalNodeParser",
            "output": str(llamaindex_path),
            **hierarchy_metrics(llamaindex, source_tokens),
        },
        "industry": {
            "implementation": "Docling section parents with paragraph/sentence children",
            "output": str(industry_path),
            **industry_metrics,
        },
    }
    comparison_path = output_directory / "comparison.json"
    report_path = output_directory / "comparison.md"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(
        render_document_report(comparison, custom, llamaindex, industry),
        encoding="utf-8",
    )
    comparison["comparison_output"] = str(comparison_path)
    comparison["report_output"] = str(report_path)
    return comparison


def hierarchy_metrics(
    hierarchy: ParentChildHierarchy,
    source_tokens: int,
) -> dict[str, Any]:
    parent_counts = [chunk.token_count for chunk in hierarchy.parents]
    child_counts = [chunk.token_count for chunk in hierarchy.children]
    children_per_parent = [len(parent.children_ids) for parent in hierarchy.parents]
    parent_ids = {parent.chunk_id for parent in hierarchy.parents}
    child_ids = {child.chunk_id for child in hierarchy.children}
    referenced_child_ids = [
        child_id
        for parent in hierarchy.parents
        for child_id in parent.children_ids
    ]
    return {
        "parent_count": len(hierarchy.parents),
        "child_count": len(hierarchy.children),
        "parent_tokens": _token_statistics(parent_counts),
        "child_tokens": _token_statistics(child_counts),
        "average_children_per_parent": (
            round(mean(children_per_parent), 2) if children_per_parent else 0
        ),
        "maximum_children_per_parent": max(children_per_parent, default=0),
        "parent_chunks_over_limit": sum(
            count > hierarchy.parent_size for count in parent_counts
        ),
        "child_chunks_over_limit": sum(
            count > hierarchy.child_size for count in child_counts
        ),
        "orphan_children": sum(
            child.parent_id not in parent_ids for child in hierarchy.children
        ),
        "missing_child_references": len(child_ids - set(referenced_child_ids)),
        "unknown_child_references": len(set(referenced_child_ids) - child_ids),
        "duplicate_child_references": (
            len(referenced_child_ids) - len(set(referenced_child_ids))
        ),
        "indexed_child_tokens": sum(child_counts),
        "child_token_expansion_ratio": (
            round(sum(child_counts) / source_tokens, 3) if source_tokens else 0
        ),
    }


def render_document_report(
    comparison: dict[str, Any],
    custom: ParentChildHierarchy,
    llamaindex: ParentChildHierarchy,
    industry: ParentChildHierarchy,
) -> str:
    lines = [
        "# Parent-child chunking comparison",
        "",
        f"Document: `{comparison['document']}`",
        "",
        (
            "Each method creates larger parent chunks for context and smaller "
            "child chunks for retrieval. The industry method also runs a labeled "
            "section-title BM25 probe; reviewed business questions remain a "
            "separate evaluation input."
        ),
        "",
        "| Metric | Custom baseline | LlamaIndex baseline | Industry structured |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("parent_count", "Parents"),
        ("child_count", "Children"),
        ("average_children_per_parent", "Average children per parent"),
        ("maximum_children_per_parent", "Maximum children per parent"),
        ("orphan_children", "Orphan children"),
        ("missing_child_references", "Missing child references"),
        ("child_chunks_over_limit", "Children over token limit"),
        ("child_token_expansion_ratio", "Child token expansion ratio"),
    ):
        lines.append(
            f"| {label} | {comparison['custom'][key]} | "
            f"{comparison['llamaindex'][key]} | "
            f"{comparison['industry'][key]} |"
        )

    lines.extend(_sample_section("Custom", custom))
    lines.extend(_sample_section("LlamaIndex", llamaindex))
    lines.extend(_sample_section("Industry structured", industry))
    quality = comparison["industry"]["structure_quality"]
    retrieval = comparison["industry"]["section_retrieval_evaluation"]
    lines.extend(
        [
            "## Industry quality checks",
            "",
            f"- Children crossing sections: {quality['children_crossing_sections']}",
            f"- Missing-word rate: {quality['missing_word_rate']}",
            f"- Duplicate-word rate: {quality['duplicate_word_rate']}",
            (
                "- Sentence-boundary break rate: "
                f"{quality['sentence_boundary_break_rate']}"
            ),
            (
                "- Token-fallback chunks requiring review: "
                f"{quality['fallback_token_split_chunks']}"
            ),
            f"- Section retrieval MRR: {retrieval.get('mean_reciprocal_rank')}",
            f"- Section retrieval Recall@5: {retrieval.get('recall_at_k', {}).get('5')}",
            "",
            "## Complete outputs",
            "",
            "- `custom_hierarchy.json` contains every custom parent and child.",
            "- `llamaindex_hierarchy.json` contains every LlamaIndex parent and child.",
            "- `industry_hierarchy.json` contains every section-aware parent and child.",
            "- `custom_chunks.txt` and `llamaindex_chunks.txt` show baseline text.",
            "- `industry_chunks.txt` shows the complete readable hierarchy.",
            "- `comparison.json` contains validation and size metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def _sample_section(title: str, hierarchy: ParentChildHierarchy) -> list[str]:
    lines = ["", f"## {title}: actual hierarchy sample", ""]
    if not hierarchy.parents:
        return [*lines, "No chunks were produced.", ""]
    parent = hierarchy.parents[0]
    children_by_id = {child.chunk_id: child for child in hierarchy.children}
    lines.extend(
        [
            f"### Parent `{parent.chunk_id}`",
            "",
            f"Tokens: {parent.token_count}; linked children: {len(parent.children_ids)}",
            "",
            "```text",
            _safe_sample(parent.text),
            "```",
            "",
        ]
    )
    for child_id in parent.children_ids[:3]:
        child = children_by_id[child_id]
        lines.extend(
            [
                f"#### Child `{child.chunk_id}`",
                "",
                f"Tokens: {child.token_count}; parent: `{child.parent_id}`",
                "",
                "```text",
                _safe_sample(child.text),
                "```",
                "",
            ]
        )
    return lines


def _token_statistics(counts: Sequence[int]) -> dict[str, float | int]:
    return {
        "total": sum(counts),
        "average": round(mean(counts), 2) if counts else 0,
        "median": round(median(counts), 2) if counts else 0,
        "minimum": min(counts, default=0),
        "maximum": max(counts, default=0),
        "under_20": sum(count < 20 for count in counts),
    }


def _safe_sample(text: str, limit: int = 1200) -> str:
    text = text.replace("```", "'''")
    return text if len(text) <= limit else f"{text[:limit].rstrip()}…"


def _write_hierarchy(hierarchy: ParentChildHierarchy, path: Path) -> None:
    path.write_text(
        json.dumps(hierarchy.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_text_hierarchy(hierarchy: ParentChildHierarchy, path: Path) -> None:
    """Write full parent and child text in an easy-to-read hierarchy."""

    children_by_id = {child.chunk_id: child for child in hierarchy.children}
    sections: list[str] = []
    for parent_number, parent in enumerate(hierarchy.parents, start=1):
        sections.extend(
            [
                f"{'=' * 24} PARENT {parent_number} {'=' * 24}",
                parent.text,
            ]
        )
        for child_number, child_id in enumerate(parent.children_ids, start=1):
            child = children_by_id[child_id]
            sections.extend(
                [
                    f"{'-' * 24} CHILD {parent_number}.{child_number} {'-' * 24}",
                    child.text,
                ]
            )
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
