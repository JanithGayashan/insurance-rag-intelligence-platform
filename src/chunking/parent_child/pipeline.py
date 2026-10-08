from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.chunking.parent_child.comparison import compare_document


PROCESSED_DATA_DIR = Path("data/processed/allianz")
ALLOWED_VALIDATION_STATUSES = frozenset({"PASS", "REVIEW"})


def run_parent_child_pipeline(
    processed_directory: Path = PROCESSED_DATA_DIR,
    parent_size: int = 1500,
    child_size: int = 300,
    child_overlap: int = 50,
    evaluation_questions_path: Path | None = None,
) -> dict[str, Any]:
    """Compare baseline and structured hierarchies on validated documents."""

    processed_directory = Path(processed_directory)
    evaluation_questions_path = evaluation_questions_path or (
        processed_directory / "parent_child_evaluation_questions.json"
    )
    evaluation_questions = _load_json_list(evaluation_questions_path)
    document_paths = sorted(processed_directory.rglob("document.txt"))
    summary: dict[str, Any] = {
        "configuration": {
            "parent_size": parent_size,
            "child_size": child_size,
            "child_overlap": child_overlap,
            "industry_child_overlap": 0,
        },
        "total_documents": len(document_paths),
        "evaluation_question_count": len(evaluation_questions),
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
                parent_size=parent_size,
                child_size=child_size,
                child_overlap=child_overlap,
                document_id=document_id,
                structured_document_path=document_path.with_suffix(".json"),
                evaluation_questions=evaluation_questions,
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
                "llamaindex": _summary_metrics(result["llamaindex"]),
                "industry": _summary_metrics(result["industry"]),
                "report": result["report_output"],
            }
        )

    summary["corpus"] = _corpus_metrics(summary["documents"])
    json_path = processed_directory / "parent_child_summary.json"
    report_path = processed_directory / "parent_child_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_path.write_text(render_corpus_report(summary), encoding="utf-8")
    summary["summary_output"] = str(json_path)
    summary["report_output"] = str(report_path)
    return summary


def render_corpus_report(summary: dict[str, Any]) -> str:
    corpus = summary["corpus"]
    lines = [
        "# Parent-child chunking corpus comparison",
        "",
        (
            f"Processed {summary['processed_documents']} documents; "
            f"skipped {summary['skipped_documents']}; "
            f"failed {summary['failed_documents']}."
        ),
        "",
        "| Metric | Custom baseline | LlamaIndex baseline | Industry structured |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("parent_count", "Total parents"),
        ("child_count", "Total children"),
        ("average_parent_tokens", "Average parent tokens"),
        ("average_child_tokens", "Average child tokens"),
        ("orphan_children", "Orphan children"),
        ("relationship_errors", "Relationship errors"),
        ("chunks_over_limit", "Chunks over limits"),
    ):
        lines.append(
            f"| {label} | {corpus['custom'][key]} | "
            f"{corpus['llamaindex'][key]} | {corpus['industry'][key]} |"
        )
    lines.extend(
        [
            "",
            "## Industry structured quality",
            "",
            (
                f"- Children crossing sections: "
                f"{corpus['industry']['children_crossing_sections']}"
            ),
            (
                f"- Missing-word rate: "
                f"{corpus['industry']['missing_word_rate']}"
            ),
            (
                f"- Duplicate-word rate: "
                f"{corpus['industry']['duplicate_word_rate']}"
            ),
            (
                f"- Sentence-boundary break rate: "
                f"{corpus['industry']['sentence_boundary_break_rate']}"
            ),
            (
                f"- Token-fallback chunks requiring review: "
                f"{corpus['industry']['fallback_token_split_chunks']}"
            ),
            (
                f"- Synthetic section retrieval MRR: "
                f"{corpus['industry']['section_retrieval_mrr']}"
            ),
            (
                f"- Synthetic section retrieval Recall@5: "
                f"{corpus['industry']['section_retrieval_recall_at_5']}"
            ),
            (
                "- Labeled question retrieval: "
                + (
                    "completed"
                    if summary["evaluation_question_count"]
                    else "not run; no reviewed question set was supplied"
                )
            ),
            (
                "- Answer accuracy and source faithfulness: not run; the answer "
                "generation stage and reviewed reference answers are not present"
            ),
            "",
            "## Per-document results",
            "",
            (
                "| Document | Custom parents/children | LlamaIndex parents/children "
                "| Industry parents/children |"
            ),
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for document in summary["documents"]:
        if document["status"] != "COMPLETED":
            continue
        lines.append(
            f"| {document['document']} | "
            f"{document['custom']['parent_count']}/{document['custom']['child_count']} | "
            f"{document['llamaindex']['parent_count']}/"
            f"{document['llamaindex']['child_count']} | "
            f"{document['industry']['parent_count']}/"
            f"{document['industry']['child_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"implementation", "output"}
    }


def _corpus_metrics(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [document for document in documents if document["status"] == "COMPLETED"]
    result: dict[str, Any] = {}
    for method in ("custom", "llamaindex", "industry"):
        parent_count = sum(document[method]["parent_count"] for document in completed)
        child_count = sum(document[method]["child_count"] for document in completed)
        parent_tokens = sum(
            document[method]["parent_tokens"]["total"] for document in completed
        )
        child_tokens = sum(
            document[method]["child_tokens"]["total"] for document in completed
        )
        result[method] = {
            "parent_count": parent_count,
            "child_count": child_count,
            "average_parent_tokens": (
                round(parent_tokens / parent_count, 2) if parent_count else 0
            ),
            "average_child_tokens": (
                round(child_tokens / child_count, 2) if child_count else 0
            ),
            "orphan_children": sum(
                document[method]["orphan_children"] for document in completed
            ),
            "relationship_errors": sum(
                document[method]["missing_child_references"]
                + document[method]["unknown_child_references"]
                + document[method]["duplicate_child_references"]
                for document in completed
            ),
            "chunks_over_limit": sum(
                document[method]["parent_chunks_over_limit"]
                + document[method]["child_chunks_over_limit"]
                for document in completed
            ),
        }
        if method == "industry":
            qualities = [document[method]["structure_quality"] for document in completed]
            source_words = sum(item["source_word_count"] for item in qualities)
            missing_words = sum(item["missing_word_count"] for item in qualities)
            duplicate_words = sum(item["duplicate_word_count"] for item in qualities)
            fallback_chunks = sum(
                item["fallback_token_split_chunks"] for item in qualities
            )
            retrievals = [
                document[method]["section_retrieval_evaluation"]
                for document in completed
                if document[method]["section_retrieval_evaluation"].get("status")
                == "COMPLETED"
            ]
            query_count = sum(item["query_count"] for item in retrievals)
            result[method].update(
                {
                    "children_crossing_sections": sum(
                        item["children_crossing_sections"] for item in qualities
                    ),
                    "missing_word_rate": (
                        round(missing_words / source_words, 6) if source_words else 0
                    ),
                    "duplicate_word_rate": (
                        round(duplicate_words / source_words, 6)
                        if source_words
                        else 0
                    ),
                    "sentence_boundary_break_rate": (
                        round(fallback_chunks / child_count, 6)
                        if child_count
                        else 0
                    ),
                    "fallback_token_split_chunks": fallback_chunks,
                    "tiny_chunks_under_threshold": sum(
                        item["tiny_chunks_under_threshold"] for item in qualities
                    ),
                    "section_retrieval_mrr": (
                        round(
                            sum(
                                item["mean_reciprocal_rank"] * item["query_count"]
                                for item in retrievals
                            )
                            / query_count,
                            6,
                        )
                        if query_count
                        else None
                    ),
                    "section_retrieval_recall_at_5": (
                        round(
                            sum(
                                item["recall_at_k"]["5"] * item["query_count"]
                                for item in retrievals
                            )
                            / query_count,
                            6,
                        )
                        if query_count
                        else None
                    ),
                    "section_retrieval_query_count": query_count,
                }
            )
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as json_file:
        value = json.load(json_file)
    return value if isinstance(value, dict) else {}


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as json_file:
        value = json.load(json_file)
    if not isinstance(value, list):
        raise ValueError("evaluation questions must be stored as a JSON list")
    return [item for item in value if isinstance(item, dict)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare baseline and Docling-structured parent-child chunking."
    )
    parser.add_argument(
        "processed_directory", nargs="?", type=Path, default=PROCESSED_DATA_DIR
    )
    parser.add_argument("--parent-size", type=int, default=1500)
    parser.add_argument("--child-size", type=int, default=300)
    parser.add_argument("--child-overlap", type=int, default=50)
    args = parser.parse_args()
    result = run_parent_child_pipeline(
        args.processed_directory,
        parent_size=args.parent_size,
        child_size=args.child_size,
        child_overlap=args.child_overlap,
    )
    print(
        "Parent-child comparison completed: "
        f"{result['processed_documents']} processed, "
        f"{result['failed_documents']} failed"
    )
