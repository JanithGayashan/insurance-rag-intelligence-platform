from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.evaluation.retrieval.dataset import (
    bind_evidence, corpus_fingerprint, load_dataset, select_questions, source_key, verify_pdf_sources,
)
from src.evaluation.retrieval.metrics import aggregate_metrics, retrieval_metrics
from src.indexing.configuration import json_sha256
from src.retrieval.dense import DenseRetriever


def evaluate_dense_retrieval(
    dataset_path: Path, retriever: DenseRetriever, output_directory: Path,
    *, k_values: Sequence[int] = (1, 3, 5, 10), include_source_checked: bool = False,
    modes: Sequence[str] = ("unfiltered", "policy_scoped"), raw_root: Path | None = None,
) -> dict[str, Any]:
    """Run a reproducible baseline, retaining every ranking and failure."""
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("K values must be positive")
    if not modes or any(mode not in {"unfiltered", "policy_scoped"} for mode in modes):
        raise ValueError("evaluation modes must be unfiltered or policy_scoped")
    dataset = load_dataset(dataset_path)
    questions = select_questions(dataset, include_source_checked)
    if "policy_scoped" in modes and any(not question.query_filters for question in questions):
        raise ValueError("policy_scoped mode requires explicit query context filters for every question")
    store = retriever.vector_store
    config = retriever.embedder.configuration
    store.check_configuration(config)
    snapshot = store.snapshot()
    corpus_hash = corpus_fingerprint(snapshot)
    if any(item["payload"].get("embedding_fingerprint") != config.fingerprint for item in snapshot.values()):
        raise ValueError("index contains incompatible embedding configurations")
    labels = bind_evidence(questions, snapshot)
    source_validation = verify_pdf_sources(questions, raw_root) if raw_root is not None else {"status": "NOT_RUN"}
    results = []
    maximum_k = max(5, max(k_values))
    for question in questions:
        evidence_groups = labels[question.question_id]
        relevant = set().union(*(set(ids) for ids in evidence_groups.values()))
        vector = retriever.embedder.embed_query(question.question)
        for mode in dict.fromkeys(modes):
            filters = question.query_filters if mode == "policy_scoped" else None
            started = time.perf_counter()
            hits = store.search(vector, limit=maximum_k, filters=filters)
            seconds = time.perf_counter() - started
            ranked_ids = [hit.chunk_id for hit in hits]
            unknown = set(ranked_ids) - {item["payload"]["chunk_id"] for item in snapshot.values()}
            if unknown:
                raise RuntimeError("index changed during evaluation")
            metrics = retrieval_metrics(ranked_ids, relevant, k_values)
            for k in sorted(set(k_values)):
                top = set(ranked_ids[:k])
                metrics[f"evidence_coverage@{k}"] = sum(bool(top & set(ids)) for ids in evidence_groups.values()) / len(evidence_groups)
            first = next((rank for rank, chunk_id in enumerate(ranked_ids, 1) if chunk_id in relevant), None)
            top5 = hits[:5]
            if not first or first > 5:
                reason = "no_results" if not top5 else (
                    "wrong_policy_in_top5" if not any(source_key(hit.source_file or "") in {source_key(item.source_file) for item in question.evidence} for hit in top5)
                    else "supporting_evidence_not_in_top5"
                )
            else:
                reason = None
            results.append({
                "question_id": question.question_id, "question": question.question,
                "category": question.category, "mode": mode, "review_status": question.review_status,
                "expected_answer": question.expected_answer,
                "evidence": [item.model_dump() for item in question.evidence],
                "relevant_chunk_ids": sorted(relevant), "evidence_groups": evidence_groups,
                "filters": filters or {}, "metrics": metrics, "first_relevant_rank": first,
                "failure_at_5": reason, "search_seconds": round(seconds, 6),
                "results": [{**hit.model_dump(mode="json"), "is_labeled_relevant": hit.chunk_id in relevant} for hit in hits],
            })
    if corpus_fingerprint(store.snapshot()) != corpus_hash:
        raise RuntimeError("index changed during evaluation; results were not published")
    by_mode = {}
    for mode in dict.fromkeys(modes):
        rows = [row for row in results if row["mode"] == mode]
        by_category = defaultdict(list)
        for row in rows:
            by_category[row["category"]].append(row["metrics"])
        by_mode[mode] = {
            "questions": len(rows), "metrics": aggregate_metrics([row["metrics"] for row in rows]),
            "by_category": {key: {"questions": len(values), "metrics": aggregate_metrics(values)} for key, values in sorted(by_category.items())},
            "failures_at_5": dict(Counter(row["failure_at_5"] for row in rows if row["failure_at_5"])),
        }
    status = "REVIEWED_BASELINE" if all(question.review_status == "approved" for question in questions) else "PROVISIONAL_BASELINE"
    summary = {
        "status": status, "dataset_id": dataset.dataset_id, "dataset_version": dataset.version,
        "dataset_sha256": json_sha256(dataset.model_dump()), "corpus_sha256": corpus_hash,
        "collection_name": store.collection_name, "embedding_configuration": config.model_dump(),
        "embedding_fingerprint": config.fingerprint, "indexed_chunks": len(snapshot),
        "questions_evaluated": len(questions), "questions_excluded": len(dataset.questions) - len(questions),
        "review_status_counts": dict(Counter(question.review_status for question in questions)),
        "k_values": sorted(set(k_values)), "label_scope": dataset.label_scope,
        "pdf_source_validation": source_validation, "by_mode": by_mode,
        "limitations": [
            "Source-checked labels are assistant-prepared until a named human approves them.",
            "This small hand-authored question set is a development baseline, not a representative production or held-out accuracy estimate.",
            "Recall measures annotated chunk labels; unannotated useful evidence may count as irrelevant.",
            "Policy-scoped results assume the application already knows the selected policy; they are separate from unfiltered search.",
            "This evaluates retrieval, not generated-answer accuracy or source faithfulness.",
        ],
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dense_retrieval_evaluation.json").write_text(json.dumps({"summary": summary, "queries": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "dense_retrieval_evaluation.md").write_text(_report(summary, results), encoding="utf-8")
    (output / "question_review.md").write_text(_review(dataset, labels, raw_root), encoding="utf-8")
    (output / "relevance_labels.json").write_text(json.dumps({"dataset_sha256": summary["dataset_sha256"], "corpus_sha256": corpus_hash, "labels": labels}, indent=2), encoding="utf-8")
    return summary


def _report(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = ["# Dense retrieval evaluation", "", f"Status: **{summary['status']}**", "",
             f"Questions: **{summary['questions_evaluated']}**; indexed chunks: **{summary['indexed_chunks']}**.", "",
             "| Mode | Hit Rate@5 | Recall@5 | MRR@5 | Evidence coverage@5 |", "| --- | --- | --- | --- | --- |"]
    for mode, value in summary["by_mode"].items():
        metrics = value["metrics"]
        lines.append(f"| {mode} | {metrics.get('hit_rate@5', 'N/A')} | {metrics.get('recall@5', 'N/A')} | {metrics.get('mrr@5', 'N/A')} | {metrics.get('evidence_coverage@5', 'N/A')} |")
    lines.extend(["", "MRR@K assigns zero when no labeled relevant passage appears within K. Recall counts annotated chunks; evidence coverage treats repeated chunks supporting the same passage as alternatives.", "",
                  "Policy-scoped search receives an explicit policy filter. These results are reported separately from unrestricted search.", "", "## Categories", ""])
    for mode, value in summary["by_mode"].items():
        lines.extend([f"### {mode}", "", "| Category | Questions | Hit Rate@5 | Recall@5 | MRR@5 |", "| --- | --- | --- | --- | --- |"])
        for category, values in value["by_category"].items():
            metrics = values["metrics"]
            lines.append(f"| {category} | {values['questions']} | {metrics.get('hit_rate@5', 'N/A')} | {metrics.get('recall@5', 'N/A')} | {metrics.get('mrr@5', 'N/A')} |")
        lines.append("")
    lines.extend(["## Failures at five", ""])
    for row in results:
        if row["failure_at_5"]:
            lines.extend([f"### {row['question_id']} / {row['mode']}", "", row["question"], "",
                          f"Reason: `{row['failure_at_5']}`; first relevant rank: {row['first_relevant_rank'] or 'not in retrieved results'}.", "",
                          f"Reference answer: {row['expected_answer']}", ""])
            for hit in row["results"][:5]:
                lines.append(f"- Rank {hit['rank']}: `{hit['source_file']}`, pages {hit['page_numbers']}; {' > '.join(hit['section_path'])}; labeled relevant={hit['is_labeled_relevant']}.")
            lines.append("")
    lines.extend(["## Interpretation limits", ""])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    return "\n".join(lines) + "\n"


def _review(dataset, labels, raw_root):
    lines = ["# Insurance question and relevance review", "",
             "Check the question, reference answer, policy version, evidence, and alternative supporting passages against the PDF. Edit the versioned dataset to set `review_status=approved` and a human `reviewer` only after review. Do not tune labels to the retriever's rankings.", ""]
    for question in dataset.questions:
        lines.extend([f"## {question.question_id}", "", f"**Question:** {question.question}", "",
                      f"**Reference answer:** {question.expected_answer}", "",
                      f"**Status:** {question.review_status}; category: {question.category}", ""])
        for evidence in question.evidence:
            path = Path(raw_root).joinpath(*source_key(evidence.source_file).split("/")) if raw_root else None
            source = f"[{evidence.source_file}](<{path.resolve().as_posix()}>)" if path else evidence.source_file
            lines.extend([f"Source: {source}; PDF pages: {evidence.pdf_pages}; section: {evidence.section}", ""])
            lines.extend(f"> {quote}\n" for quote in evidence.quotes)
            for alternative in evidence.alternatives:
                lines.extend([f"Alternative support: PDF pages {alternative.pdf_pages}; section: {alternative.section}", ""])
                lines.extend(f"> {quote}\n" for quote in alternative.quotes)
            lines.append("Supporting chunk IDs: " + ", ".join(labels.get(question.question_id, {}).get(evidence.evidence_id, [])))
            lines.append("")
    return "\n".join(lines) + "\n"
