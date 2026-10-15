"""Compare LlamaIndex retrievers with pooled, explicitly incomplete judgments."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median

import numpy as np

from src.evaluation.retrieval.dataset import (
    bind_evidence, corpus_fingerprint, load_dataset, select_questions, verify_pdf_sources,
)
from src.evaluation.retrieval.metrics import aggregate_judged_metrics, judged_retrieval_metrics
from src.evaluation.retrieval.models import PoolJudgment
from src.evaluation.retrieval.pooling import (
    accepted_judgments, build_pool, make_candidate, merge_judgments, review_markdown, text_hash,
)
from src.indexing.configuration import json_sha256
from src.retrieval.hybrid import LlamaIndexRetrievalSuite


def evaluate_retrieval_comparison(
    dataset_path, dense_retriever, output_directory, *, methods=("dense", "bm25", "hybrid"),
    pool_depth=20, candidate_k=20, k_values=(1, 3, 5, 10), include_source_checked=False,
    modes=("unfiltered", "policy_scoped"), raw_root=None, judgments_path=None,
    outside_pool_sample=2, require_judged_top5=False,
    reranker=None, rerank_candidates=20, progress_callback=None,
):
    if not k_values or any(k <= 0 for k in k_values) or pool_depth < max(5, max(k_values)):
        raise ValueError("pool_depth must cover all positive metric cutoffs, including five")
    if candidate_k < pool_depth or outside_pool_sample < 0:
        raise ValueError("candidate_k must cover pool_depth; outside sample cannot be negative")
    if not methods or any(method not in LlamaIndexRetrievalSuite.METHODS for method in methods):
        raise ValueError("methods must be dense, bm25, hybrid or hybrid_reranked")
    if "hybrid_reranked" in methods and (reranker is None or pool_depth != rerank_candidates or candidate_k < rerank_candidates):
        raise ValueError("reranking evaluation requires a reranker and pool_depth == rerank_candidates <= candidate_k")
    if not modes or any(mode not in {"unfiltered", "policy_scoped"} for mode in modes):
        raise ValueError("modes must be unfiltered or policy_scoped")
    k_values = tuple(sorted(set(k_values) | {5}))
    methods, modes = tuple(dict.fromkeys(methods)), tuple(dict.fromkeys(modes))
    dataset = load_dataset(Path(dataset_path))
    questions = select_questions(dataset, include_source_checked)
    if "policy_scoped" in modes and any(not question.query_filters for question in questions):
        raise ValueError("policy_scoped mode requires explicit query context filters")
    store, config = dense_retriever.vector_store, dense_retriever.embedder.configuration
    store.check_configuration(config)
    snapshot = store.snapshot()
    corpus_hash, dataset_hash = corpus_fingerprint(snapshot), json_sha256(dataset.model_dump())
    if any(item["payload"].get("embedding_fingerprint") != config.fingerprint for item in snapshot.values()):
        raise ValueError("index contains incompatible embedding configurations")
    labels = bind_evidence(questions, snapshot)
    source_validation = verify_pdf_sources(questions, raw_root) if raw_root is not None else {"status": "NOT_RUN"}
    suite = LlamaIndexRetrievalSuite(dense_retriever, snapshot=snapshot, candidate_k=candidate_k,
                                    reranker=reranker, rerank_candidates=rerank_candidates)
    runs = []
    for question in questions:
        for mode in modes:
            filters = question.query_filters if mode == "policy_scoped" else None
            started = time.perf_counter()
            hits, diagnostics = suite.retrieve_all_with_diagnostics(question.question, pool_depth, filters, methods)
            elapsed = time.perf_counter() - started
            for method in methods:
                runs.append({"question_id": question.question_id, "method": method, "mode": mode,
                             "filters": filters or {}, "hits": hits[method],
                             "shared_search_seconds": round(elapsed, 6),
                             "search_seconds": diagnostics["method_seconds"][method],
                             "rerank_seconds": diagnostics.get("rerank_seconds", 0) if method == "hybrid_reranked" else 0,
                             "rerank_candidate_ids": diagnostics.get("rerank_candidate_ids", []),
                             "scored_windows": diagnostics.get("rerank_window_count", 0) if method == "hybrid_reranked" else 0})
            if progress_callback is not None:
                progress_callback({"question_id": question.question_id, "mode": mode,
                                   "completed": len(runs) // len(methods), "total": len(questions) * len(modes),
                                   "seconds": elapsed, "rerank_seconds": diagnostics.get("rerank_seconds", 0)})
    candidates, seed = build_pool(questions, snapshot, runs, labels, dataset_hash, corpus_hash)
    # A reproducible audit sample outside the retrieval pool can reveal shared misses.
    # Audited chunks do not enter rankings or become negative labels automatically.
    payloads = {item["payload"]["chunk_id"]: item["payload"] for item in snapshot.values()}
    existing = {(item["question_id"], item["chunk_id"]) for item in candidates}
    for question in questions:
        unseen = [chunk_id for chunk_id in payloads if (question.question_id, chunk_id) not in existing]
        unseen.sort(key=lambda chunk: hashlib.sha256(f"{corpus_hash}:{question.question_id}:{chunk}".encode()).hexdigest())
        for chunk_id in unseen[:outside_pool_sample]:
            payload = payloads[chunk_id]
            candidate = make_candidate(question, payload)
            candidate["origins"] = [{"method": "outside_pool_audit", "mode": "audit", "rank": None}]
            candidates.append(candidate)
            seed.judgments.append(PoolJudgment(question_id=question.question_id, chunk_id=chunk_id,
                                               text_sha256=text_hash(payload["text"])))
    existing_review = Path(output_directory) / "judgments.json"
    review_path = judgments_path if judgments_path is not None else (existing_review if existing_review.exists() else None)
    assessments = merge_judgments(seed, review_path, questions, snapshot)
    accepted = accepted_judgments(assessments, include_source_checked)
    question_map = {question.question_id: question for question in questions}
    # Preserve additional reviewed pairs supplied from an earlier/different pool.
    candidate_keys = {(item["question_id"], item["chunk_id"]) for item in candidates}
    for assessment in assessments.judgments:
        if (assessment.question_id, assessment.chunk_id) not in candidate_keys:
            candidate = make_candidate(question_map[assessment.question_id], payloads[assessment.chunk_id])
            candidate["origins"] = [{"method": "imported_judgment", "mode": "review", "rank": None}]
            candidates.append(candidate)
    assessment_map = {(item.question_id, item.chunk_id): item for item in assessments.judgments}
    # Review may discover additional passages supporting an existing required fact.
    for item in assessments.judgments:
        if accepted.get(item.question_id, {}).get(item.chunk_id) == 1:
            for evidence_id in item.evidence_ids:
                labels[item.question_id][evidence_id] = sorted(set(labels[item.question_id][evidence_id]) | {item.chunk_id})
    rows = []
    for run in runs:
        question = question_map[run["question_id"]]
        metrics = judged_retrieval_metrics(
            [hit.chunk_id for hit in run["hits"]], accepted.get(question.question_id, {}),
            labels[question.question_id], k_values,
        )
        if require_judged_top5 and metrics.get("precision@5") is None:
            raise ValueError("top-five results have unjudged pairs; complete the pool review before enforcing exact precision")
        rows.append({
            **{key: value for key, value in run.items() if key != "hits"},
            "question": question.question, "category": question.category,
            "evidence_groups": labels[question.question_id], "metrics": metrics,
            "results": [{**hit.model_dump(mode="json"),
                         "relevance": accepted.get(question.question_id, {}).get(hit.chunk_id),
                         "review_status": assessment_map[(question.question_id, hit.chunk_id)].review_status}
                        for hit in run["hits"]],
        })
    if corpus_fingerprint(store.snapshot()) != corpus_hash:
        raise RuntimeError("index changed during evaluation; results were not published")
    by_method = {}
    for method in methods:
        by_method[method] = {}
        for mode in modes:
            selected = [row for row in rows if row["method"] == method and row["mode"] == mode]
            by_category = defaultdict(list)
            for row in selected:
                by_category[row["category"]].append(row["metrics"])
            by_method[method][mode] = {
                **aggregate_judged_metrics([row["metrics"] for row in selected]),
                "by_category": {name: aggregate_judged_metrics(values) for name, values in sorted(by_category.items())},
                "latency_seconds": {"mean": mean(row["search_seconds"] for row in selected),
                                    "median": median(row["search_seconds"] for row in selected),
                                    "p95": float(np.percentile([row["search_seconds"] for row in selected], 95))},
            }
    priority = {(item["question_id"], item["chunk_id"]) for item in candidates if item["top5_priority"]}
    unresolved = [key for key in priority if key[1] not in accepted.get(key[0], {})]
    summary = {
        "status": "PROVISIONAL_BASELINE" if any(question.review_status != "approved" for question in questions)
                  or unresolved or any(item.review_status == "source_checked" for item in assessments.judgments)
                  else "REVIEWED_TOP5_POOLED_BASELINE",
        "dataset_id": dataset.dataset_id, "dataset_sha256": dataset_hash, "corpus_sha256": corpus_hash,
        "embedding_fingerprint": config.fingerprint, "embedding_configuration": config.model_dump(),
        "collection_name": store.collection_name, "indexed_chunks": len(snapshot),
        "questions_evaluated": len(questions), "k_values": sorted(set(k_values)),
        "methods": list(methods), "modes": list(modes), "pool_depth": pool_depth,
        "fusion_candidate_k": candidate_k, "fusion": "LlamaIndex reciprocal_rerank; equal contribution; rank constant 60; num_queries=1",
        "libraries": {name: version(name) for name in ("llama-index-core", "llama-index-retrievers-bm25", "bm25s", "PyStemmer")},
        "pdf_source_validation": source_validation, "by_method": by_method,
        "reranker": {"configuration": reranker.configuration, "fingerprint": reranker.fingerprint,
                     "load_seconds": reranker.load_seconds, "candidates": rerank_candidates}
                    if "hybrid_reranked" in methods else None,
        "pool": {"question_chunk_pairs": len(candidates), "top5_pairs": len(priority),
                 "unjudged_top5_pairs": len(unresolved), "outside_pool_sample_per_question": outside_pool_sample,
                 "judgment_status_counts": dict(Counter(item.review_status for item in assessments.judgments)),
                 "accepted_relevant_pairs": sum(value == 1 for values in accepted.values() for value in values.values()),
                 "accepted_irrelevant_pairs": sum(value == 0 for values in accepted.values() for value in values.values())},
        "limitations": [
            "Unjudged pairs remain unknown; null top-K metrics are unresolved, not zero. Bounds reflect only missing judgments, not annotation errors.",
            "Pooled recall measures accepted annotated positives, not all relevant chunks in the corpus.",
            "Evidence coverage counts reviewed/reference evidence units once; it is not generated-answer accuracy or exhaustive semantic matching.",
            "Source-checked labels are assistant-prepared until a named human approves them; pool membership is not a relevance judgment.",
            "The outside-pool audit is a diagnostic sample, not a statistical estimate of full-corpus recall.",
            "Known-policy results receive explicit policy context and are reported separately from unfiltered search.",
            "Forty development questions are not a held-out production benchmark. BM25 statistics are recomputed within each filtered corpus.",
        ],
    }
    if "hybrid" in methods and "hybrid_reranked" in methods:
        summary["paired_hybrid_comparison"] = paired_hybrid_comparison(rows, modes)
        summary["limitations"].extend([
            "Reranking reorders the same fixed hybrid candidate set; it cannot recover passages missing from that set.",
            "Oversized query/passage pairs use temporary token windows and maximum-score aggregation. More windows can affect scoring.",
            "Latency uses shared sequential component timings, includes first-use BM25 work, and excludes reranker startup. It is not API or production latency.",
        ])
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "retrieval_comparison.json": {"summary": summary, "queries": rows},
        "pool.json": {"dataset_sha256": dataset_hash, "corpus_sha256": corpus_hash,
                      "questions": [question.model_dump() for question in questions], "candidates": candidates},
        "judgments.json": assessments.model_dump(),
    }
    for filename, value in artifacts.items():
        (output / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "pool_review.md").write_text(review_markdown(candidates, assessments), encoding="utf-8")
    (output / "retrieval_comparison.md").write_text(comparison_markdown(summary), encoding="utf-8")
    if summary.get("paired_hybrid_comparison"):
        (output / "ranking_changes.md").write_text(ranking_changes_markdown(rows, modes), encoding="utf-8")
    return summary


def comparison_markdown(summary):
    lines = ["# LlamaIndex retrieval comparison", "", f"Status: **{summary['status']}**", "",
             "Precision, Hit Rate and MRR are N/A when unresolved judgments could change them. "
             "Precision bounds assume unknown pairs could be either relevant or irrelevant; they are not confidence intervals.", "",
             "| Method | Mode | Precision@5 | Precision bounds | Judged fraction@5 | Hit lower bound@5 | Pooled recall@5 | MRR bounds@5 | Evidence coverage@5 | Mean seconds |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    def show(value):
        return "N/A" if value is None else f"{value:.4f}"
    for method, modes in summary["by_method"].items():
        for mode, values in modes.items():
            m = values["metrics"]
            lines.append(f"| {method} | {mode} | {show(m.get('precision@5'))} | "
                         f"{show(m.get('precision_lower_bound@5'))}–{show(m.get('precision_upper_bound@5'))} | "
                         f"{show(m.get('judged_fraction@5'))} | {show(m.get('hit_rate_lower_bound@5'))} | "
                         f"{show(m.get('pooled_recall@5'))} | {show(m.get('mrr_lower_bound@5'))}–{show(m.get('mrr_upper_bound@5'))} | "
                         f"{show(m.get('evidence_coverage@5'))} | {values['latency_seconds']['mean']:.4f} |")
    if summary.get("paired_hybrid_comparison"):
        lines.extend(["", "## Paired hybrid comparison", "",
                      "Both rankings use the exact same hybrid candidate IDs. Changes below concern annotated evidence, not production answer accuracy.", ""])
        for mode, comparison in summary["paired_hybrid_comparison"].items():
            lines.append(f"- {mode}: evidence coverage@5 improved on {comparison['improved_questions']} questions, "
                         f"regressed on {comparison['regressed_questions']}, tied on {comparison['tied_questions']}; "
                         f"mean added reranking time {comparison['mean_rerank_seconds']:.4f} seconds.")
            lines.append(f"  Annotated evidence is missing from the candidate shortlist for: "
                         f"{', '.join(comparison['questions_missing_candidate_evidence']) or 'none'}.")
    lines.extend(["", f"Pool: {summary['pool']['question_chunk_pairs']} question/chunk pairs; "
                  f"{summary['pool']['unjudged_top5_pairs']} top-five pairs still unjudged.", "",
                  "Review pool_review.md against source PDFs and edit judgments.json. Rerun with --judgments pointing to "
                  "that file. Keep approved labels only for actual human review; --include-source-checked admits provisional judgments. "
                  "Use --require-judged-top5 to refuse publishing an incomplete top-five evaluation.", "",
                  "## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"


def paired_hybrid_comparison(rows, modes):
    lookup = {(row["question_id"], row["mode"], row["method"]): row for row in rows}
    result = {}
    for mode in modes:
        pairs = [(row, lookup[(row["question_id"], mode, "hybrid_reranked")])
                 for row in rows if row["mode"] == mode and row["method"] == "hybrid"]
        changes = []
        for baseline, reranked in pairs:
            ids = baseline["rerank_candidate_ids"]
            if ids != reranked["rerank_candidate_ids"] or set(ids) != {hit["chunk_id"] for hit in baseline["results"]} or set(ids) != {hit["chunk_id"] for hit in reranked["results"]}:
                raise RuntimeError("hybrid comparison did not preserve the candidate set")
            delta = reranked["metrics"]["evidence_coverage@5"] - baseline["metrics"]["evidence_coverage@5"]
            first = lambda row: next((rank for rank, hit in enumerate(row["results"], 1) if hit["relevance"] == 1), None)
            known = {hit["chunk_id"] for hit in baseline["results"] if hit["relevance"] == 1}
            candidate_coverage = sum(bool(known & set(ids)) for ids in baseline["evidence_groups"].values()) / len(baseline["evidence_groups"])
            changes.append({"question_id": baseline["question_id"], "evidence_coverage_delta@5": delta,
                            "hybrid_first_labeled_rank": first(baseline), "reranked_first_labeled_rank": first(reranked),
                            "candidate_ids": ids, "candidate_evidence_coverage": candidate_coverage,
                            "rerank_seconds": reranked["rerank_seconds"]})
        result[mode] = {"improved_questions": sum(row["evidence_coverage_delta@5"] > 0 for row in changes),
                        "regressed_questions": sum(row["evidence_coverage_delta@5"] < 0 for row in changes),
                        "tied_questions": sum(row["evidence_coverage_delta@5"] == 0 for row in changes),
                        "questions_missing_candidate_evidence": [row["question_id"] for row in changes if row["candidate_evidence_coverage"] < 1],
                        "mean_candidate_evidence_coverage": mean(row["candidate_evidence_coverage"] for row in changes),
                        "mean_rerank_seconds": mean(row["rerank_seconds"] for row in changes), "questions": changes}
    return result


def ranking_changes_markdown(rows, modes):
    lines = ["# Actual hybrid rankings before and after reranking", "",
             "Every evaluated question/context is included. Scores have different scales: hybrid uses RRF, "
             "reranked uses cross-encoder scores. Unknown relevance is displayed as None, not irrelevant. "
             "Full original chunk text is retained in retrieval_comparison.json; excerpts below are shortened for reading.", ""]
    lookup = {(row["question_id"], row["mode"], row["method"]): row for row in rows}
    for mode in modes:
        for row in rows:
            if row["mode"] != mode or row["method"] != "hybrid":
                continue
            lines.extend([f"## {row['question_id']} / {mode}", "", row["question"], ""])
            for method in ("hybrid", "hybrid_reranked"):
                current = lookup[(row["question_id"], mode, method)]
                lines.extend([f"### {method}", "",
                              f"Annotated evidence coverage@5: {current['metrics']['evidence_coverage@5']}; "
                              f"measured seconds: {current['search_seconds']:.4f}", ""])
                for hit in current["results"][:5]:
                    lines.extend([f"**Rank {hit['rank']}** — {hit['source_file']}; PDF pages {hit['page_numbers']}; "
                                  f"score {hit['score']:.6f}; relevance {hit['relevance']}", "",
                                  f"Chunk: `{hit['chunk_id']}`", "",
                                  hit["text"][:500] + ("…" if len(hit["text"]) > 500 else ""), ""])
    return "\n".join(lines) + "\n"
