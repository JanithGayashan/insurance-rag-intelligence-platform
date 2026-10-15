"""Diverse candidate pooling and auditable, snapshot-bound relevance judgments."""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.evaluation.retrieval.dataset import normalized_text, source_key
from src.evaluation.retrieval.models import JudgmentSet, PoolJudgment


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_candidate(question, payload):
    return {
        "question_id": question.question_id, "chunk_id": payload["chunk_id"],
        "question": question.question, "expected_answer": question.expected_answer,
        "text": payload["text"], "text_sha256": text_hash(payload["text"]),
        "source_file": payload.get("source_file"), "source_sha256": payload.get("source_sha256"),
        "page_numbers": payload.get("page_numbers", []), "section_path": payload.get("section_path", []),
        "origins": [], "top5_priority": False,
    }


def build_pool(questions, snapshot, runs, evidence_labels, dataset_sha256, corpus_sha256):
    payloads = {item["payload"]["chunk_id"]: item["payload"] for item in snapshot.values()}
    question_map = {question.question_id: question for question in questions}
    candidates = {}

    def add(question_id, chunk_id):
        key = (question_id, chunk_id)
        if chunk_id not in payloads:
            raise RuntimeError("index changed; retrieved chunk is absent from snapshot")
        if key not in candidates:
            candidates[key] = make_candidate(question_map[question_id], payloads[chunk_id])
        return candidates[key]

    for run in runs:
        for hit in run["hits"]:
            candidate = add(run["question_id"], hit.chunk_id)
            candidate["origins"].append({"method": run["method"], "mode": run["mode"], "rank": hit.rank})
            candidate["top5_priority"] |= hit.rank <= 5
    assessments = []
    for question in questions:
        for ids in evidence_labels[question.question_id].values():
            for chunk_id in ids:
                add(question.question_id, chunk_id)
    for (question_id, chunk_id), candidate in sorted(candidates.items()):
        question = question_map[question_id]
        ids = [evidence_id for evidence_id, chunks in evidence_labels[question_id].items() if chunk_id in chunks]
        quotes = []
        for passage in question.evidence:
            if passage.evidence_id not in ids:
                continue
            for location in [passage, *passage.alternatives]:
                if (set(candidate["page_numbers"]) & set(location.pdf_pages)
                        and all(normalized_text(quote) in normalized_text(candidate["text"]) for quote in location.quotes)):
                    quotes.extend(location.quotes)
        assessments.append(PoolJudgment(
            question_id=question_id, chunk_id=chunk_id, text_sha256=candidate["text_sha256"],
            relevance=1 if ids else None,
            review_status=question.review_status if ids else "unjudged",
            reviewer=question.reviewer if ids else None,
            rationale="Bound to source evidence quotations before retrieval." if ids else "",
            supporting_quotes=list(dict.fromkeys(quotes)), evidence_ids=ids,
        ))
    return list(candidates.values()), JudgmentSet(
        dataset_sha256=dataset_sha256, corpus_sha256=corpus_sha256, judgments=assessments,
    )


def merge_judgments(seed, path: Path | None, questions, snapshot):
    if path is None:
        return seed
    imported = JudgmentSet.model_validate_json(Path(path).read_text(encoding="utf-8"))
    if (imported.dataset_sha256, imported.corpus_sha256) != (seed.dataset_sha256, seed.corpus_sha256):
        raise ValueError("judgments belong to a different dataset or corpus snapshot")
    payloads = {item["payload"]["chunk_id"]: item["payload"] for item in snapshot.values()}
    question_map = {question.question_id: question for question in questions}
    merged = {(item.question_id, item.chunk_id): item for item in seed.judgments}
    for item in imported.judgments:
        if item.question_id not in question_map or item.chunk_id not in payloads:
            raise ValueError("judgment references an unknown question or chunk")
        text = payloads[item.chunk_id]["text"]
        if item.text_sha256 != text_hash(text):
            raise ValueError("judgment chunk text changed")
        if any(not normalized_text(quote) or normalized_text(quote) not in normalized_text(text)
               for quote in item.supporting_quotes):
            raise ValueError("judgment supporting quote is absent from chunk")
        if set(item.evidence_ids) - {evidence.evidence_id for evidence in question_map[item.question_id].evidence}:
            raise ValueError("judgment references unknown evidence units")
        if item.relevance == 1 and not any(
            source_key(payloads[item.chunk_id].get("source_file") or "") == source_key(evidence.source_file)
            and payloads[item.chunk_id].get("source_sha256") == evidence.source_sha256
            for evidence in question_map[item.question_id].evidence
        ):
            raise ValueError("positive judgment references the wrong policy or document version")
        key = (item.question_id, item.chunk_id)
        existing = merged.get(key)
        if existing and existing.relevance == 1 and item.relevance == 0:
            raise ValueError("negative judgment contradicts source evidence; revise the question dataset first")
        if existing and existing.review_status == "approved" and item.review_status != "approved":
            continue
        if item.relevance is not None:
            # Preserve source-established evidence bindings when a reviewer only updates relevance.
            merged[key] = item.model_copy(update={"evidence_ids": sorted(set(item.evidence_ids) | set(existing.evidence_ids if existing else []))})
    return seed.model_copy(update={"judgments": [merged[key] for key in sorted(merged)]})


def accepted_judgments(judgment_set, include_source_checked=False):
    accepted = {}
    for item in judgment_set.judgments:
        if item.review_status == "approved" or (include_source_checked and item.review_status == "source_checked"):
            accepted.setdefault(item.question_id, {})[item.chunk_id] = item.relevance
    return accepted


def review_markdown(candidates, judgments):
    lookup = {(item.question_id, item.chunk_id): item for item in judgments.judgments}
    lines = ["# Pooled question/chunk review", "",
             "Edit judgments.json; use relevance 0/1, a rationale, and source quotations for positives. "
             "Only a human may set approved with their actual name. Map supporting evidence_ids to required facts. "
             "Unjudged is not a negative. Review top-five candidates first. Retrieval origins are kept in pool.json, "
             "not displayed here to reduce ranking bias.", ""]
    for candidate in sorted(candidates, key=lambda row: (not row["top5_priority"], row["question_id"], row["chunk_id"])):
        item = lookup[(candidate["question_id"], candidate["chunk_id"])]
        lines.extend([f"## {item.question_id} / {item.chunk_id}", "", candidate["question"], "",
                      f"Reference answer: {candidate['expected_answer']}", "",
                      "Supporting evidence unit IDs: " + ", ".join(item.evidence_ids or ["Assign from pool.json questions[].evidence after review"]), "",
                      f"Source: {candidate['source_file']}; PDF pages: {candidate['page_numbers']}", "",
                      f"Status: {item.review_status}; relevance: {item.relevance}; top-five priority: {candidate['top5_priority']}", "",
                      candidate["text"], ""])
    return "\n".join(lines) + "\n"
