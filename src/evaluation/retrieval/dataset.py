from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from src.indexing.configuration import json_sha256
from src.evaluation.retrieval.models import InsuranceQuestion, QuestionDataset


def normalized_text(text: str) -> str:
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", text)).strip()


def source_key(source_file: str) -> str:
    return source_file.replace("\\", "/")


def load_dataset(path: Path) -> QuestionDataset:
    return QuestionDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def select_questions(dataset: QuestionDataset, include_source_checked: bool = False) -> list[InsuranceQuestion]:
    statuses = {"approved", "source_checked"} if include_source_checked else {"approved"}
    selected = [item for item in dataset.questions if item.review_status in statuses]
    if not selected:
        raise ValueError("no approved questions; review the dataset or use --include-source-checked for a provisional baseline")
    return selected


def corpus_fingerprint(points: Mapping[str, dict[str, Any]]) -> str:
    rows = []
    for point_id, value in points.items():
        payload = value["payload"]
        rows.append({"point_id": point_id, "chunk_id": payload["chunk_id"],
                     "document_id": payload["document_id"], "source_file": source_key(payload.get("source_file") or ""),
                     "text_sha256": hashlib.sha256(payload["text"].encode("utf-8")).hexdigest(),
                     "record_sha256": payload.get("record_sha256")})
    return json_sha256(sorted(rows, key=lambda item: item["point_id"]))


def bind_evidence(questions: list[InsuranceQuestion], points: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Bind source quotes to chunks before retrieval; never label from rankings."""
    payloads = [value["payload"] for value in points.values()]
    if len({payload["chunk_id"] for payload in payloads}) != len(payloads):
        raise ValueError("index contains duplicate chunk IDs")
    normalized = [(payload, normalized_text(payload["text"])) for payload in payloads]
    result = {}
    for question in questions:
        labels = {}
        for passage in question.evidence:
            candidates = [(payload, text) for payload, text in normalized
                          if source_key(payload.get("source_file") or "") == source_key(passage.source_file)
                          and payload.get("source_sha256") == passage.source_sha256]
            if not candidates:
                raise ValueError(f"{question.question_id}: source missing or fingerprint changed: {passage.source_file}")
            matches = set()
            for location in [passage, *passage.alternatives]:
                quotes = [normalized_text(quote) for quote in location.quotes]
                if any(not quote for quote in quotes):
                    raise ValueError("evidence must contain searchable text")
                location_matches = {payload["chunk_id"] for payload, text in candidates
                                    if set(payload.get("page_numbers", [])) & set(location.pdf_pages)
                                    and all(quote in text for quote in quotes)}
                if not location_matches:
                    raise ValueError(f"{question.question_id}: no chunk contains complete evidence {passage.evidence_id}; review labels after rechunking")
                matches.update(location_matches)
            relevant = sorted(matches)
            if not relevant:
                raise ValueError(f"{question.question_id}: no chunk contains complete evidence {passage.evidence_id}; review labels after rechunking")
            labels[passage.evidence_id] = relevant
        result[question.question_id] = labels
    return result


def verify_pdf_sources(questions: list[InsuranceQuestion], raw_root: Path) -> dict[str, Any]:
    """Verify fingerprints and cited passages on original PDF pages."""
    from pypdf import PdfReader

    readers = {}
    checked = 0
    root = Path(raw_root).resolve()
    for question in questions:
        for evidence in question.evidence:
            relative = PurePosixPath(source_key(evidence.source_file))
            path = root.joinpath(*relative.parts).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"source PDF missing or outside corpus: {evidence.source_file}")
            if path not in readers:
                fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
                if fingerprint != evidence.source_sha256:
                    raise ValueError(f"source PDF fingerprint changed: {evidence.source_file}")
                readers[path] = (fingerprint, PdfReader(path), {})
            fingerprint, reader, page_cache = readers[path]
            if fingerprint != evidence.source_sha256:
                raise ValueError(f"source PDF fingerprint changed: {evidence.source_file}")
            for location in [evidence, *evidence.alternatives]:
                texts = []
                for page in location.pdf_pages:
                    if page > len(reader.pages):
                        raise ValueError(f"invalid PDF page for {question.question_id}: {page}")
                    if page not in page_cache:
                        page_cache[page] = normalized_text(reader.pages[page - 1].extract_text() or "")
                    texts.append(page_cache[page])
                text = " ".join(texts)
                if any(normalized_text(quote) not in text for quote in location.quotes):
                    raise ValueError(f"{question.question_id}: PDF text does not verify evidence; inspect original page visually")
                checked += 1
    return {"status": "PASS", "pdf_documents_checked": len(readers), "evidence_passages_checked": checked}
