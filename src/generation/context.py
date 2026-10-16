from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence

import tiktoken

from src.generation.configuration import GenerationConfiguration
from src.generation.models import ContextSource, PreparedContext
from src.indexing.models import RetrievalHit


def context_text(sources: Sequence[ContextSource]) -> str:
    """Serialize only evidence and source headers, never internal retrieval metadata."""
    return json.dumps([
        {"source_id": item.source_id, "source_file": item.source_file,
         "document_id": item.document_id, "pdf_pages": item.page_numbers,
         "section_path": item.section_path, "text": item.text}
        for item in sources
    ], ensure_ascii=False, indent=2)


def prepare_context(question: str, hits: Sequence[RetrievalHit],
                    configuration: GenerationConfiguration) -> PreparedContext:
    encoding = tiktoken.get_encoding(configuration.tokenizer_encoding)
    question = question.strip()
    if not question:
        raise ValueError("question cannot be blank")
    if len(encoding.encode(question, disallowed_special=())) > configuration.max_question_tokens:
        raise ValueError("question exceeds the configured token limit")
    sources, skipped = [], []
    seen_ids, seen_text = set(), set()
    for hit in hits:
        reason = None
        normalized = " ".join(hit.text.split())
        identity = (hit.document_id, normalized)
        if hit.chunk_id in seen_ids or identity in seen_text:
            reason = "duplicate"
        elif not normalized:
            reason = "empty_text"
        elif not hit.source_file or not hit.document_id or not hit.page_numbers or any(page < 1 for page in hit.page_numbers):
            reason = "missing_provenance"
        elif not math.isfinite(hit.score):
            reason = "nonfinite_score"
        if reason:
            skipped.append({"chunk_id": hit.chunk_id, "reason": reason})
            continue
        seen_ids.add(hit.chunk_id)
        seen_text.add(identity)
        source = ContextSource(
            source_id=f"S{len(sources) + 1}", chunk_id=hit.chunk_id,
            document_id=hit.document_id, source_file=hit.source_file,
            page_numbers=sorted(set(hit.page_numbers)), section_path=hit.section_path,
            source_refs=hit.source_refs, text=hit.text,
            retrieval_rank=hit.rank, retrieval_score=hit.score,
            text_sha256=hashlib.sha256(hit.text.encode("utf-8")).hexdigest(),
        )
        # Whole chunks preserve lists, tables, exclusions and conditions. Never
        # silently cut a passage to make it fit, or merge different policies.
        if len(encoding.encode(context_text([*sources, source]), disallowed_special=())) > configuration.context_token_budget:
            skipped.append({"chunk_id": hit.chunk_id, "reason": "context_budget"})
            continue
        sources.append(source)
    text = context_text(sources)
    return PreparedContext(
        question=question, sources=sources, skipped=skipped,
        token_count=len(encoding.encode(text, disallowed_special=())),
        token_budget=configuration.context_token_budget,
        context_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
