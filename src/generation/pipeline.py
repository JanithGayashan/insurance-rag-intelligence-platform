from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import uuid

import tiktoken
from pydantic import ValidationError

from src.generation.configuration import GenerationConfiguration
from src.generation.context import prepare_context
from src.indexing.configuration import json_sha256
from src.generation.llamaindex import GenerationError, LlamaIndexAnswerGenerator, response_format
from src.generation.models import AnswerDraft, AnswerResult, PreparedContext
from src.generation.prompts import PROMPT_VERSION, build_messages


INSUFFICIENT_ANSWER = "I don't have enough evidence in the retrieved policy excerpts to answer this reliably. Please specify the policy or provide the relevant policy document."
INVALID_ANSWER = "I couldn't produce an answer with valid supporting citations from the retrieved policy excerpts."


def validate_draft(draft: AnswerDraft, context: PreparedContext) -> list[str]:
    """Check citation identity and quotation integrity, not semantic entailment."""
    sources = {item.source_id: item for item in context.sources}
    errors = []
    for index, claim in enumerate(draft.claims, 1):
        if not claim.text.strip() or re.search(r"\[S\d+\]", claim.text):
            errors.append(f"claim {index}: empty text or embedded citation marker")
        for evidence in claim.evidence:
            if evidence.source_id not in sources:
                errors.append(f"claim {index}: unknown source identifier")
            elif " ".join(evidence.quote.split()) not in " ".join(sources[evidence.source_id].text.split()):
                errors.append(f"claim {index}: supporting quotation is not in the cited source")
    return errors


class RAGAnswerService:
    def __init__(self, retrieval_suite, configuration: GenerationConfiguration | None = None,
                 *, generator=None, method: str = "hybrid_reranked", top_k: int = 5,
                 dataset_id: str | None = None):
        if method not in retrieval_suite.METHODS or not 0 < top_k <= 20:
            raise ValueError("unsupported retrieval method or top_k outside 1..20")
        self.retrieval_suite = retrieval_suite
        self.configuration = configuration or GenerationConfiguration()
        self.generator = generator
        self.method, self.top_k = method, top_k
        self.dataset_id = dataset_id

    def answer(self, question: str, *, filters=None, dry_run: bool = False) -> AnswerResult:
        # Reject oversized/blank questions before embedding or provider work.
        prepare_context(question, [], self.configuration)
        question = question.strip()
        started = time.perf_counter()
        hits_by_method, diagnostics = self.retrieval_suite.retrieve_all_with_diagnostics(
            question, top_k=self.top_k, filters=filters or None, methods=(self.method,),
        )
        hits = hits_by_method[self.method]
        context = prepare_context(question, hits, self.configuration)
        messages = build_messages(context)
        encoding = tiktoken.get_encoding(self.configuration.tokenizer_encoding)
        estimated_input = sum(len(encoding.encode(message.content or "", disallowed_special=())) for message in messages)
        estimated_input += len(encoding.encode(json.dumps(response_format()), disallowed_special=())) + 32
        diagnostics = dict(diagnostics, retrieved_hits=[hit.model_dump(mode="json") for hit in hits],
                           prompt_version=PROMPT_VERSION, input_tokens_estimate=estimated_input,
                           token_estimate_note="Includes schema and estimated message overhead; provider usage is authoritative.")
        if self.dataset_id is not None:
            diagnostics["dataset_id"] = self.dataset_id
        snapshot = getattr(self.retrieval_suite, "snapshot", None)
        if snapshot is not None:
            diagnostics["index_snapshot_sha256"] = json_sha256([
                {"point_id": key, "payload": item["payload"]} for key, item in sorted(snapshot.items())])
        dense = getattr(self.retrieval_suite, "dense", None)
        if dense is not None:
            diagnostics["collection"] = dense.vector_store.collection_name
            diagnostics["embedding_configuration"] = dense.embedder.configuration.model_dump()
        result = AnswerResult(
            run_id=uuid.uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(),
            question=question, status="prepared" if dry_run else "insufficient_evidence",
            answer="Context prepared. No OpenAI API call was made." if dry_run else INSUFFICIENT_ANSWER,
            context=context, configuration=self.configuration.model_dump(), retrieval_method=self.method,
            filters=dict(filters or {}), diagnostics=diagnostics,
        )
        if dry_run or not context.sources:
            result.diagnostics.update(generation_seconds=0, api_calls=0, total_seconds=time.perf_counter() - started)
            return result
        generator = self.generator or LlamaIndexAnswerGenerator(self.configuration)
        generation_started = time.perf_counter()
        try:
            output = generator.generate(context)
        except GenerationError as error:
            result.status, result.answer = "provider_error", str(error)
            result.diagnostics.update(generation_seconds=time.perf_counter() - generation_started,
                                      api_calls=1, total_seconds=time.perf_counter() - started)
            return result
        result.usage, result.provider_response_id = output.usage, output.response_id
        result.diagnostics.update(generation_seconds=time.perf_counter() - generation_started, api_calls=1,
                                  provider_status=output.status, provider_model=output.model,
                                  model_output=output.text,
                                  total_seconds=time.perf_counter() - started)
        if output.refused:
            result.status, result.answer = "refused", "The model declined to answer this question."
            return result
        if output.status != "completed":
            result.status, result.answer = "incomplete", "The model did not finish its answer. No partial answer was published."
            return result
        try:
            draft = AnswerDraft.model_validate_json(output.text)
        except ValidationError:
            result.status, result.answer = "invalid_output", INVALID_ANSWER
            result.validation_errors = ["The model output did not match the required answer schema."]
            return result
        errors = validate_draft(draft, context)
        if errors:
            result.status, result.answer, result.validation_errors = "invalid_output", INVALID_ANSWER, errors
            return result
        if draft.status == "insufficient_evidence":
            return result
        result.status, result.claims = "answered", draft.claims
        result.answer = "\n\n".join(claim.text.strip() + " " + " ".join(
            f"[{source_id}]" for source_id in dict.fromkeys(e.source_id for e in claim.evidence)) for claim in draft.claims)
        cited = {evidence.source_id for claim in draft.claims for evidence in claim.evidence}
        result.citations = [source for source in context.sources if source.source_id in cited]
        return result


def save_answer_artifacts(result: AnswerResult, output_root: Path) -> Path:
    directory = Path(output_root) / result.run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "answer.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    sources = "\n".join(f"[{item.source_id}] {item.source_file}, PDF pages {', '.join(map(str, item.page_numbers))}" for item in result.citations)
    (directory / "answer.txt").write_text(f"Question: {result.question}\nStatus: {result.status}\n\n{result.answer}\n\n{sources}\n", encoding="utf-8")
    context_parts = [f"Question: {result.question}\nContext tokens: {result.context.token_count}/{result.context.token_budget}\n"]
    for source in result.context.sources:
        context_parts.append(f"[{source.source_id}] {source.source_file}\nPDF pages: {source.page_numbers}\nSection: {' / '.join(source.section_path)}\nChunk: {source.chunk_id}\n\n{source.text}")
    (directory / "context.txt").write_text("\n\n".join(context_parts) + "\n", encoding="utf-8")
    prompt = {"prompt_version": PROMPT_VERSION,
              "messages": [{"role": message.role.value, "content": message.content} for message in build_messages(result.context)],
              "text": response_format(), "configuration": result.configuration,
              "context_sha256": result.context.context_sha256}
    (directory / "prompt.json").write_text(json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8")
    return directory
