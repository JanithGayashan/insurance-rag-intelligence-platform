from __future__ import annotations

from contextlib import ExitStack
from pathlib import PurePosixPath

from src.api.configuration import APIConfiguration
from src.api.models import ChatRequest, ChatResponse, Citation, PolicyDocument
from src.generation.configuration import GenerationConfiguration, require_api_key
from src.generation.pipeline import save_answer_artifacts
from src.configuration.dataset import get_dataset_configuration


class BackendGenerationError(RuntimeError):
    def __init__(self, request_id: str):
        self.request_id = request_id


class RAGRuntime:
    """Owned by one worker thread: load once, reuse for requests, close on shutdown."""

    def __init__(self, configuration: APIConfiguration):
        self.configuration = configuration
        self.dataset = get_dataset_configuration(configuration.dataset_id)
        self.stack = ExitStack()
        self.service = None
        self.documents: list[PolicyDocument] = []

    def start(self):
        from src.generation.cli import open_answer_service
        from src.generation.llamaindex import LlamaIndexAnswerGenerator

        generation = GenerationConfiguration.from_environment()
        require_api_key()
        try:
            self.service = self.stack.enter_context(open_answer_service(
                generation, qdrant_path=self.configuration.qdrant_path,
                collection=self.configuration.collection,
                dataset=self.dataset,
            ))
            generator = LlamaIndexAnswerGenerator(generation)
            self.service.generator = generator
            self.stack.callback(generator.close)
            grouped = {}
            for item in self.service.retrieval_suite.snapshot.values():
                payload = item["payload"]
                if not payload.get("source_file"):
                    continue
                identifier = payload["document_id"]
                source = payload["source_file"].replace("\\", "/")
                if identifier not in grouped:
                    grouped[identifier] = PolicyDocument(document_id=identifier, source_file=source,
                        display_name=PurePosixPath(source).name, chunk_count=0,
                        **self.dataset.source_metadata(source))
                grouped[identifier].chunk_count += 1
            self.documents = sorted(grouped.values(), key=lambda item: item.source_file)
        except BaseException:
            self.stack.close()
            raise

    def answer(self, payload: ChatRequest) -> ChatResponse:
        if self.service is None:
            raise RuntimeError("RAG runtime is not initialized")
        filters = {"document_id": payload.document_id} if payload.document_id else None
        result = self.service.answer(payload.message, filters=filters)
        result.diagnostics["dataset_id"] = self.dataset.dataset_id
        if self.configuration.save_artifacts:
            save_answer_artifacts(result, self.configuration.output_directory)
        if result.status not in {"answered", "insufficient_evidence", "refused"}:
            raise BackendGenerationError(result.run_id)
        quotations = {}
        for claim in result.claims:
            for evidence in claim.evidence:
                quotations.setdefault(evidence.source_id, []).append(evidence.quote)
        return ChatResponse(
            request_id=result.run_id, status=result.status, answer=result.answer,
            citations=[Citation(source_id=source.source_id, document_id=source.document_id,
                source_file=source.source_file.replace("\\", "/"), page_numbers=source.page_numbers,
                quotes=list(dict.fromkeys(quotations.get(source.source_id, [])))) for source in result.citations],
        )

    def close(self):
        self.stack.close()
        self.service = None
