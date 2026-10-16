from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys

from src.generation.configuration import PROJECT_ROOT, GenerationConfiguration, require_api_key
from src.generation.pipeline import RAGAnswerService, save_answer_artifacts
from src.configuration.dataset import DatasetConfiguration, get_dataset_configuration


@contextmanager
def open_answer_service(configuration: GenerationConfiguration, *, method="hybrid_reranked", top_k=5,
                        candidate_k=20, rerank_candidates=20,
                        qdrant_path: Path | None = None, collection: str | None = None,
                        dataset: DatasetConfiguration | None = None):
    """Reuse local BGE, Qdrant and native LlamaIndex retrieval; never re-index."""
    from src.indexing.embeddings import DenseEmbeddingService
    from src.indexing.qdrant_store import QdrantVectorStore
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import LlamaIndexRetrievalSuite

    if not 0 < top_k <= 20 or not top_k <= rerank_candidates <= candidate_k <= 100:
        raise ValueError("require 1 <= top_k <= rerank_candidates <= candidate_k <= 100 and top_k <= 20")
    dataset = dataset or get_dataset_configuration()
    embedder = DenseEmbeddingService()
    name = collection or dataset.collection
    url = os.getenv("QDRANT_URL", "").strip()
    store = (QdrantVectorStore.remote(url, name, embedder.model_name, embedder.dimension,
                                      api_key=os.getenv("QDRANT_API_KEY") or None) if url else
             QdrantVectorStore.local(qdrant_path or dataset.qdrant_path, name,
                                     embedder.model_name, embedder.dimension))
    try:
        dense = DenseRetriever(embedder, store)
        reranker = None
        if method == "hybrid_reranked":
            from src.retrieval.reranking import LlamaIndexCrossEncoderReranker
            reranker = LlamaIndexCrossEncoderReranker()
        suite = LlamaIndexRetrievalSuite(dense, candidate_k=candidate_k,
                                        reranker=reranker, rerank_candidates=rerank_candidates)
        yield RAGAnswerService(suite, configuration, method=method, top_k=top_k, dataset_id=dataset.dataset_id)
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask an insurance question using LlamaIndex and grounded OpenAI generation.")
    parser.add_argument("question", help="Question about an indexed insurance policy")
    parser.add_argument("--source-file", help="Exact indexed source_file value; optional policy selection")
    parser.add_argument("--document-id")
    parser.add_argument("--validation-status", choices=["PASS", "REVIEW"])
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid", "hybrid_reranked"], default="hybrid_reranked")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--model", help="Defaults to OPENAI_MODEL or gpt-5-nano")
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--qdrant-path", type=Path)
    parser.add_argument("--collection")
    parser.add_argument("--output", type=Path, help="Defaults to the active dataset's answer output folder")
    parser.add_argument("--dry-run", action="store_true", help="Save retrieved context and prompt without calling OpenAI")
    args = parser.parse_args(argv)
    from src.generation.llamaindex import GenerationError
    try:
        configuration = GenerationConfiguration.from_environment(
            args.env_file, model=args.model, context_token_budget=args.context_tokens,
            max_output_tokens=args.max_output_tokens,
        )
        if not args.dry_run:
            require_api_key()
        filters = {key: value for key, value in {"source_file": args.source_file,
                   "document_id": args.document_id, "validation_status": args.validation_status}.items() if value is not None}
        with open_answer_service(configuration, method=args.method, top_k=args.top_k,
                                 candidate_k=args.candidate_k, rerank_candidates=args.rerank_candidates,
                                 qdrant_path=args.qdrant_path, collection=args.collection) as service:
            result = service.answer(args.question, filters=filters, dry_run=args.dry_run)
        directory = save_answer_artifacts(result, args.output or get_dataset_configuration().answer_output_directory)
    except (ValueError, GenerationError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(result.answer)
    if result.citations:
        print("\nSources:")
        for source in result.citations:
            print(f"[{source.source_id}] {source.source_file}, PDF pages {', '.join(map(str, source.page_numbers))}")
    print(f"\nStatus: {result.status}\nArtifacts: {directory}")
    if result.usage.estimated_cost_usd is not None:
        print(f"Estimated API cost: ${result.usage.estimated_cost_usd:.6f}")
    return 0 if result.status in {"answered", "insufficient_evidence", "prepared"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
