from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.chunking.semantic.embeddings import DEFAULT_EMBEDDING_MODEL
from src.indexing.embeddings import DenseEmbeddingService
from src.indexing.pipeline import DEFAULT_COLLECTION, LOCAL_QDRANT_PATH
from src.indexing.qdrant_store import QdrantVectorStore
from src.retrieval.dense import DenseRetriever


def retrieve_question(
    question: str,
    top_k: int = 5,
    document_id: str | None = None,
    source_file: str | None = None,
    validation_status: str | None = None,
    local_qdrant_path: Path = LOCAL_QDRANT_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    revision: str | None = None,
    method: str = "dense",
    candidate_k: int = 20,
) -> list[dict[str, object]]:
    """Query the persisted dense index and return serializable results."""

    load_dotenv()
    embedder = DenseEmbeddingService(embedding_model, revision=revision)
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    if qdrant_url:
        store = QdrantVectorStore.remote(
            qdrant_url,
            collection_name,
            embedder.model_name,
            embedder.dimension,
            api_key=os.getenv("QDRANT_API_KEY") or None,
        )
    else:
        store = QdrantVectorStore.local(
            local_qdrant_path,
            collection_name,
            embedder.model_name,
            embedder.dimension,
        )
    filters = {
        key: value
        for key, value in {
            "document_id": document_id,
            "source_file": source_file,
            "validation_status": validation_status,
        }.items()
        if value is not None
    }
    try:
        dense = DenseRetriever(embedder, store)
        if method == "dense":
            hits = dense.retrieve(question, top_k=top_k, filters=filters or None)
        else:
            from src.retrieval.hybrid import LlamaIndexRetrievalSuite
            hits = LlamaIndexRetrievalSuite(dense, candidate_k=candidate_k).retrieve(
                question, top_k=top_k, filters=filters or None, method=method,
            )
        return [hit.model_dump(mode="json") for hit in hits]
    finally:
        store.close()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve structure-aware insurance chunks from Qdrant."
    )
    parser.add_argument("question", help="Insurance question to retrieve for")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--document-id")
    parser.add_argument("--source-file")
    parser.add_argument("--validation-status")
    parser.add_argument("--qdrant-path", type=Path, default=LOCAL_QDRANT_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--revision", help="Must match the revision used to build the index.")
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid"], default="dense")
    parser.add_argument("--candidate-k", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    results = retrieve_question(
        arguments.question,
        top_k=arguments.top_k,
        document_id=arguments.document_id,
        source_file=arguments.source_file,
        validation_status=arguments.validation_status,
        local_qdrant_path=arguments.qdrant_path,
        collection_name=arguments.collection,
        embedding_model=arguments.embedding_model,
        revision=arguments.revision,
        method=arguments.method,
        candidate_k=arguments.candidate_k,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
