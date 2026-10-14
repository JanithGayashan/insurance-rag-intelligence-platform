from __future__ import annotations

from collections.abc import Mapping

from src.indexing.embeddings import DenseEmbedder
from src.indexing.models import RetrievalHit
from src.indexing.qdrant_store import QdrantVectorStore


class DenseRetriever:
    """Embed a question and retrieve ranked structure-aware chunks."""

    def __init__(
        self,
        embedder: DenseEmbedder,
        vector_store: QdrantVectorStore,
    ) -> None:
        if embedder.model_name != vector_store.embedding_model:
            raise ValueError("query and indexed-document embedding models must match")
        if embedder.dimension != vector_store.vector_dimension:
            raise ValueError("query and Qdrant vector dimensions must match")
        vector_store.check_configuration(embedder.configuration)
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: Mapping[str, str | int | bool] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalHit]:
        if not question.strip():
            raise ValueError("question cannot be empty")
        vector = self.embedder.embed_query(question)
        return self.vector_store.search(
            vector,
            limit=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )
