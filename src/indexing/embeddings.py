from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from src.chunking.semantic.embeddings import DEFAULT_EMBEDDING_MODEL


class DenseEmbedder(Protocol):
    """Embedding interface used by indexing and retrieval."""

    model_name: str

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed passages and return normalized vectors."""

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one query and return a normalized vector."""


class DenseEmbeddingService:
    """Sentence Transformers service with query/document task methods."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        dimension_method = getattr(
            self.model,
            "get_embedding_dimension",
            self.model.get_sentence_embedding_dimension,
        )
        resolved_dimension = dimension_method()
        if not resolved_dimension:
            raise ValueError(f"embedding dimension is unavailable for {model_name}")
        self._dimension = int(resolved_dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        method = getattr(self.model, "encode_document", self.model.encode)
        return self._encode(method, list(texts))

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query cannot be empty")
        method = getattr(self.model, "encode_query", self.model.encode)
        return self._encode(method, [query])[0]

    def _encode(self, method: object, texts: list[str]) -> np.ndarray:
        vectors = method(  # type: ignore[operator]
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = np.asarray(vectors, dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != self.dimension:
            raise ValueError(
                f"embedding model returned {result.shape}; expected (*, {self.dimension})"
            )
        return result
