from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from importlib.metadata import version
from typing import Protocol

import numpy as np

from src.chunking.semantic.embeddings import DEFAULT_EMBEDDING_MODEL
from src.indexing.configuration import (
    BGE_QUERY_INSTRUCTION,
    DEFAULT_BGE_REVISION,
    EmbeddingConfiguration,
)


class DenseEmbedder(Protocol):
    """Embedding interface used by indexing and retrieval."""

    model_name: str
    configuration: EmbeddingConfiguration

    def count_tokens(self, text: str, *, query: bool = False) -> int:
        """Count actual model input tokens, including instructions and special tokens."""

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
        revision: str | None = None,
        query_instruction: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        resolved_revision = revision or (
            DEFAULT_BGE_REVISION if model_name == DEFAULT_EMBEDDING_MODEL else None
        )
        self.model = SentenceTransformer(model_name, device=device, revision=resolved_revision)
        dimension_method = getattr(
            self.model,
            "get_embedding_dimension",
            self.model.get_sentence_embedding_dimension,
        )
        resolved_dimension = dimension_method()
        if not resolved_dimension:
            raise ValueError(f"embedding dimension is unavailable for {model_name}")
        self._dimension = int(resolved_dimension)
        self.tokenizer = self.model.tokenizer
        self.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
        actual_revision = getattr(self.model[0].auto_model.config, "_commit_hash", None)
        if not actual_revision:
            raise ValueError("embedding model must resolve to an immutable Hugging Face revision")
        tokenizer_json = (
            self.tokenizer.backend_tokenizer.to_str()
            if hasattr(self.tokenizer, "backend_tokenizer")
            else json.dumps(self.tokenizer.get_vocab(), sort_keys=True)
        )
        pooling = self.model[1].get_config_dict()
        self.configuration = EmbeddingConfiguration(
            model_name=model_name,
            model_revision=actual_revision,
            tokenizer_sha256=hashlib.sha256(tokenizer_json.encode("utf-8")).hexdigest(),
            dimension=self.dimension,
            max_sequence_length=int(self.model.max_seq_length),
            query_instruction=(
                query_instruction if query_instruction is not None
                else BGE_QUERY_INSTRUCTION if model_name == DEFAULT_EMBEDDING_MODEL else ""
            ),
            pooling=json.dumps(pooling, sort_keys=True),
            backend=f"sentence-transformers/{version('sentence-transformers')};transformers/{version('transformers')}",
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        method = getattr(self.model, "encode_document", self.model.encode)
        self._check_lengths(texts)
        return self._encode(method, list(texts), self.configuration.document_instruction)

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query cannot be empty")
        method = getattr(self.model, "encode_query", self.model.encode)
        self._check_lengths([query], query=True)
        return self._encode(method, [query], self.configuration.query_instruction)[0]

    def count_tokens(self, text: str, *, query: bool = False) -> int:
        instruction = (
            self.configuration.query_instruction if query
            else self.configuration.document_instruction
        )
        return len(self.tokenizer(instruction + text, truncation=False, add_special_tokens=True)["input_ids"])

    def _check_lengths(self, texts: Sequence[str], *, query: bool = False) -> None:
        for text in texts:
            count = self.count_tokens(text, query=query)
            if count > self.configuration.max_sequence_length:
                raise ValueError(f"embedding input has {count} tokens; maximum is {self.configuration.max_sequence_length}; silent truncation is disabled")

    def _encode(self, method: object, texts: list[str], instruction: str) -> np.ndarray:
        vectors = method(  # type: ignore[operator]
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            prompt=instruction,
        )
        result = np.asarray(vectors, dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != self.dimension:
            raise ValueError(
                f"embedding model returned {result.shape}; expected (*, {self.dimension})"
            )
        return result
