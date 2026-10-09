from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class TextEmbedder(Protocol):
    """Small interface shared by the custom and LlamaIndex implementations."""

    model_name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one normalized vector per input text."""


class SentenceTransformerEmbedder:
    """Local Sentence Transformers embedder loaded once and reused."""

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

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dimension = self.model.get_sentence_embedding_dimension() or 0
            return np.empty((0, dimension), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def cosine_similarities(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Calculate row-wise cosine similarity, including non-normalized inputs."""

    if left.shape != right.shape:
        raise ValueError("left and right embedding arrays must have the same shape")
    if left.size == 0:
        return np.empty(0, dtype=np.float32)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    denominator = np.where(denominator == 0, 1.0, denominator)
    return np.sum(left * right, axis=1) / denominator
