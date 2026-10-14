from __future__ import annotations

import hashlib
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.chunking.models import Chunk


_POINT_NAMESPACE = uuid.UUID("41fc37ef-e975-4e5c-a3f4-a15708e44d57")


class IndexRecord(BaseModel):
    """Validated structure-aware chunk prepared for vector indexing."""

    model_config = ConfigDict(extra="forbid")

    point_id: str
    chunk_id: str
    document_id: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_count: int = Field(ge=0)
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    strategy: str
    source_file: str | None = None
    source_sha256: str | None = None
    validation_status: str | None = None
    content_sha256: str
    embedding_token_count: int | None = Field(default=None, ge=0)
    parent_chunk_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "IndexRecord":
        if chunk.strategy != "structure-aware":
            raise ValueError(
                f"expected a structure-aware chunk, received {chunk.strategy!r}"
            )
        text = chunk.text.strip()
        if not text:
            raise ValueError(f"chunk {chunk.chunk_id!r} has empty text")
        source_sha256 = _optional_string(chunk.metadata.get("source_sha256"))
        if source_sha256 and source_sha256 != chunk.document_id:
            raise ValueError(
                f"chunk {chunk.chunk_id!r} document fingerprint does not match metadata"
            )
        return cls(
            point_id=point_id_for_chunk(chunk.chunk_id),
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            text=text,
            token_count=chunk.token_count or 0,
            section_path=chunk.section_path,
            page_numbers=chunk.page_numbers,
            source_refs=chunk.source_refs,
            strategy=chunk.strategy,
            source_file=_optional_string(chunk.metadata.get("source_file")),
            source_sha256=source_sha256,
            validation_status=_optional_string(
                chunk.metadata.get("validation_status")
            ),
            content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            metadata=chunk.metadata,
        )

    def to_payload(self, embedding_model: str) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "token_count": self.token_count,
            "section_path": self.section_path,
            "page_numbers": self.page_numbers,
            "source_refs": self.source_refs,
            "strategy": self.strategy,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "validation_status": self.validation_status,
            "content_sha256": self.content_sha256,
            "embedding_token_count": self.embedding_token_count,
            "parent_chunk_id": self.parent_chunk_id,
            "embedding_model": embedding_model,
            "metadata": self.metadata,
        }


class RetrievalHit(BaseModel):
    """One ranked Qdrant dense-retrieval result."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    score: float
    point_id: str
    chunk_id: str
    document_id: str
    text: str
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    source_file: str | None = None
    validation_status: str | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def point_id_for_chunk(chunk_id: str) -> str:
    """Use the same stable point identity for source and fallback chunks."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))
