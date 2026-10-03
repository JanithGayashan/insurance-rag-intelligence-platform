from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """
    Represents one chunk produced from a source document.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str

    chunk_index: int = Field(ge=0)

    text: str

    page_numbers: list[int] = Field(
        default_factory=list
    )

    section_path: list[str] = Field(
        default_factory=list
    )

    source_refs: list[str] = Field(
        default_factory=list
    )

    strategy: str

    token_count: int | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )