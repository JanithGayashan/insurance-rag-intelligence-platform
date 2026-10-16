from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    document_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be blank")
        return value


class PolicyDocument(BaseModel):
    document_id: str
    source_file: str
    display_name: str
    chunk_count: int
    insurer: str = ""
    business_line: str = "general"
    category: str = "uncategorized"
    category_key: str = "general/uncategorized"


class Citation(BaseModel):
    source_id: str
    document_id: str
    source_file: str
    page_numbers: list[int]
    quotes: list[str]


class ChatResponse(BaseModel):
    request_id: str
    status: Literal["answered", "insufficient_evidence", "refused"]
    answer: str
    citations: list[Citation]
