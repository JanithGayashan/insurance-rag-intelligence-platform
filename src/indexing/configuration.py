from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingConfiguration(BaseModel):
    """Immutable identity of the complete embedding transformation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str
    model_revision: str = Field(min_length=1)
    tokenizer_sha256: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    max_sequence_length: int = Field(gt=2)
    document_instruction: str = ""
    query_instruction: str = ""
    pooling: str = "cls"
    normalize: bool = True
    precision: str = "float32"
    backend: str = "sentence-transformers"

    @property
    def fingerprint(self) -> str:
        return json_sha256(self.model_dump())


def json_sha256(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
