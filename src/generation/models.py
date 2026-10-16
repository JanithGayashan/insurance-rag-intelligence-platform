from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextSource(StrictModel):
    source_id: str
    chunk_id: str
    document_id: str
    source_file: str
    page_numbers: list[int]
    section_path: list[str]
    source_refs: list[str]
    text: str
    retrieval_rank: int
    retrieval_score: float
    text_sha256: str


class PreparedContext(StrictModel):
    question: str
    sources: list[ContextSource]
    token_count: int
    token_budget: int
    skipped: list[dict[str, str]]
    context_sha256: str


class QuotedEvidence(StrictModel):
    source_id: str = Field(description="An existing source identifier such as S1.")
    quote: str = Field(min_length=12, description="A verbatim supporting excerpt from that source. Preserve complete conditions and amounts.")

    @field_validator("quote")
    @classmethod
    def substantive_quote(cls, value: str) -> str:
        if len(" ".join(value.split())) < 12:
            raise ValueError("supporting quotations require at least 12 non-padding characters")
        return value


class GroundedClaim(StrictModel):
    text: str = Field(min_length=1, description="One supported answer statement, without citation markers.")
    evidence: list[QuotedEvidence] = Field(min_length=1)


class AnswerDraft(StrictModel):
    """The provider's strict JSON schema, independently checked against context."""

    status: Literal["answered", "insufficient_evidence"]
    claims: list[GroundedClaim]

    @model_validator(mode="after")
    def consistent_status(self):
        if (self.status == "answered") != bool(self.claims):
            raise ValueError("answered requires claims; insufficient_evidence requires no claims")
        return self


class TokenUsage(StrictModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    estimated_cost_usd: float | None = None
    pricing_as_of: str | None = None


class AnswerResult(StrictModel):
    run_id: str
    created_at: str
    question: str
    status: Literal["answered", "insufficient_evidence", "invalid_output", "incomplete", "refused", "prepared", "provider_error"]
    answer: str
    claims: list[GroundedClaim] = Field(default_factory=list)
    citations: list[ContextSource] = Field(default_factory=list)
    context: PreparedContext
    configuration: dict[str, Any]
    retrieval_method: str
    filters: dict[str, Any]
    diagnostics: dict[str, Any]
    usage: TokenUsage = Field(default_factory=TokenUsage)
    provider_response_id: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
