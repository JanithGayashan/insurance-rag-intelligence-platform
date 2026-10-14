from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdf_pages: list[int] = Field(min_length=1)
    quotes: list[str] = Field(min_length=1)
    section: str = ""

    @model_validator(mode="after")
    def validate_evidence(self):
        if any(page < 1 for page in self.pdf_pages):
            raise ValueError("PDF page numbers are one-based and positive")
        if any(not quote.strip() for quote in self.quotes):
            raise ValueError("evidence quotes cannot be empty")
        return self


class EvidencePassage(EvidenceLocation):
    evidence_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_sha256: str = Field(min_length=1)
    alternatives: list[EvidenceLocation] = Field(default_factory=list)


class InsuranceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    evidence: list[EvidencePassage] = Field(min_length=1)
    query_filters: dict[str, str | int | bool] = Field(default_factory=dict)
    review_status: Literal["draft", "source_checked", "approved", "excluded"] = "draft"
    reviewer: str | None = None
    review_notes: str = ""

    @model_validator(mode="after")
    def validate_review(self):
        if not self.question.strip() or not self.expected_answer.strip():
            raise ValueError("question and reference answer cannot be blank")
        if self.review_status == "approved" and not (self.reviewer or "").strip():
            raise ValueError("approved labels require a named human reviewer")
        if len({item.evidence_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate evidence IDs in a question")
        return self


class QuestionDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    version: int = Field(gt=0)
    description: str
    label_scope: str
    questions: list[InsuranceQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ids(self):
        ids = [question.question_id for question in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate question IDs")
        return self
