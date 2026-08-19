from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ParsedElement:
    type: str
    text: str
    page_number: int | None = None
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedTable:
    rows: list[list[str]]
    page_number: int | None = None
    bbox: list[float] | None = None

    markdown: str | None = None
    html: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedPage:
    page_number: int
    text: str = ""
    markdown: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    parser_name: str
    source_file: str
    success: bool

    text: str = ""
    markdown: str | None = None

    pages: list[ParsedPage] = field(default_factory=list)
    elements: list[ParsedElement] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    processing_time_seconds: float = 0.0

    warnings: list[str] = field(default_factory=list)

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)