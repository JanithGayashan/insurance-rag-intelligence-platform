from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticUnit:
    """Smallest structure-preserving unit considered for semantic grouping."""

    text: str
    section_path: tuple[str, ...]
    page_numbers: tuple[int, ...]
    source_refs: tuple[str, ...]
    element_type: str
    fallback_token_split: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class SemanticGroup:
    """Units selected for one chunk and the reason its boundary was created."""

    units: tuple[SemanticUnit, ...]
    boundary_reason: str
    boundary_distance: float | None = None
    breakpoint_threshold: float | None = None
