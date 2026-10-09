from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.chunking.models import Chunk
from src.chunking.parent_child.structured import StructuredParentChildChunker
from src.chunking.semantic.embeddings import (
    TextEmbedder,
    cosine_similarities,
)
from src.chunking.semantic.models import SemanticGroup, SemanticUnit
from src.chunking.structure_aware.chunker import StructureAwareChunker, _Block


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]*[A-Z0-9])")


class CustomSemanticChunker(StructureAwareChunker):
    """Split within Docling sections at embedding-distance breakpoints.

    Docling sections are hard boundaries. Semantic distance chooses boundaries
    inside a section, while minimum and maximum token limits prevent unusable
    fragments and chunks that exceed downstream model limits.
    """

    strategy = "semantic-custom-structure-constrained"

    def __init__(
        self,
        embedder: TextEmbedder,
        min_chunk_size: int = 80,
        target_chunk_size: int = 250,
        max_chunk_size: int = 450,
        breakpoint_percentile: float = 90.0,
        buffer_size: int = 1,
        encoding_name: str = "cl100k_base",
    ) -> None:
        _validate_configuration(
            min_chunk_size,
            target_chunk_size,
            max_chunk_size,
            breakpoint_percentile,
            buffer_size,
        )
        super().__init__(
            chunk_size=max_chunk_size,
            overlap=0,
            encoding_name=encoding_name,
        )
        self.embedder = embedder
        self.min_chunk_size = min_chunk_size
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.encoding_name = encoding_name
        self._structure_extractor = StructuredParentChildChunker(
            parent_size=max(max_chunk_size * 4, max_chunk_size + 1),
            child_size=max_chunk_size,
            encoding_name=encoding_name,
        )

    def chunk_file(
        self,
        document_path: Path,
        document_id: str | None = None,
        document_metadata: Mapping[str, Any] | None = None,
    ) -> list[Chunk]:
        document_path = Path(document_path)
        with document_path.open("r", encoding="utf-8") as document_file:
            document = json.load(document_file)
        return self.chunk_document(
            document,
            document_id or document_path.parent.name,
            document_metadata,
        )

    def chunk_document(
        self,
        document: Mapping[str, Any],
        document_id: str,
        document_metadata: Mapping[str, Any] | None = None,
    ) -> list[Chunk]:
        if not document_id.strip():
            raise ValueError("document_id cannot be empty")
        if not isinstance(document, Mapping):
            raise TypeError("document must be a mapping")
        blocks = self.extract_semantic_blocks(document)
        if not blocks:
            raise ValueError("Docling document contains no chunkable content")

        chunks: list[Chunk] = []
        for section_blocks in self._group_blocks_by_section(blocks):
            units = self._make_units(section_blocks)
            for group in self._semantic_groups(units):
                chunks.append(
                    self._build_semantic_chunk(
                        group,
                        document,
                        document_id,
                        len(chunks),
                        document_metadata,
                    )
                )
        return chunks

    def extract_semantic_blocks(self, document: Mapping[str, Any]) -> list[_Block]:
        """Use the insurance-aware heading inference shared with parent-child."""

        return list(self._structure_extractor._extract_section_blocks(document))

    @staticmethod
    def _group_blocks_by_section(blocks: Sequence[_Block]) -> list[list[_Block]]:
        groups: list[list[_Block]] = []
        pending: list[_Block] = []
        for block in blocks:
            if pending and pending[0].section_path != block.section_path:
                groups.append(pending)
                pending = []
            pending.append(block)
        if pending:
            groups.append(pending)
        return groups

    def _make_units(self, blocks: Sequence[_Block]) -> list[SemanticUnit]:
        units: list[SemanticUnit] = []
        for block in blocks:
            if block.element_type == "table":
                units.extend(self._units_from_structural_parts(block))
                continue
            sentences = [
                sentence.strip()
                for sentence in _SENTENCE_BOUNDARY.split(block.text)
                if sentence.strip()
            ]
            if not sentences:
                continue
            for sentence in sentences:
                sentence_block = _Block(
                    text=sentence,
                    page_numbers=block.page_numbers,
                    source_refs=block.source_refs,
                    section_path=block.section_path,
                    element_type=block.element_type,
                    metadata=block.metadata,
                )
                units.extend(self._units_from_structural_parts(sentence_block))
        return units

    def _units_from_structural_parts(self, block: _Block) -> list[SemanticUnit]:
        oversized = self._count_tokens(self._render_blocks([block])) > self.max_chunk_size
        parts = self._split_block(block) if oversized else [block]
        return [
            SemanticUnit(
                text=part.text,
                section_path=part.section_path,
                page_numbers=part.page_numbers,
                source_refs=part.source_refs,
                element_type=part.element_type,
                fallback_token_split=oversized,
                fallback_reason=("oversized_structural_unit" if oversized else None),
            )
            for part in parts
        ]

    def _semantic_groups(self, units: Sequence[SemanticUnit]) -> list[SemanticGroup]:
        if not units:
            return []
        if len(units) == 1:
            return [SemanticGroup(tuple(units), "section_end")]

        windows = []
        for index in range(len(units)):
            start = max(0, index - self.buffer_size)
            end = min(len(units), index + self.buffer_size + 1)
            windows.append(" ".join(unit.text for unit in units[start:end]))
        vectors = self.embedder.encode(windows)
        similarities = cosine_similarities(vectors[:-1], vectors[1:])
        distances = 1.0 - similarities
        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        semantic_breaks = {
            index + 1
            for index, distance in enumerate(distances)
            if float(distance) >= threshold
        }

        groups: list[SemanticGroup] = []
        pending: list[SemanticUnit] = []
        for index, unit in enumerate(units):
            candidate = [*pending, unit]
            if pending and self._group_token_count(candidate) > self.max_chunk_size:
                groups.append(
                    SemanticGroup(
                        tuple(pending),
                        "maximum_token_limit",
                        float(distances[index - 1]),
                        threshold,
                    )
                )
                pending = [unit]
            else:
                pending = candidate

            boundary_index = index + 1
            if (
                boundary_index in semantic_breaks
                and self._group_token_count(pending) >= self.min_chunk_size
            ):
                groups.append(
                    SemanticGroup(
                        tuple(pending),
                        "semantic_distance",
                        float(distances[index]),
                        threshold,
                    )
                )
                pending = []
            elif (
                self._group_token_count(pending) >= self.target_chunk_size
                and index < len(distances)
                and float(distances[index]) >= float(np.median(distances))
            ):
                groups.append(
                    SemanticGroup(
                        tuple(pending),
                        "target_size_semantic_local_minimum",
                        float(distances[index]),
                        threshold,
                    )
                )
                pending = []

        if pending:
            groups.append(SemanticGroup(tuple(pending), "section_end", None, threshold))
        return self._merge_tiny_groups(groups)

    def _merge_tiny_groups(self, groups: Sequence[SemanticGroup]) -> list[SemanticGroup]:
        merged: list[SemanticGroup] = []
        for group in groups:
            if (
                merged
                and self._group_token_count(group.units) < self.min_chunk_size
                and self._group_token_count((*merged[-1].units, *group.units))
                <= self.max_chunk_size
            ):
                previous = merged.pop()
                merged.append(
                    SemanticGroup(
                        (*previous.units, *group.units),
                        group.boundary_reason,
                        group.boundary_distance,
                        group.breakpoint_threshold,
                    )
                )
            else:
                merged.append(group)
        if (
            len(merged) > 1
            and self._group_token_count(merged[0].units) < self.min_chunk_size
            and self._group_token_count((*merged[0].units, *merged[1].units))
            <= self.max_chunk_size
        ):
            first, second = merged[:2]
            merged[:2] = [
                SemanticGroup(
                    (*first.units, *second.units),
                    second.boundary_reason,
                    second.boundary_distance,
                    second.breakpoint_threshold,
                )
            ]
        return merged

    def _group_token_count(self, units: Sequence[SemanticUnit]) -> int:
        if not units:
            return 0
        prefix = self._section_prefix(units[0].section_path)
        body = " ".join(unit.text for unit in units)
        text = f"{prefix}\n\n{body}" if prefix else body
        return self._count_tokens(text)

    def _build_semantic_chunk(
        self,
        group: SemanticGroup,
        document: Mapping[str, Any],
        document_id: str,
        chunk_index: int,
        document_metadata: Mapping[str, Any] | None,
    ) -> Chunk:
        section_path = group.units[0].section_path
        prefix = self._section_prefix(section_path)
        separator = "\n" if all(unit.element_type == "list_item" for unit in group.units) else " "
        body = separator.join(unit.text for unit in group.units)
        text = f"{prefix}\n\n{body}" if prefix else body
        metadata = dict(document_metadata or {})
        metadata.update(
            {
                "document_name": document.get("name"),
                "embedding_model": self.embedder.model_name,
                "semantic_unit_count": len(group.units),
                "element_types": list(dict.fromkeys(unit.element_type for unit in group.units)),
                "boundary_reason": group.boundary_reason,
                "boundary_distance": group.boundary_distance,
                "breakpoint_threshold": group.breakpoint_threshold,
                "breakpoint_percentile": self.breakpoint_percentile,
                "buffer_size": self.buffer_size,
                "fallback_token_split": any(unit.fallback_token_split for unit in group.units),
                "fallback_reasons": list(
                    dict.fromkeys(
                        unit.fallback_reason
                        for unit in group.units
                        if unit.fallback_reason
                    )
                ),
                "section_boundary_enforced": True,
            }
        )
        return Chunk(
            chunk_id=f"{document_id}-semantic-custom-{chunk_index:04d}",
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
            page_numbers=sorted({page for unit in group.units for page in unit.page_numbers}),
            section_path=list(section_path),
            source_refs=list(
                dict.fromkeys(ref for unit in group.units for ref in unit.source_refs if ref)
            ),
            strategy=self.strategy,
            token_count=self._count_tokens(text),
            metadata=metadata,
        )


def _validate_configuration(
    minimum: int,
    target: int,
    maximum: int,
    percentile: float,
    buffer_size: int,
) -> None:
    if minimum <= 0:
        raise ValueError("min_chunk_size must be greater than 0")
    if not minimum <= target <= maximum:
        raise ValueError("chunk sizes must satisfy minimum <= target <= maximum")
    if not 0 < percentile <= 100:
        raise ValueError("breakpoint_percentile must be in (0, 100]")
    if buffer_size < 0:
        raise ValueError("buffer_size cannot be negative")
