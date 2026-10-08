from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.chunking.parent_child.models import (
    HierarchicalChunk,
    ParentChildHierarchy,
)
from src.chunking.parent_child.validation import validate_configuration
from src.chunking.structure_aware.chunker import StructureAwareChunker


_HEADING_LABELS = {"title", "section_header"}
_IGNORED_LABELS = {"page_header", "page_footer"}
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=(?:[\"'“‘(]*[A-Z0-9]))"
)
_WORD = re.compile(r"\w+", re.UNICODE)
_MAJOR_SECTION = re.compile(
    r"^(?:"
    r"preamble|index|coverage|key definitions?|general exclusions?|conditions?|"
    r"additional covers?|claims?(?: handling)? procedure|"
    r"grievances?(?:/complaints?)?.*|complaints? handling.*|"
    r"dispute resolution.*|general terms?|appendix.*|annexure"
    r")$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _SectionHeading:
    text: str
    source_ref: str


@dataclass(frozen=True)
class _ChildDraft:
    body: str
    section_path: tuple[str, ...]
    page_numbers: tuple[int, ...]
    source_refs: tuple[str, ...]
    element_types: tuple[str, ...]
    source_block_count: int
    fallback_token_split: bool = False
    fallback_reason: str | None = None
    structural_split: str | None = None


class StructuredParentChildChunker(StructureAwareChunker):
    """Create section parents and paragraph/sentence children from Docling JSON.

    Heading, list, table, reading-order, page, and source-reference information
    comes from Docling. Children never cross a section boundary. Raw token
    slicing is reserved for a paragraph, table row, or sentence that cannot fit
    by itself, and every such fallback is marked in metadata.
    """

    strategy = "parent-child-docling-structured"

    def __init__(
        self,
        parent_size: int = 1500,
        child_size: int = 300,
        tiny_chunk_threshold: int = 20,
        encoding_name: str = "cl100k_base",
    ) -> None:
        validate_configuration(parent_size, child_size, 0)
        if tiny_chunk_threshold < 0:
            raise ValueError("tiny_chunk_threshold cannot be negative")
        super().__init__(
            chunk_size=child_size,
            overlap=0,
            encoding_name=encoding_name,
        )
        self.parent_size = parent_size
        self.child_size = child_size
        self.tiny_chunk_threshold = tiny_chunk_threshold
        self.encoding_name = encoding_name

    def chunk_file(
        self,
        document_path: Path,
        document_id: str | None = None,
        document_metadata: Mapping[str, Any] | None = None,
    ) -> ParentChildHierarchy:
        document_path = Path(document_path)
        with document_path.open("r", encoding="utf-8") as document_file:
            document = json.load(document_file)
        return self.chunk_document(
            document=document,
            document_id=document_id or document_path.parent.name,
            document_metadata=document_metadata,
        )

    def chunk_document(
        self,
        document: Mapping[str, Any],
        document_id: str,
        document_metadata: Mapping[str, Any] | None = None,
    ) -> ParentChildHierarchy:
        if not document_id.strip():
            raise ValueError("document_id cannot be empty")
        if not isinstance(document, Mapping):
            raise TypeError("document must be a mapping")

        blocks = list(self._extract_section_blocks(document))
        if not blocks:
            raise ValueError("Docling document contains no chunkable content")

        section_groups = list(self._group_blocks_by_section(blocks))
        section_drafts: list[tuple[tuple[str, ...], list[_ChildDraft]]] = []
        for section_path, section_blocks in section_groups:
            drafts = self._build_child_drafts(section_path, section_blocks)
            if drafts:
                section_drafts.append((section_path, drafts))

        parent_sections = list(self._group_drafts_by_parent_section(section_drafts))
        parents: list[HierarchicalChunk] = []
        children: list[HierarchicalChunk] = []
        for parent_path, drafts in parent_sections:
            parent_groups = self._group_drafts_into_parents(parent_path, drafts)
            for section_segment_index, parent_drafts in enumerate(parent_groups):
                parent_index = len(parents)
                parent_id = (
                    f"{document_id}-structured-parent-{parent_index:04d}"
                )
                child_ids: list[str] = []
                for child_index_in_parent, draft in enumerate(parent_drafts):
                    child_index = len(children)
                    child_id = (
                        f"{document_id}-structured-child-{child_index:04d}"
                    )
                    child_ids.append(child_id)
                    child_text = self._render_section(
                        draft.section_path, draft.body
                    )
                    child_metadata = dict(document_metadata or {})
                    child_metadata.update(
                        {
                            "parent_index": parent_index,
                            "child_index_in_parent": child_index_in_parent,
                            "element_types": list(draft.element_types),
                            "source_block_count": draft.source_block_count,
                            "contains_table": "table" in draft.element_types,
                            "fallback_token_split": draft.fallback_token_split,
                            "fallback_reason": draft.fallback_reason,
                            "structural_split": draft.structural_split,
                            "sentence_boundary_preserved": not (
                                draft.fallback_token_split
                            ),
                        }
                    )
                    children.append(
                        HierarchicalChunk(
                            chunk_id=child_id,
                            document_id=document_id,
                            chunk_index=child_index,
                            level=1,
                            text=child_text,
                            token_count=self._count_tokens(child_text),
                            page_numbers=list(draft.page_numbers),
                            section_path=list(draft.section_path),
                            source_refs=list(draft.source_refs),
                            parent_id=parent_id,
                            is_leaf=True,
                            strategy=self.strategy,
                            metadata=child_metadata,
                        )
                    )

                parent_text = self._render_parent(parent_path, parent_drafts)
                parent_metadata = dict(document_metadata or {})
                parent_metadata.update(
                    {
                        "document_name": document.get("name"),
                        "section_segment_index": section_segment_index,
                        "section_segment_count": len(parent_groups),
                        "child_section_count": len(
                            {draft.section_path for draft in parent_drafts}
                        ),
                        "contains_table": any(
                            "table" in draft.element_types
                            for draft in parent_drafts
                        ),
                        "fallback_child_count": sum(
                            draft.fallback_token_split for draft in parent_drafts
                        ),
                    }
                )
                parents.append(
                    HierarchicalChunk(
                        chunk_id=parent_id,
                        document_id=document_id,
                        chunk_index=parent_index,
                        level=0,
                        text=parent_text,
                        token_count=self._count_tokens(parent_text),
                        page_numbers=self._merge_ints(
                            draft.page_numbers for draft in parent_drafts
                        ),
                        section_path=list(parent_path),
                        source_refs=self._merge_strings(
                            draft.source_refs for draft in parent_drafts
                        ),
                        children_ids=child_ids,
                        is_leaf=False,
                        strategy=self.strategy,
                        metadata=parent_metadata,
                    )
                )

        quality = self._content_quality(blocks, section_drafts)
        parent_paths = {
            parent.chunk_id: tuple(parent.section_path) for parent in parents
        }
        quality["children_crossing_sections"] = sum(
            child.parent_id not in parent_paths
            or tuple(
                child.section_path[
                    : len(parent_paths.get(child.parent_id or "", ()))
                ]
            )
            != parent_paths.get(child.parent_id or "", ())
            for child in children
        )
        return ParentChildHierarchy(
            document_id=document_id,
            strategy=self.strategy,
            parent_size=self.parent_size,
            child_size=self.child_size,
            child_overlap=0,
            parents=parents,
            children=children,
            metadata={
                "tokenizer": self.encoding_name,
                "source_format": "docling-json",
                "parent_boundary": "docling-section",
                "child_boundary": "paragraph-sentence-table-row",
                "quality": quality,
            },
        )

    def _extract_section_blocks(
        self,
        document: Mapping[str, Any],
    ) -> Iterator[Any]:
        registry = self._build_registry(document)
        body = document.get("body")
        if not isinstance(body, Mapping):
            raise ValueError("Docling document is missing a body object")

        visited: set[str] = set()
        major_heading: _SectionHeading | None = None
        pending_headings: list[_SectionHeading] = []
        content_since_heading = False

        for node, group_labels in self._walk_children(body, registry, visited):
            label = str(node.get("label", "unspecified"))
            source_ref = str(node.get("self_ref", ""))
            if label in _IGNORED_LABELS:
                continue
            if label in _HEADING_LABELS:
                heading_text = self._normalize_text(node.get("text", ""))
                if not heading_text:
                    continue
                heading = _SectionHeading(heading_text, source_ref)
                if major_heading is None or self._is_major_heading(heading_text):
                    major_heading = heading
                    pending_headings = [heading]
                elif not content_since_heading and pending_headings:
                    pending_headings.append(heading)
                else:
                    pending_headings = (
                        [major_heading, heading]
                        if major_heading is not None
                        and major_heading.text != heading.text
                        else [heading]
                    )
                content_since_heading = False
                continue

            headings = pending_headings or [
                _SectionHeading("Document introduction", "")
            ]
            block = self._node_to_block(
                node=node,
                group_labels=group_labels,
                heading_stack=headings,
            )
            if block is not None:
                content_since_heading = True
                yield block

    @staticmethod
    def _is_major_heading(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip(" .:")
        if _MAJOR_SECTION.match(normalized):
            return True
        return bool(
            re.match(
                r"^section\s+\d+\s*:\s*(?:general\s+)?exclusions?$",
                normalized,
                re.IGNORECASE,
            )
            or re.match(
                r"^section\s+(?:\d+|[ivxlcdm]+)\s*[-–]\s*\S+",
                normalized,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _group_blocks_by_section(
        blocks: Sequence[Any],
    ) -> Iterator[tuple[tuple[str, ...], list[Any]]]:
        current_path: tuple[str, ...] | None = None
        pending: list[Any] = []
        for block in blocks:
            path = tuple(block.section_path) or ("Document introduction",)
            if current_path is not None and path != current_path:
                yield current_path, pending
                pending = []
            current_path = path
            pending.append(block)
        if current_path is not None and pending:
            yield current_path, pending

    def _build_child_drafts(
        self,
        section_path: tuple[str, ...],
        blocks: Sequence[Any],
    ) -> list[_ChildDraft]:
        result: list[_ChildDraft] = []
        pending: _ChildDraft | None = None

        def flush() -> None:
            nonlocal pending
            if pending is not None:
                result.append(pending)
                pending = None

        for block in blocks:
            pieces = self._split_structured_block(section_path, block)
            for piece in pieces:
                if "table" in piece.element_types:
                    flush()
                    result.append(piece)
                    continue
                candidate = piece if pending is None else self._combine(pending, piece)
                if self._fits_child(candidate):
                    pending = candidate
                else:
                    flush()
                    pending = piece
        flush()
        return result

    def _split_structured_block(
        self,
        section_path: tuple[str, ...],
        block: Any,
    ) -> list[_ChildDraft]:
        base = _ChildDraft(
            body=block.text,
            section_path=section_path,
            page_numbers=tuple(block.page_numbers),
            source_refs=tuple(block.source_refs),
            element_types=(block.element_type,),
            source_block_count=1,
        )
        if self._fits_child(base):
            return [base]
        if block.element_type == "table":
            return self._split_table(section_path, block)

        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(block.text)
            if sentence.strip()
        ]
        if len(sentences) <= 1:
            return self._token_fallback(base, "oversized_sentence_or_paragraph")

        pieces: list[_ChildDraft] = []
        pending_text = ""
        for sentence in sentences:
            candidate_text = (
                f"{pending_text} {sentence}".strip() if pending_text else sentence
            )
            candidate = self._replace_body(base, candidate_text)
            if self._fits_child(candidate):
                pending_text = candidate_text
                continue
            if pending_text:
                pieces.append(self._replace_body(base, pending_text))
                pending_text = ""
            sentence_draft = self._replace_body(base, sentence)
            if self._fits_child(sentence_draft):
                pending_text = sentence
            else:
                pieces.extend(
                    self._token_fallback(sentence_draft, "oversized_sentence")
                )
        if pending_text:
            pieces.append(self._replace_body(base, pending_text))
        return pieces

    def _split_table(
        self,
        section_path: tuple[str, ...],
        block: Any,
    ) -> list[_ChildDraft]:
        rows = [line.strip() for line in block.text.splitlines() if line.strip()]
        if rows and rows[0] == "[Table]":
            rows = rows[1:]
        base = _ChildDraft(
            body="",
            section_path=section_path,
            page_numbers=tuple(block.page_numbers),
            source_refs=tuple(block.source_refs),
            element_types=("table",),
            source_block_count=1,
            structural_split="table_rows",
        )
        pieces: list[_ChildDraft] = []
        if not rows:
            return self._token_fallback(
                self._replace_body(base, block.text), "oversized_table"
            )
        pending_rows: list[str] = []
        for row in rows:
            candidate_rows = [*pending_rows, row]
            candidate = self._replace_body(
                base, "[Table]\n" + "\n".join(candidate_rows)
            )
            if self._fits_child(candidate):
                pending_rows = candidate_rows
                continue
            if pending_rows:
                pieces.append(
                    self._replace_body(base, "[Table]\n" + "\n".join(pending_rows))
                )
                pending_rows = []
            row_draft = self._replace_body(base, f"[Table]\n{row}")
            if self._fits_child(row_draft):
                pending_rows = [row]
            else:
                pieces.extend(self._token_fallback(row_draft, "oversized_table_row"))
        if pending_rows:
            pieces.append(
                self._replace_body(base, "[Table]\n" + "\n".join(pending_rows))
            )
        return pieces

    def _token_fallback(
        self,
        draft: _ChildDraft,
        reason: str,
    ) -> list[_ChildDraft]:
        prefix = self._section_prefix(draft.section_path)
        prefix_tokens = self._count_tokens(f"{prefix}\n\n") if prefix else 0
        available = max(1, self.child_size - prefix_tokens)
        tokens = self.encoding.encode(draft.body)
        pieces: list[_ChildDraft] = []
        start = 0
        while start < len(tokens):
            segment_tokens = tokens[start : start + available]
            segment = self.encoding.decode(segment_tokens).strip()
            candidate = _ChildDraft(
                body=segment,
                section_path=draft.section_path,
                page_numbers=draft.page_numbers,
                source_refs=draft.source_refs,
                element_types=draft.element_types,
                source_block_count=draft.source_block_count,
                fallback_token_split=True,
                fallback_reason=reason,
                structural_split=draft.structural_split,
            )
            while segment_tokens and not self._fits_child(candidate):
                segment_tokens = segment_tokens[:-1]
                candidate = self._replace_body(
                    candidate, self.encoding.decode(segment_tokens).strip()
                )
            if candidate.body:
                pieces.append(candidate)
            start += max(1, len(segment_tokens))
        return pieces

    def _group_drafts_into_parents(
        self,
        parent_path: tuple[str, ...],
        drafts: Sequence[_ChildDraft],
    ) -> list[list[_ChildDraft]]:
        groups: list[list[_ChildDraft]] = []
        pending: list[_ChildDraft] = []
        for draft in drafts:
            candidate = [*pending, draft]
            if not pending or self._count_tokens(
                self._render_parent(parent_path, candidate)
            ) <= self.parent_size:
                pending = candidate
            else:
                groups.append(pending)
                pending = [draft]
        if pending:
            groups.append(pending)
        return groups

    @staticmethod
    def _group_drafts_by_parent_section(
        section_drafts: Sequence[tuple[tuple[str, ...], list[_ChildDraft]]],
    ) -> Iterator[tuple[tuple[str, ...], list[_ChildDraft]]]:
        current_path: tuple[str, ...] | None = None
        pending: list[_ChildDraft] = []
        for child_path, drafts in section_drafts:
            parent_path = child_path[:1] or ("Document introduction",)
            if current_path is not None and parent_path != current_path:
                yield current_path, pending
                pending = []
            current_path = parent_path
            pending.extend(drafts)
        if current_path is not None and pending:
            yield current_path, pending

    def _fits_child(self, draft: _ChildDraft) -> bool:
        return self._count_tokens(
            self._render_section(draft.section_path, draft.body)
        ) <= self.child_size

    def _render_section(self, section_path: Sequence[str], body: str) -> str:
        prefix = self._section_prefix(section_path)
        return f"{prefix}\n\n{body}" if prefix else body

    def _render_parent(
        self,
        parent_path: Sequence[str],
        drafts: Sequence[_ChildDraft],
    ) -> str:
        parts: list[str] = []
        active_child_path: tuple[str, ...] | None = None
        parent_tuple = tuple(parent_path)
        for draft in drafts:
            if draft.section_path != active_child_path:
                subsection = draft.section_path[len(parent_tuple) :]
                if subsection:
                    parts.append("Subsection: " + " > ".join(subsection))
                active_child_path = draft.section_path
            parts.append(draft.body)
        return self._render_section(parent_path, "\n\n".join(parts))

    @staticmethod
    def _replace_body(draft: _ChildDraft, body: str) -> _ChildDraft:
        return _ChildDraft(
            body=body,
            section_path=draft.section_path,
            page_numbers=draft.page_numbers,
            source_refs=draft.source_refs,
            element_types=draft.element_types,
            source_block_count=draft.source_block_count,
            fallback_token_split=draft.fallback_token_split,
            fallback_reason=draft.fallback_reason,
            structural_split=draft.structural_split,
        )

    @staticmethod
    def _combine(left: _ChildDraft, right: _ChildDraft) -> _ChildDraft:
        return _ChildDraft(
            body=f"{left.body}\n\n{right.body}",
            section_path=left.section_path,
            page_numbers=tuple(sorted(set(left.page_numbers) | set(right.page_numbers))),
            source_refs=tuple(dict.fromkeys((*left.source_refs, *right.source_refs))),
            element_types=tuple(
                dict.fromkeys((*left.element_types, *right.element_types))
            ),
            source_block_count=left.source_block_count + right.source_block_count,
            fallback_token_split=(
                left.fallback_token_split or right.fallback_token_split
            ),
            fallback_reason=left.fallback_reason or right.fallback_reason,
            structural_split=left.structural_split or right.structural_split,
        )

    def _content_quality(
        self,
        blocks: Sequence[Any],
        section_drafts: Sequence[tuple[tuple[str, ...], list[_ChildDraft]]],
    ) -> dict[str, Any]:
        drafts = [draft for _, items in section_drafts for draft in items]
        source_words = Counter(
            word
            for block in blocks
            for word in self._normalized_words(block.text)
        )
        output_words = Counter(
            word for draft in drafts for word in self._normalized_words(draft.body)
        )
        missing_words = sum((source_words - output_words).values())
        duplicate_words = sum((output_words - source_words).values())
        source_word_count = sum(source_words.values())
        expected_refs = {
            source_ref
            for block in blocks
            for source_ref in block.source_refs
            if source_ref
        }
        output_refs = {
            source_ref
            for draft in drafts
            for source_ref in draft.source_refs
            if source_ref
        }
        fallback_chunks = sum(draft.fallback_token_split for draft in drafts)
        return {
            "source_block_count": len(blocks),
            "source_word_count": source_word_count,
            "missing_word_count": missing_words,
            "missing_word_rate": self._rate(missing_words, source_word_count),
            "duplicate_word_count": duplicate_words,
            "duplicate_word_rate": self._rate(duplicate_words, source_word_count),
            "source_reference_coverage_rate": self._rate(
                len(expected_refs & output_refs), len(expected_refs)
            ),
            "fallback_token_split_chunks": fallback_chunks,
            "sentence_boundary_break_rate": self._rate(
                fallback_chunks, len(drafts)
            ),
            "tiny_chunks_under_threshold": sum(
                self._count_tokens(
                    self._render_section(draft.section_path, draft.body)
                )
                < self.tiny_chunk_threshold
                for draft in drafts
            ),
        }

    @staticmethod
    def _normalized_words(text: str) -> list[str]:
        return [word.casefold() for word in _WORD.findall(text)]

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    @staticmethod
    def _merge_ints(values: Sequence[Sequence[int]]) -> list[int]:
        return sorted({item for sequence in values for item in sequence})

    @staticmethod
    def _merge_strings(values: Sequence[Sequence[str]]) -> list[str]:
        return list(
            dict.fromkeys(item for sequence in values for item in sequence if item)
        )
