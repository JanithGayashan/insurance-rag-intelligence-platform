from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import tiktoken

from src.chunking.models import Chunk


_HEADING_LABELS = {"title", "section_header"}
_IGNORED_LABELS = {"page_header", "page_footer"}


@dataclass(frozen=True)
class _Heading:
    text: str
    source_ref: str


@dataclass(frozen=True)
class _Block:
    text: str
    page_numbers: tuple[int, ...]
    source_refs: tuple[str, ...]
    section_path: tuple[str, ...]
    element_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class StructureAwareChunker:
    """Create retrieval chunks from native Docling document JSON.

    Reading order, section boundaries, lists, tables, page provenance,
    and native JSON references are preserved. Adjacent elements from the
    same section are packed together until the token budget is reached.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk_file(
        self,
        document_path: Path,
        document_id: str | None = None,
        document_metadata: Mapping[str, Any] | None = None,
    ) -> list[Chunk]:
        """Load a Docling JSON file and return structure-aware chunks."""

        document_path = Path(document_path)
        with document_path.open("r", encoding="utf-8") as document_file:
            document = json.load(document_file)

        resolved_document_id = document_id or document_path.parent.name
        return self.chunk_document(
            document=document,
            document_id=resolved_document_id,
            document_metadata=document_metadata,
        )

    def chunk_document(
        self,
        document: Mapping[str, Any],
        document_id: str,
        document_metadata: Mapping[str, Any] | None = None,
    ) -> list[Chunk]:
        """Chunk one native Docling document dictionary."""

        if not document_id.strip():
            raise ValueError("document_id cannot be empty")
        if not isinstance(document, Mapping):
            raise TypeError("document must be a mapping")

        blocks = list(self._extract_blocks(document))
        chunks: list[Chunk] = []
        pending: list[_Block] = []

        def flush_pending() -> None:
            if not pending:
                return
            chunks.append(
                self._build_chunk(
                    blocks=pending,
                    document=document,
                    document_id=document_id,
                    chunk_index=len(chunks),
                    document_metadata=document_metadata,
                )
            )
            pending.clear()

        for block in blocks:
            if block.element_type == "table":
                flush_pending()
                for table_part in self._split_block(block):
                    chunks.append(
                        self._build_chunk(
                            blocks=[table_part],
                            document=document,
                            document_id=document_id,
                            chunk_index=len(chunks),
                            document_metadata=document_metadata,
                        )
                    )
                continue

            if pending and pending[0].section_path != block.section_path:
                flush_pending()

            candidate = [*pending, block]
            if self._count_tokens(self._render_blocks(candidate)) <= self.chunk_size:
                pending.append(block)
                continue

            flush_pending()
            block_parts = self._split_block(block)
            if len(block_parts) > 1:
                for block_part in block_parts:
                    chunks.append(
                        self._build_chunk(
                            blocks=[block_part],
                            document=document,
                            document_id=document_id,
                            chunk_index=len(chunks),
                            document_metadata=document_metadata,
                        )
                    )
            else:
                pending.append(block)

        flush_pending()
        return chunks

    def _extract_blocks(
        self,
        document: Mapping[str, Any],
    ) -> Iterator[_Block]:
        registry = self._build_registry(document)
        body = document.get("body")
        if not isinstance(body, Mapping):
            raise ValueError("Docling document is missing a body object")

        heading_stack: list[_Heading] = []
        visited: set[str] = set()

        for node, group_labels in self._walk_children(
            body,
            registry,
            visited,
        ):
            label = str(node.get("label", "unspecified"))
            source_ref = str(node.get("self_ref", ""))

            if label in _IGNORED_LABELS:
                continue

            if label in _HEADING_LABELS:
                heading_text = self._normalize_text(node.get("text", ""))
                if not heading_text:
                    continue
                heading_level = self._infer_heading_level(
                    node,
                    heading_text,
                    heading_stack,
                )
                del heading_stack[max(0, heading_level - 1) :]
                heading_stack.append(
                    _Heading(
                        text=heading_text,
                        source_ref=source_ref,
                    )
                )
                continue

            block = self._node_to_block(
                node=node,
                group_labels=group_labels,
                heading_stack=heading_stack,
            )
            if block is not None:
                yield block

    @staticmethod
    def _build_registry(
        document: Mapping[str, Any],
    ) -> dict[str, Mapping[str, Any]]:
        registry: dict[str, Mapping[str, Any]] = {}
        for collection_name in (
            "groups",
            "texts",
            "pictures",
            "tables",
            "key_value_items",
            "form_items",
        ):
            collection = document.get(collection_name, [])
            if not isinstance(collection, Sequence) or isinstance(
                collection,
                (str, bytes),
            ):
                continue
            for node in collection:
                if not isinstance(node, Mapping):
                    continue
                source_ref = node.get("self_ref")
                if isinstance(source_ref, str):
                    registry[source_ref] = node
        return registry

    def _walk_children(
        self,
        parent: Mapping[str, Any],
        registry: Mapping[str, Mapping[str, Any]],
        visited: set[str],
        group_labels: tuple[str, ...] = (),
    ) -> Iterator[tuple[Mapping[str, Any], tuple[str, ...]]]:
        children = parent.get("children", [])
        if not isinstance(children, Sequence) or isinstance(
            children,
            (str, bytes),
        ):
            return

        for child_reference in children:
            if not isinstance(child_reference, Mapping):
                continue
            source_ref = child_reference.get("$ref")
            if not isinstance(source_ref, str) or source_ref in visited:
                continue
            node = registry.get(source_ref)
            if node is None:
                continue
            visited.add(source_ref)

            node_children = node.get("children", [])
            if isinstance(node_children, Sequence) and not isinstance(
                node_children,
                (str, bytes),
            ) and node_children:
                node_label = str(node.get("label", "group"))
                yield from self._walk_children(
                    node,
                    registry,
                    visited,
                    (*group_labels, node_label),
                )
                continue

            yield node, group_labels

    def _node_to_block(
        self,
        node: Mapping[str, Any],
        group_labels: tuple[str, ...],
        heading_stack: Sequence[_Heading],
    ) -> _Block | None:
        source_ref = str(node.get("self_ref", ""))
        label = str(node.get("label", "unspecified"))
        page_numbers = self._page_numbers(node)
        heading_refs = tuple(
            heading.source_ref
            for heading in heading_stack
            if heading.source_ref
        )
        section_path = tuple(heading.text for heading in heading_stack)

        if source_ref.startswith("#/tables/"):
            table_text, rows = self._render_table(node)
            if not table_text:
                return None
            return _Block(
                text=table_text,
                page_numbers=page_numbers,
                source_refs=(*heading_refs, source_ref),
                section_path=section_path,
                element_type="table",
                metadata={
                    "docling_label": label,
                    "group_labels": list(group_labels),
                    "table_rows": rows,
                },
            )

        text = self._normalize_text(node.get("text", ""))
        if not text:
            return None
        if label == "list_item" or "list" in group_labels:
            text = f"- {text}"
        elif label == "caption":
            text = f"Caption: {text}"

        return _Block(
            text=text,
            page_numbers=page_numbers,
            source_refs=(*heading_refs, source_ref),
            section_path=section_path,
            element_type=label,
            metadata={
                "docling_label": label,
                "group_labels": list(group_labels),
            },
        )

    def _build_chunk(
        self,
        blocks: Sequence[_Block],
        document: Mapping[str, Any],
        document_id: str,
        chunk_index: int,
        document_metadata: Mapping[str, Any] | None,
    ) -> Chunk:
        text = self._render_blocks(blocks)
        page_numbers = sorted(
            {
                page_number
                for block in blocks
                for page_number in block.page_numbers
            }
        )
        source_refs = list(
            dict.fromkeys(
                source_ref
                for block in blocks
                for source_ref in block.source_refs
                if source_ref
            )
        )
        element_types = list(
            dict.fromkeys(block.element_type for block in blocks)
        )
        metadata = dict(document_metadata or {})
        metadata.update(
            {
                "document_name": document.get("name"),
                "element_types": element_types,
                "block_count": len(blocks),
                "contains_table": "table" in element_types,
                "section_depth": len(blocks[0].section_path),
            }
        )
        split_parts = [
            block.metadata.get("split_part")
            for block in blocks
            if block.metadata.get("split_part") is not None
        ]
        if split_parts:
            metadata["split_part"] = split_parts[0]
            metadata["oversized_block_split"] = True

        return Chunk(
            chunk_id=f"{document_id}-structure-{chunk_index:04d}",
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
            page_numbers=page_numbers,
            section_path=list(blocks[0].section_path),
            source_refs=source_refs,
            strategy="structure-aware",
            token_count=self._count_tokens(text),
            metadata=metadata,
        )

    def _split_block(self, block: _Block) -> list[_Block]:
        if self._count_tokens(self._render_blocks([block])) <= self.chunk_size:
            return [block]

        table_rows = block.metadata.get("table_rows")
        if block.element_type == "table" and isinstance(table_rows, list):
            row_parts = self._split_table_rows(block, table_rows)
            if row_parts:
                return row_parts

        prefix = self._section_prefix(block.section_path)
        separator = "\n\n" if prefix else ""
        prefix_token_count = self._count_tokens(prefix + separator)
        available_tokens = max(1, self.chunk_size - prefix_token_count)
        overlap = min(self.overlap, max(0, available_tokens - 1))
        step = max(1, available_tokens - overlap)
        tokens = self.encoding.encode(block.text)
        parts: list[_Block] = []

        for start in range(0, len(tokens), step):
            segment_tokens = tokens[start : start + available_tokens]
            if not segment_tokens:
                break
            segment = self.encoding.decode(segment_tokens)
            part = replace(
                block,
                text=segment,
                metadata={
                    **block.metadata,
                    "split_part": len(parts),
                },
            )
            parts.append(part)
            if start + available_tokens >= len(tokens):
                break
        return parts

    def _split_table_rows(
        self,
        block: _Block,
        rows: list[Any],
    ) -> list[_Block]:
        normalized_rows = [
            str(row).strip()
            for row in rows
            if str(row).strip()
        ]
        if not normalized_rows:
            return []

        parts: list[_Block] = []
        pending_rows: list[str] = []
        for row in normalized_rows:
            candidate_rows = [*pending_rows, row]
            candidate = replace(
                block,
                text="[Table]\n" + "\n".join(candidate_rows),
            )
            if (
                pending_rows
                and self._count_tokens(self._render_blocks([candidate]))
                > self.chunk_size
            ):
                parts.append(
                    replace(
                        block,
                        text="[Table]\n" + "\n".join(pending_rows),
                        metadata={
                            **block.metadata,
                            "table_rows": list(pending_rows),
                            "split_part": len(parts),
                        },
                    )
                )
                pending_rows = [row]
            else:
                pending_rows = candidate_rows

        if pending_rows:
            parts.append(
                replace(
                    block,
                    text="[Table]\n" + "\n".join(pending_rows),
                    metadata={
                        **block.metadata,
                        "table_rows": list(pending_rows),
                        "split_part": len(parts),
                    },
                )
            )

        if any(
            self._count_tokens(self._render_blocks([part])) > self.chunk_size
            for part in parts
        ):
            return []
        return parts

    def _render_blocks(self, blocks: Sequence[_Block]) -> str:
        if not blocks:
            return ""
        prefix = self._section_prefix(blocks[0].section_path)
        body = "\n\n".join(block.text for block in blocks)
        return f"{prefix}\n\n{body}" if prefix else body

    def _section_prefix(self, section_path: Sequence[str]) -> str:
        if not section_path:
            return ""
        prefix_budget = max(8, self.chunk_size // 3)
        selected: list[str] = []
        for heading in reversed(section_path):
            candidate = "Section: " + " > ".join(reversed([heading, *selected]))
            if selected and self._count_tokens(candidate) > prefix_budget:
                break
            selected.insert(0, heading)
        prefix = "Section: " + " > ".join(selected)
        prefix_tokens = self.encoding.encode(prefix)
        if len(prefix_tokens) > prefix_budget:
            prefix = self.encoding.decode(prefix_tokens[:prefix_budget])
        return prefix

    @staticmethod
    def _page_numbers(node: Mapping[str, Any]) -> tuple[int, ...]:
        provenance = node.get("prov", [])
        if not isinstance(provenance, Sequence) or isinstance(
            provenance,
            (str, bytes),
        ):
            return ()
        pages = {
            int(item["page_no"])
            for item in provenance
            if isinstance(item, Mapping)
            and isinstance(item.get("page_no"), int)
        }
        return tuple(sorted(pages))

    def _infer_heading_level(
        self,
        node: Mapping[str, Any],
        text: str,
        heading_stack: Sequence[_Heading],
    ) -> int:
        for source in (node, node.get("metadata", {})):
            if not isinstance(source, Mapping):
                continue
            for key in ("level", "heading_level"):
                level = source.get(key)
                if isinstance(level, int) and level > 0:
                    return level

        if node.get("label") == "title":
            return 1
        if text.isupper() and len(text.split()) <= 15:
            return 1
        if re.match(r"^section\s+[\divxlcdm]+\b", text, re.IGNORECASE):
            return 2
        numbered_heading = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+", text)
        if numbered_heading:
            return 2 + numbered_heading.group(1).count(".") + 1
        if heading_stack and re.match(
            r"^section\s+[\divxlcdm]+\b",
            heading_stack[-1].text,
            re.IGNORECASE,
        ):
            return 3
        return 2 if heading_stack else 1

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"[ \t\f\v]+", " ", value).strip()

    @staticmethod
    def _render_table(
        node: Mapping[str, Any],
    ) -> tuple[str, list[str]]:
        data = node.get("data", {})
        if not isinstance(data, Mapping):
            return "", []
        grid = data.get("grid", [])
        rows: list[str] = []
        if isinstance(grid, Sequence) and not isinstance(grid, (str, bytes)):
            for grid_row in grid:
                if not isinstance(grid_row, Sequence) or isinstance(
                    grid_row,
                    (str, bytes),
                ):
                    continue
                cells = []
                for cell in grid_row:
                    if not isinstance(cell, Mapping):
                        continue
                    cell_text = StructureAwareChunker._normalize_text(
                        cell.get("text", "")
                    ).replace("|", "\\|")
                    cells.append(cell_text)
                if any(cells):
                    rows.append(" | ".join(cells))
        if not rows:
            table_cells = data.get("table_cells", [])
            if isinstance(table_cells, Sequence) and not isinstance(
                table_cells,
                (str, bytes),
            ):
                ordered_cells = sorted(
                    (
                        cell
                        for cell in table_cells
                        if isinstance(cell, Mapping)
                    ),
                    key=lambda cell: (
                        cell.get("start_row_offset_idx", 0),
                        cell.get("start_col_offset_idx", 0),
                    ),
                )
                current_row: int | None = None
                row_cells: list[str] = []
                for cell in ordered_cells:
                    row_index = int(cell.get("start_row_offset_idx", 0))
                    if current_row is not None and row_index != current_row:
                        rows.append(" | ".join(row_cells))
                        row_cells = []
                    current_row = row_index
                    row_cells.append(
                        StructureAwareChunker._normalize_text(
                            cell.get("text", "")
                        ).replace("|", "\\|")
                    )
                if row_cells:
                    rows.append(" | ".join(row_cells))
        if not rows:
            return "", []
        return "[Table]\n" + "\n".join(rows), rows

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))


def write_chunks(chunks: Sequence[Chunk], output_path: Path) -> None:
    """Write chunks as UTF-8 JSON for later embedding and indexing."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            [chunk.model_dump(mode="json") for chunk in chunks],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
