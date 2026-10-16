from __future__ import annotations

import json
from copy import deepcopy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from pydantic import PrivateAttr

import tiktoken
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc import ContentLayer, DocItemLabel
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer
from docling_core.transforms.chunker import DocChunk
from docling_core.transforms.serializer.base import BaseSerializerProvider
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer, OrigListItemMarkerMode

from src.chunking.models import Chunk
from src.chunking.structure_aware.chunker import StructureAwareChunker
from src.chunking.structure_aware.context import resolve_contexts
from src.chunking.structure_aware.text import fitting_prefix, render_text, split_text


class _PolicySerializerProvider(BaseSerializerProvider):
    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        serializer = ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())
        serializer.params = serializer.params.model_copy(update={
            "layers": set(ContentLayer),
            "labels": set(DocItemLabel) - {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.PICTURE},
            "orig_list_item_marker_mode": OrigListItemMarkerMode.ALWAYS,
            "ensure_valid_list_item_marker": False,
            "traverse_pictures": True,
        })
        return serializer


class _SectionSafeHybridChunker(HybridChunker):
    """Restrict native peer merging to the same source section identity.

    Docling compares heading strings. Repeated headings can have identical
    strings in different sections, so partition first, then let Docling merge.
    The pinned Docling API is covered by the repeated-heading regression test.
    """
    _source_sections: dict[str, str] = PrivateAttr(default_factory=dict)

    def _merge_chunks_with_matching_metadata(self, chunks: Iterable[DocChunk]) -> list[DocChunk]:
        result: list[DocChunk] = []
        pending: list[DocChunk] = []
        previous: frozenset[str] | None = None
        for chunk in chunks:
            sections = frozenset(self._source_sections[item.self_ref] for item in chunk.meta.doc_items if item.self_ref in self._source_sections)
            if pending and sections != previous:
                result.extend(super()._merge_chunks_with_matching_metadata(pending))
                pending = []
            pending.append(chunk)
            previous = sections
        if pending:
            result.extend(super()._merge_chunks_with_matching_metadata(pending))
        return result


class DoclingHybridStructureChunker:
    """Adapt Docling's native HybridChunker to the project's Chunk model."""

    def __init__(
        self,
        chunk_size: int = 500,
        encoding_name: str = "cl100k_base",
        merge_peers: bool = True,
        repeat_table_header: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        self.chunk_size = chunk_size
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.native_chunker = _SectionSafeHybridChunker(
            tokenizer=OpenAITokenizer(
                tokenizer=self.encoding,
                max_tokens=chunk_size,
            ),
            merge_peers=merge_peers,
            repeat_table_header=repeat_table_header,
            always_emit_headings=True,
            serializer_provider=_PolicySerializerProvider(),
        )

    def chunk_file(
        self,
        document_path: Path,
        document_id: str | None = None,
        document_metadata: Mapping[str, Any] | None = None,
    ) -> list[Chunk]:
        """Load serialized Docling JSON and return native hybrid chunks."""

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
    ) -> list[Chunk]:
        """Validate one Docling document and adapt its native chunks."""

        if not document_id.strip():
            raise ValueError("document_id cannot be empty")
        if not isinstance(document, Mapping):
            raise TypeError("document must be a mapping")

        # Work on a copy: keep the parser's native output immutable.
        normalized = deepcopy(dict(document))
        helper = StructureAwareChunker()
        registry = helper._build_registry(normalized)
        items = list(helper._walk_children(normalized["body"], registry, set()))
        contexts = resolve_contexts([node for node, _ in items])
        heading_refs_by_path = {context.headings: context.refs for context in contexts.values()}
        for node, _ in items:
            if node.get("label") == "section_header":
                context = contexts[node["self_ref"]]
                node["text"] = context.headings[-1]
                title_count = sum(registry.get(normalized_ref, {}).get("label") == "title" for normalized_ref in context.refs)
                node["level"] = max(1, len(context.headings) - title_count)
        docling_document = DoclingDocument.model_validate(normalized)
        if isinstance(self.native_chunker, _SectionSafeHybridChunker):
            self.native_chunker._source_sections = {ref: context.refs[-1] if context.refs else "introduction" for ref, context in contexts.items()}
        native_chunks = self.native_chunker.chunk(dl_doc=docling_document)
        chunks: list[Chunk] = []

        for native_index, native_chunk in enumerate(native_chunks):
            contextualized_text = self.native_chunker.contextualize(native_chunk)
            doc_items = list(native_chunk.meta.doc_items)
            section_path = list(native_chunk.meta.headings or [])
            native_meta = native_chunk.meta.model_dump(mode="json")
            captions = list(native_meta.get("captions") or [])
            page_numbers = sorted(
                {
                    provenance.page_no
                    for item in doc_items
                    for provenance in (item.prov or [])
                }
            )
            source_refs = list(
                dict.fromkeys(
                    str(item.self_ref)
                    for item in doc_items
                    if item.self_ref
                )
            )
            matching_contexts = [contexts[ref] for ref in source_refs if ref in contexts and contexts[ref].headings == tuple(section_path)]
            heading_source_refs = list(dict.fromkeys(ref for context in matching_contexts for ref in context.refs)) or list(heading_refs_by_path.get(tuple(section_path), ()))
            element_types = list(
                dict.fromkeys(self._label_value(item.label) for item in doc_items)
            )

            metadata = dict(document_metadata or {})
            metadata.update(
                {
                    "document_name": docling_document.name,
                    "element_types": element_types,
                    "block_count": len(doc_items),
                    "contains_table": any(
                        label in {"table", "document_index"}
                        for label in element_types
                    ),
                    "captions": captions,
                    "section_depth": len(section_path),
                    "native_text_token_count": self._count_tokens(
                        native_chunk.text
                    ),
                    "contextualized_with_headings": (
                        contextualized_text != native_chunk.text
                    ),
                    "chunker": "docling.HybridChunker",
                    "table_serializer": "docling.MarkdownTableSerializer",
                    "native_chunk_index": native_index,
                    "body_text": native_chunk.text,
                    "heading_source_refs": heading_source_refs,
                    "body_source_refs": source_refs,
                    "heading_context_inferred": any(contexts.get(ref) and contexts[ref].inferred for ref in source_refs),
                    "tokenizer": self.encoding.name,
                    "chunk_size": self.chunk_size,
                    "list_markers": {str(item.self_ref): item.marker for item in doc_items if hasattr(item, "marker") and item.marker},
                }
            )
            parts = [(contextualized_text, {})]
            if self._count_tokens(contextualized_text) > self.chunk_size:
                full_prefix = "\n".join([*section_path, *captions])
                prefix = fitting_prefix(full_prefix, self.encoding, min(self.chunk_size // 3, max(0, self.chunk_size - 8)))
                # Empty heading-only chunks still contain useful content.
                body = native_chunk.text or full_prefix
                segments = split_text(body, self.encoding, self.chunk_size, prefix=prefix)
                parts = [(render_text(prefix, segment.text), {
                    "oversized_block_split": True, "fallback_token_split": segment.hard_split,
                    "fallback_reason": "contextualized-native-chunk-over-budget", "split_part": index,
                    "context_prefix_truncated": prefix != full_prefix,
                    "native_text_span": {"start": segment.start, "end": segment.end},
                    "body_text": segment.text,
                }) for index, segment in enumerate(segments)]
            for text, split_metadata in parts:
                chunk_index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}-docling-hybrid-{chunk_index:04d}",
                        document_id=document_id,
                        chunk_index=chunk_index,
                        text=text,
                        page_numbers=page_numbers,
                        section_path=section_path,
                        source_refs=source_refs,
                        strategy="structure-aware-docling-hybrid",
                        token_count=self._count_tokens(text),
                        metadata={**metadata, **split_metadata},
                    )
                )

        return chunks

    @staticmethod
    def _label_value(label: Any) -> str:
        value = getattr(label, "value", label)
        return str(value)

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
