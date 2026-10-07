from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tiktoken
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc import DoclingDocument

from src.chunking.models import Chunk


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
        self.native_chunker = HybridChunker(
            tokenizer=OpenAITokenizer(
                tokenizer=self.encoding,
                max_tokens=chunk_size,
            ),
            merge_peers=merge_peers,
            repeat_table_header=repeat_table_header,
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

        docling_document = DoclingDocument.model_validate(document)
        native_chunks = self.native_chunker.chunk(dl_doc=docling_document)
        chunks: list[Chunk] = []

        for chunk_index, native_chunk in enumerate(native_chunks):
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
                }
            )

            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{document_id}-docling-hybrid-{chunk_index:04d}"
                    ),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=contextualized_text,
                    page_numbers=page_numbers,
                    section_path=section_path,
                    source_refs=source_refs,
                    strategy="structure-aware-docling-hybrid",
                    token_count=self._count_tokens(contextualized_text),
                    metadata=metadata,
                )
            )

        return chunks

    @staticmethod
    def _label_value(label: Any) -> str:
        value = getattr(label, "value", label)
        return str(value)

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
