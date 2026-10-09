from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from pydantic import PrivateAttr

from src.chunking.models import Chunk
from src.chunking.semantic.custom import CustomSemanticChunker, _validate_configuration
from src.chunking.semantic.embeddings import TextEmbedder
from src.chunking.structure_aware.chunker import _Block


class _LlamaIndexEmbeddingAdapter(BaseEmbedding):
    """Expose the project's shared local embedder through LlamaIndex."""

    _embedder: TextEmbedder = PrivateAttr()

    def __init__(self, embedder: TextEmbedder) -> None:
        super().__init__(model_name=embedder.model_name)
        self._embedder = embedder

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._embedder.encode([text])[0].tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._embedder.encode(texts).tolist()

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)


class LlamaIndexSemanticChunker(CustomSemanticChunker):
    """LlamaIndex semantic splitter constrained to individual Docling sections."""

    strategy = "semantic-llamaindex-structure-constrained"

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
            embedder=embedder,
            min_chunk_size=min_chunk_size,
            target_chunk_size=target_chunk_size,
            max_chunk_size=max_chunk_size,
            breakpoint_percentile=breakpoint_percentile,
            buffer_size=buffer_size,
            encoding_name=encoding_name,
        )
        self.splitter = SemanticSplitterNodeParser(
            embed_model=_LlamaIndexEmbeddingAdapter(embedder),
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=int(breakpoint_percentile),
            include_metadata=False,
            include_prev_next_rel=False,
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
        blocks = self.extract_semantic_blocks(document)
        if not blocks:
            raise ValueError("Docling document contains no chunkable content")
        chunks: list[Chunk] = []
        for section_blocks in self._group_blocks_by_section(blocks):
            chunks.extend(
                self._chunk_section(
                    section_blocks,
                    document,
                    document_id,
                    len(chunks),
                    document_metadata,
                )
            )
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
            chunk.chunk_id = f"{document_id}-semantic-llamaindex-{index:04d}"
        return chunks

    def _chunk_section(
        self,
        blocks: Sequence[_Block],
        document: Mapping[str, Any],
        document_id: str,
        start_index: int,
        document_metadata: Mapping[str, Any] | None,
    ) -> list[Chunk]:
        section_path = blocks[0].section_path
        body = "\n\n".join(block.text for block in blocks)
        nodes = self.splitter.get_nodes_from_documents([Document(text=body)])
        texts = [node.get_content().strip() for node in nodes if node.get_content().strip()]
        sized_texts = self._enforce_size_limits(texts, section_path)
        page_numbers = sorted({page for block in blocks for page in block.page_numbers})
        source_refs = list(dict.fromkeys(ref for block in blocks for ref in block.source_refs if ref))
        chunks: list[Chunk] = []
        for offset, (node_text, fallback_token_split) in enumerate(sized_texts):
            prefix = self._section_prefix(section_path)
            rendered = f"{prefix}\n\n{node_text}" if prefix else node_text
            metadata = dict(document_metadata or {})
            metadata.update(
                {
                    "document_name": document.get("name"),
                    "embedding_model": self.embedder.model_name,
                    "library": "LlamaIndex SemanticSplitterNodeParser",
                    "breakpoint_percentile": self.breakpoint_percentile,
                    "buffer_size": self.buffer_size,
                    "fallback_token_split": fallback_token_split,
                    "fallback_reason": (
                        "llamaindex_chunk_exceeded_maximum" if fallback_token_split else None
                    ),
                    "section_boundary_enforced": True,
                }
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-semantic-llamaindex-{start_index + offset:04d}",
                    document_id=document_id,
                    chunk_index=start_index + offset,
                    text=rendered,
                    page_numbers=page_numbers,
                    section_path=list(section_path),
                    source_refs=source_refs,
                    strategy=self.strategy,
                    token_count=self._count_tokens(rendered),
                    metadata=metadata,
                )
            )
        return chunks

    def _enforce_size_limits(
        self,
        texts: Sequence[str],
        section_path: tuple[str, ...],
    ) -> list[tuple[str, bool]]:
        result: list[tuple[str, bool]] = []
        for text in texts:
            block = _Block(
                text=text,
                page_numbers=(),
                source_refs=(),
                section_path=section_path,
                element_type="text",
            )
            oversized = self._count_tokens(self._render_blocks([block])) > self.max_chunk_size
            result.extend((part.text, oversized) for part in self._split_block(block))
        merged: list[tuple[str, bool]] = []
        for text, fallback in result:
            if merged:
                candidate = f"{merged[-1][0]}\n\n{text}"
                prefix = self._section_prefix(section_path)
                rendered = f"{prefix}\n\n{candidate}" if prefix else candidate
                if self._count_tokens(rendered) <= self.max_chunk_size and self._count_tokens(
                    f"{prefix}\n\n{merged[-1][0]}" if prefix else merged[-1][0]
                ) < self.min_chunk_size:
                    merged[-1] = (candidate, merged[-1][1] or fallback)
                    continue
            merged.append((text, fallback))
        return merged
