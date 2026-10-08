from __future__ import annotations

from pathlib import Path

import tiktoken

from src.chunking.parent_child.models import (
    HierarchicalChunk,
    ParentChildHierarchy,
)
from src.chunking.parent_child.validation import validate_configuration
from src.chunking.recursive.custom import RecursiveChunker


class CustomParentChildChunker:
    """Build a two-level hierarchy with the project's recursive splitter."""

    strategy = "parent-child-custom"

    def __init__(
        self,
        parent_size: int = 1500,
        child_size: int = 300,
        child_overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        validate_configuration(parent_size, child_size, child_overlap)
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.parent_splitter = RecursiveChunker(
            chunk_size=parent_size,
            overlap=0,
            encoding_name=encoding_name,
        )
        self.child_splitter = RecursiveChunker(
            chunk_size=child_size,
            overlap=child_overlap,
            encoding_name=encoding_name,
        )

    def chunk_text(self, text: str, document_id: str) -> ParentChildHierarchy:
        text = text.strip()
        document_id = document_id.strip()
        if not text:
            raise ValueError("text cannot be empty")
        if not document_id:
            raise ValueError("document_id cannot be empty")

        parent_parts = self._bounded_texts(
            [
                chunk.text
                for chunk in self.parent_splitter.chunk_text(text, document_id)
            ],
            self.parent_size,
        )
        parents: list[HierarchicalChunk] = []
        children: list[HierarchicalChunk] = []

        for parent_index, parent_text in enumerate(parent_parts):
            parent_id = f"{document_id}-custom-parent-{parent_index:04d}"
            child_parts = self._bounded_texts(
                [
                    chunk.text
                    for chunk in self.child_splitter.chunk_text(
                        parent_text, document_id
                    )
                ],
                self.child_size,
            )
            child_ids: list[str] = []
            for child_index_in_parent, child_text in enumerate(child_parts):
                child_index = len(children)
                child_id = f"{document_id}-custom-child-{child_index:04d}"
                child_ids.append(child_id)
                children.append(
                    HierarchicalChunk(
                        chunk_id=child_id,
                        document_id=document_id,
                        chunk_index=child_index,
                        level=1,
                        text=child_text,
                        token_count=len(self.encoding.encode(child_text)),
                        parent_id=parent_id,
                        is_leaf=True,
                        strategy=self.strategy,
                        metadata={
                            "parent_index": parent_index,
                            "child_index_in_parent": child_index_in_parent,
                            "splitter": "custom-recursive",
                        },
                    )
                )

            parents.append(
                HierarchicalChunk(
                    chunk_id=parent_id,
                    document_id=document_id,
                    chunk_index=parent_index,
                    level=0,
                    text=parent_text,
                    token_count=len(self.encoding.encode(parent_text)),
                    children_ids=child_ids,
                    is_leaf=False,
                    strategy=self.strategy,
                    metadata={"splitter": "custom-recursive"},
                )
            )

        return ParentChildHierarchy(
            document_id=document_id,
            strategy=self.strategy,
            parent_size=self.parent_size,
            child_size=self.child_size,
            child_overlap=self.child_overlap,
            parents=parents,
            children=children,
            metadata={"tokenizer": self.encoding_name, "parent_overlap": 0},
        )

    def _bounded_texts(self, texts: list[str], limit: int) -> list[str]:
        """Guarantee the public token limit after separators are rejoined."""

        bounded: list[str] = []
        for text in texts:
            tokens = self.encoding.encode(text)
            if len(tokens) <= limit:
                bounded.append(text)
                continue
            bounded.extend(
                self.encoding.decode(tokens[start : start + limit]).strip()
                for start in range(0, len(tokens), limit)
                if tokens[start : start + limit]
            )
        return [text for text in bounded if text]

    def chunk_file(
        self,
        document_path: Path,
        document_id: str | None = None,
    ) -> ParentChildHierarchy:
        document_path = Path(document_path)
        return self.chunk_text(
            document_path.read_text(encoding="utf-8"),
            document_id or document_path.parent.name,
        )
