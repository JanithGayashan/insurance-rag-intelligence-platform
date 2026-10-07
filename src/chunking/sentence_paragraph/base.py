from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

import tiktoken

from src.chunking.models import Chunk


class SentenceParagraphChunkerBase(ABC):
    """Keep paragraph boundaries and split oversized paragraphs by sentence."""

    strategy: str
    chunk_id_label: str
    implementation: str

    def __init__(
        self,
        chunk_size: int = 500,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        self.chunk_size = chunk_size
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk_file(
        self,
        document_path: Path,
        document_id: str | None = None,
    ) -> list[Chunk]:
        document_path = Path(document_path)
        text = document_path.read_text(encoding="utf-8")
        return self.chunk_text(
            text=text,
            document_id=document_id or document_path.parent.name,
        )

    def chunk_text(self, text: str, document_id: str) -> list[Chunk]:
        if not document_id.strip():
            raise ValueError("document_id cannot be empty")

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", text)
            if paragraph.strip()
        ]
        chunks: list[Chunk] = []

        for paragraph_index, paragraph in enumerate(paragraphs):
            paragraph_tokens = self._count_tokens(paragraph)
            if paragraph_tokens <= self.chunk_size:
                self._append_chunk(
                    chunks=chunks,
                    text=paragraph,
                    document_id=document_id,
                    paragraph_index=paragraph_index,
                    sentence_count=len(self._split_sentences(paragraph)),
                    oversized_paragraph=False,
                    oversized_sentence_split=False,
                )
                continue

            sentences = self._split_sentences(paragraph)
            pending: list[str] = []

            def flush_pending() -> None:
                if not pending:
                    return
                self._append_chunk(
                    chunks=chunks,
                    text=" ".join(pending),
                    document_id=document_id,
                    paragraph_index=paragraph_index,
                    sentence_count=len(pending),
                    oversized_paragraph=True,
                    oversized_sentence_split=False,
                )
                pending.clear()

            for sentence in sentences:
                if self._count_tokens(sentence) > self.chunk_size:
                    flush_pending()
                    for sentence_part in self._split_tokens(sentence):
                        self._append_chunk(
                            chunks=chunks,
                            text=sentence_part,
                            document_id=document_id,
                            paragraph_index=paragraph_index,
                            sentence_count=1,
                            oversized_paragraph=True,
                            oversized_sentence_split=True,
                        )
                    continue

                candidate = " ".join([*pending, sentence])
                if pending and self._count_tokens(candidate) > self.chunk_size:
                    flush_pending()
                pending.append(sentence)

            flush_pending()

        return chunks

    @abstractmethod
    def _split_sentences(self, text: str) -> list[str]:
        """Return sentence units for one paragraph."""

    def _split_tokens(self, text: str) -> list[str]:
        tokens = self.encoding.encode(text)
        return [
            self.encoding.decode(tokens[start : start + self.chunk_size])
            for start in range(0, len(tokens), self.chunk_size)
        ]

    def _append_chunk(
        self,
        chunks: list[Chunk],
        text: str,
        document_id: str,
        paragraph_index: int,
        sentence_count: int,
        oversized_paragraph: bool,
        oversized_sentence_split: bool,
    ) -> None:
        chunk_index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=(
                    f"{document_id}-{self.chunk_id_label}-{chunk_index:04d}"
                ),
                document_id=document_id,
                chunk_index=chunk_index,
                text=text,
                strategy=self.strategy,
                token_count=self._count_tokens(text),
                metadata={
                    "implementation": self.implementation,
                    "natural_boundary": "paragraph",
                    "paragraph_index": paragraph_index,
                    "sentence_count": sentence_count,
                    "oversized_paragraph": oversized_paragraph,
                    "oversized_sentence_split": oversized_sentence_split,
                },
            )
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
