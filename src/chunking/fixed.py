from __future__ import annotations

import tiktoken

from src.chunking.models import Chunk


class FixedTokenChunker:
    """
    Split text into fixed-size token chunks with overlap.
    """

    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0"
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative"
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(
            encoding_name
        )

    def chunk_text(
        self,
        text: str,
        document_id: str,
    ) -> list[Chunk]:
        """
        Split text into fixed-size token chunks.
        """

        tokens = self.encoding.encode(text)

        chunks: list[Chunk] = []

        step = self.chunk_size - self.overlap

        for start in range(
            0,
            len(tokens),
            step,
        ):
            end = min(
                start + self.chunk_size,
                len(tokens),
            )

            chunk_tokens = tokens[start:end]

            if not chunk_tokens:
                break

            chunk_text = self.encoding.decode(
                chunk_tokens
            )

            chunk_index = len(chunks)

            chunk = Chunk(
                chunk_id=(
                    f"{document_id}-chunk-{chunk_index:04d}"
                ),
                document_id=document_id,
                chunk_index=chunk_index,
                text=chunk_text,
                strategy="fixed",
                token_count=len(chunk_tokens),
            )

            chunks.append(chunk)

            if end == len(tokens):
                break

        return chunks