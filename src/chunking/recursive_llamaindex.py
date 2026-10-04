from __future__ import annotations

import tiktoken

from llama_index.core.node_parser import (
    SentenceSplitter,
)

from src.chunking.models import Chunk


class LlamaIndexRecursiveChunker:
    """
    Recursive chunker using LlamaIndex's
    SentenceSplitter.

    LlamaIndex handles the recursive splitting,
    sentence preservation, merging, and overlap.
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
                "overlap must be smaller than "
                "chunk_size"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

        # Explicitly select the tokenizer so
        # this experiment uses the same tokenizer
        # as the custom implementation.
        self.encoding = tiktoken.get_encoding(
            encoding_name
        )

        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            tokenizer=self.encoding.encode,
            paragraph_separator="\n\n",
        )

    def chunk_text(
        self,
        text: str,
        document_id: str,
    ) -> list[Chunk]:
        """
        Split text using LlamaIndex SentenceSplitter.
        """

        text_chunks = (
            self.splitter.split_text(text)
        )

        chunks: list[Chunk] = []

        for index, text_chunk in enumerate(
            text_chunks
        ):

            token_count = len(
                self.encoding.encode(
                    text_chunk
                )
            )

            chunk = Chunk(
                chunk_id=(
                    f"{document_id}"
                    f"-llama-recursive-"
                    f"{index:04d}"
                ),
                document_id=document_id,
                chunk_index=index,
                text=text_chunk,
                strategy="recursive-llamaindex",
                token_count=token_count,
            )

            chunks.append(chunk)

        return chunks