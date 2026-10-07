from __future__ import annotations

import re

import tiktoken

from src.chunking.models import Chunk


class RecursiveChunker:
    """
    Custom recursive text chunker.

    Splitting priority:
        1. Paragraphs
        2. Sentences
        3. Words/whitespace
        4. Tokens

    This implementation is intentionally written from scratch
    so the recursive chunking algorithm can be understood.
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

        # Recursive splitting hierarchy.
        self.separators = [
            # Level 1: paragraphs
            re.compile(r"\n\s*\n+"),

            # Level 2: sentences
            # Avoid splitting simple numbered items such as "1."
            re.compile(
                r"(?<!\d[.!?])"
                r"(?<=[.!?])"
                r"\s+"
                r"(?=[A-Z(])"
            ),

            # Level 3: whitespace / words
            re.compile(r"\s+"),
        ]

    def _token_count(self, text: str) -> int:
        """
        Count tokens using the selected encoding.
        """

        return len(
            self.encoding.encode(text)
        )

    def _split_recursive(
        self,
        text: str,
        separator_index: int = 0,
    ) -> list[str]:
        """
        Recursively split text until each piece
        can fit within the chunk size.
        """

        text = text.strip()

        if not text:
            return []

        # Base condition:
        # The text already fits.
        if (
            self._token_count(text)
            <= self.chunk_size
        ):
            return [text]

        # No separators left.
        # Fall back to direct token splitting.
        if (
            separator_index
            >= len(self.separators)
        ):
            tokens = self.encoding.encode(
                text
            )

            pieces = []

            for start in range(
                0,
                len(tokens),
                self.chunk_size,
            ):
                token_slice = tokens[
                    start:
                    start + self.chunk_size
                ]

                pieces.append(
                    self.encoding.decode(
                        token_slice
                    )
                )

            return pieces

        separator = self.separators[
            separator_index
        ]

        parts = [
            part.strip()
            for part in separator.split(text)
            if part.strip()
        ]

        # Separator did not actually split the text.
        # Move to the next smaller level.
        if len(parts) <= 1:
            return self._split_recursive(
                text,
                separator_index + 1,
            )

        pieces: list[str] = []

        for part in parts:

            if (
                self._token_count(part)
                <= self.chunk_size
            ):
                pieces.append(part)

            else:
                # Part is still too large.
                # Recursively split it using
                # the next separator.
                pieces.extend(
                    self._split_recursive(
                        part,
                        separator_index + 1,
                    )
                )

        return pieces

    def _merge_pieces(
        self,
        pieces: list[str],
    ) -> list[str]:
        """
        Combine small pieces into chunks while
        respecting the maximum token size.

        Overlap is created using complete pieces
        whenever possible so we do not unnecessarily
        break sentence boundaries.
        """

        chunks: list[str] = []

        current_pieces: list[str] = []
        current_tokens = 0

        for piece in pieces:

            piece_tokens = self._token_count(
                piece
            )

            # First piece.
            if not current_pieces:

                current_pieces = [piece]
                current_tokens = piece_tokens

                continue

            # Piece still fits.
            if (
                current_tokens
                + piece_tokens
                <= self.chunk_size
            ):

                current_pieces.append(piece)
                current_tokens += piece_tokens

                continue

            # Current chunk is full.
            chunks.append(
                " ".join(current_pieces)
            )

            # Build overlap from the end of
            # the previous chunk.
            overlap_pieces: list[str] = []
            overlap_tokens = 0

            for previous_piece in reversed(
                current_pieces
            ):

                previous_tokens = (
                    self._token_count(
                        previous_piece
                    )
                )

                if (
                    overlap_tokens
                    + previous_tokens
                    > self.overlap
                ):
                    break

                overlap_pieces.append(
                    previous_piece
                )

                overlap_tokens += (
                    previous_tokens
                )

            overlap_pieces.reverse()

            current_pieces = overlap_pieces
            current_tokens = overlap_tokens

            # Add the new piece.
            if (
                current_tokens
                + piece_tokens
                <= self.chunk_size
            ):

                current_pieces.append(piece)
                current_tokens += piece_tokens

            else:
                chunks.append(
                    " ".join(current_pieces)
                )

                current_pieces = [piece]
                current_tokens = piece_tokens

        # Add final chunk.
        if current_pieces:

            chunks.append(
                " ".join(current_pieces)
            )

        return chunks

    def chunk_text(
        self,
        text: str,
        document_id: str,
    ) -> list[Chunk]:
        """
        Split text recursively and return
        project Chunk objects.
        """

        pieces = self._split_recursive(
            text
        )

        text_chunks = self._merge_pieces(
            pieces
        )

        chunks: list[Chunk] = []

        for index, text_chunk in enumerate(
            text_chunks
        ):

            token_count = self._token_count(
                text_chunk
            )

            chunk = Chunk(
                chunk_id=(
                    f"{document_id}"
                    f"-custom-recursive-"
                    f"{index:04d}"
                ),
                document_id=document_id,
                chunk_index=index,
                text=text_chunk,
                strategy="recursive-custom",
                token_count=token_count,
            )

            chunks.append(chunk)

        return chunks