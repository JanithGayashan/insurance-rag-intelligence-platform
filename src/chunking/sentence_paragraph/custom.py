from __future__ import annotations

import re

from src.chunking.sentence_paragraph.base import (
    SentenceParagraphChunkerBase,
)


class CustomSentenceParagraphChunker(SentenceParagraphChunkerBase):
    """Sentence/paragraph chunking with a local regex sentence detector."""

    strategy = "sentence-paragraph-custom"
    chunk_id_label = "sentence-paragraph-custom"
    implementation = "custom-regex"

    _sentence_boundary = re.compile(
        r"(?<=[.!?])\s+(?=(?:[\"'“‘(]*[A-Z0-9]))"
    )

    def _split_sentences(self, text: str) -> list[str]:
        sentences = [
            sentence.strip()
            for sentence in self._sentence_boundary.split(text)
            if sentence.strip()
        ]
        return sentences or [text.strip()]
