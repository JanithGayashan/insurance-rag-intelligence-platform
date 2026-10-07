from __future__ import annotations

import spacy

from src.chunking.sentence_paragraph.base import (
    SentenceParagraphChunkerBase,
)


class SpacySentenceParagraphChunker(SentenceParagraphChunkerBase):
    """Sentence/paragraph chunking with spaCy's rule-based Sentencizer."""

    strategy = "sentence-paragraph-spacy"
    chunk_id_label = "sentence-paragraph-spacy"
    implementation = "spacy-sentencizer"

    def __init__(
        self,
        chunk_size: int = 500,
        encoding_name: str = "cl100k_base",
    ) -> None:
        super().__init__(chunk_size=chunk_size, encoding_name=encoding_name)
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    def _split_sentences(self, text: str) -> list[str]:
        sentences = [
            sentence.text.strip()
            for sentence in self.nlp(text).sents
            if sentence.text.strip()
        ]
        return sentences or [text.strip()]
