import json

import pytest

from src.chunking.sentence_paragraph.comparison import compare_document
from src.chunking.sentence_paragraph.custom import (
    CustomSentenceParagraphChunker,
)
from src.chunking.sentence_paragraph.spacy import (
    SpacySentenceParagraphChunker,
)


TEST_TEXT = (
    "Policy Overview\n\n"
    "The policy covers accidental damage. It also covers fire damage.\n\n"
    "Exclusions\n\n"
    "Wear and tear is excluded. Mechanical breakdown is excluded."
)


@pytest.mark.parametrize(
    "chunker_class,strategy",
    [
        (CustomSentenceParagraphChunker, "sentence-paragraph-custom"),
        (SpacySentenceParagraphChunker, "sentence-paragraph-spacy"),
    ],
)
def test_sentence_paragraph_chunkers_keep_paragraph_boundaries(
    chunker_class,
    strategy,
):
    chunks = chunker_class(chunk_size=100).chunk_text(
        TEST_TEXT,
        document_id="policy",
    )

    assert len(chunks) == 4
    assert chunks[0].text == "Policy Overview"
    assert chunks[1].text == (
        "The policy covers accidental damage. It also covers fire damage."
    )
    assert chunks[1].metadata["sentence_count"] == 2
    assert [chunk.chunk_index for chunk in chunks] == list(range(4))
    assert all(chunk.strategy == strategy for chunk in chunks)
    assert all(chunk.token_count <= 100 for chunk in chunks)


@pytest.mark.parametrize(
    "chunker_class",
    [CustomSentenceParagraphChunker, SpacySentenceParagraphChunker],
)
def test_oversized_paragraphs_split_on_sentence_boundaries(chunker_class):
    text = " ".join(
        f"Coverage sentence number {index} applies."
        for index in range(30)
    )
    chunks = chunker_class(chunk_size=40).chunk_text(text, "policy")

    assert len(chunks) > 1
    assert all(chunk.token_count <= 40 for chunk in chunks)
    assert all(chunk.metadata["oversized_paragraph"] for chunk in chunks)
    assert all(chunk.text.rstrip().endswith("applies.") for chunk in chunks)


def test_comparison_writes_actual_outputs(tmp_path):
    document_path = tmp_path / "policy" / "document.txt"
    document_path.parent.mkdir()
    document_path.write_text(TEST_TEXT, encoding="utf-8")

    result = compare_document(document_path, chunk_size=100)
    output_directory = document_path.parent / "sentence_paragraph_comparison"

    assert result["custom"]["chunk_count"] == 4
    assert result["spacy"]["chunk_count"] == 4
    assert (output_directory / "custom_chunks.json").exists()
    assert (output_directory / "spacy_chunks.json").exists()
    assert (output_directory / "comparison.json").exists()
    assert (output_directory / "comparison.md").exists()
    stored = json.loads(
        (output_directory / "spacy_chunks.json").read_text(encoding="utf-8")
    )
    assert stored[0]["strategy"] == "sentence-paragraph-spacy"


@pytest.mark.parametrize(
    "chunker_class",
    [CustomSentenceParagraphChunker, SpacySentenceParagraphChunker],
)
def test_sentence_paragraph_chunkers_validate_input(chunker_class):
    with pytest.raises(ValueError, match="chunk_size"):
        chunker_class(chunk_size=0)

    chunker = chunker_class()
    with pytest.raises(ValueError, match="document_id"):
        chunker.chunk_text(TEST_TEXT, " ")
