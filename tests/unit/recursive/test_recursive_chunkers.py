from src.chunking.recursive.custom import (
    RecursiveChunker,
)
from src.chunking.recursive.llamaindex import (
    LlamaIndexRecursiveChunker,
)


TEST_TEXT = (
    "Insurance policies provide protection against "
    "different financial risks. The insured must "
    "understand the terms and conditions of the policy. "
    "Coverage is subject to exclusions and limitations. "
    "Claims must be submitted according to the required "
    "procedure. The insurer may request supporting "
    "documents when processing a claim. "
) * 100


def test_custom_recursive_chunking():

    chunker = RecursiveChunker(
        chunk_size=300,
        overlap=50,
    )

    chunks = chunker.chunk_text(
        text=TEST_TEXT,
        document_id="test-document",
    )

    assert len(chunks) > 1

    for index, chunk in enumerate(chunks):

        assert chunk.chunk_index == index
        assert chunk.document_id == "test-document"
        assert chunk.strategy == "recursive-custom"
        assert chunk.text
        assert chunk.token_count <= 300


def test_llamaindex_recursive_chunking():

    chunker = LlamaIndexRecursiveChunker(
        chunk_size=300,
        overlap=50,
    )

    chunks = chunker.chunk_text(
        text=TEST_TEXT,
        document_id="test-document",
    )

    assert len(chunks) > 1

    for index, chunk in enumerate(chunks):

        assert chunk.chunk_index == index
        assert chunk.document_id == "test-document"
        assert chunk.strategy == "recursive-llamaindex"
        assert chunk.text
        assert chunk.token_count <= 300
