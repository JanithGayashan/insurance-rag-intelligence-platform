from src.chunking.fixed import FixedTokenChunker


def test_fixed_token_chunking():
    text = (
        "Insurance policy provides coverage for "
        "the insured vehicle and related risks. "
    ) * 200

    chunker = FixedTokenChunker(
        chunk_size=100,
        overlap=20,
    )

    chunks = chunker.chunk_text(
        text=text,
        document_id="test-document",
    )

    assert len(chunks) > 1

    for chunk in chunks[:-1]:
        assert chunk.token_count == 100

    assert chunks[-1].token_count <= 100

    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.document_id == "test-document"
        assert chunk.strategy == "fixed"
        assert chunk.text


def test_fixed_token_chunk_overlap():
    text = (
        "Insurance policy provides coverage for "
        "the insured vehicle and related risks. "
    ) * 100

    chunker = FixedTokenChunker(
        chunk_size=100,
        overlap=20,
    )

    chunks = chunker.chunk_text(
        text=text,
        document_id="test-document",
    )

    encoding = chunker.encoding

    first_tokens = encoding.encode(
        chunks[0].text
    )

    second_tokens = encoding.encode(
        chunks[1].text
    )

    assert first_tokens[-20:] == second_tokens[:20]