from src.chunking.models import Chunk


def test_chunk_creation():
    chunk = Chunk(
        chunk_id="test-001",
        document_id="document-001",
        chunk_index=0,
        text="This is a test chunk.",
        page_numbers=[1],
        section_path=["Preamble"],
        source_refs=["#/texts/11"],
        strategy="fixed",
        token_count=5,
    )

    assert chunk.chunk_id == "test-001"
    assert chunk.document_id == "document-001"
    assert chunk.chunk_index == 0
    assert chunk.text == "This is a test chunk."
    assert chunk.page_numbers == [1]
    assert chunk.section_path == ["Preamble"]
    assert chunk.source_refs == ["#/texts/11"]
    assert chunk.strategy == "fixed"
    assert chunk.token_count == 5