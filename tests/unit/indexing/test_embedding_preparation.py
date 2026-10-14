import pytest

from src.indexing.embeddings import DenseEmbeddingService
from src.indexing.preparation import prepare_embedding_records
from tests.unit.indexing.test_qdrant_index import KeywordEmbedder, _record


def test_fallback_preserves_all_characters_and_section_provenance():
    embedder = KeywordEmbedder()
    embedder.configuration = embedder.configuration.model_copy(update={"max_sequence_length": 16})
    text = "Section: Coverage\n\n" + "Flood damage is covered. Claims need receipts.\n\n" * 12
    source = _record(0, text)
    parts, audit = prepare_embedding_records([source], embedder)
    assert len(parts) > 1
    reconstructed = []
    previous_end = 0
    for part in parts:
        start, end = part.metadata["source_character_start"], part.metadata["source_character_end"]
        assert start == previous_end
        original_slice = source.text[start:end]
        assert part.text == ("Section: Coverage\n\n" if start else "") + original_slice
        reconstructed.append(original_slice)
        previous_end = end
        assert embedder.count_tokens(part.text) <= 16
        assert part.parent_chunk_id == source.chunk_id
        assert part.section_path == source.section_path
        assert part.source_refs == source.source_refs
        assert part.page_numbers == source.page_numbers
        assert part.metadata["embedding_fallback"] is True
    assert "".join(reconstructed) == source.text
    assert audit["truncated_inputs"] == 0
    assert len({part.point_id for part in parts}) == len(parts)


def test_fallback_limits_punctuation_tokens_even_when_word_count_is_small():
    class PunctuationEmbedder(KeywordEmbedder):
        def count_tokens(self, text, *, query=False):
            return len(text) + 2

    embedder = PunctuationEmbedder()
    embedder.configuration = embedder.configuration.model_copy(update={"max_sequence_length": 32})
    parts, _ = prepare_embedding_records([_record(0, "." * 130)], embedder)
    original = "." * 130
    assert "".join(original[part.metadata["source_character_start"]:part.metadata["source_character_end"]] for part in parts) == original
    assert all(part.embedding_token_count <= 32 for part in parts)


def test_oversized_heading_fails_instead_of_silently_truncating():
    embedder = KeywordEmbedder()
    embedder.configuration = embedder.configuration.model_copy(update={"max_sequence_length": 8})
    source = _record(0, "Section: " + "heading " * 10 + "\n\nFlood damage.")
    with pytest.raises(ValueError, match="heading"):
        prepare_embedding_records([source], embedder)


def test_embedding_service_rejects_overlong_documents_and_queries_before_inference():
    # Exercise the real service's length guard without downloading model weights.
    service = DenseEmbeddingService.__new__(DenseEmbeddingService)
    service.configuration = KeywordEmbedder.configuration.model_copy(update={"max_sequence_length": 8})
    service._dimension = 6
    service.count_tokens = lambda text, query=False: len(text.split()) + 2

    class Model:
        def encode(self, texts, **kwargs):
            pytest.fail("overlong input must not reach inference")

    service.model = Model()
    with pytest.raises(ValueError, match="silent truncation"):
        service.embed_documents(["word " * 20])
    with pytest.raises(ValueError, match="silent truncation"):
        service.embed_query("word " * 20)
