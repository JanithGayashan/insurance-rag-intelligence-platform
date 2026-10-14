import hashlib

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.chunking.models import Chunk
from src.indexing.configuration import EmbeddingConfiguration
from src.indexing.models import IndexRecord
from src.indexing.qdrant_store import QdrantVectorStore
from src.retrieval.dense import DenseRetriever


class KeywordEmbedder:
    model_name = "keyword-test-model"
    dimension = 6
    configuration = EmbeddingConfiguration(
        model_name=model_name, model_revision="test-v1", tokenizer_sha256="test-tokenizer-v1",
        dimension=dimension, max_sequence_length=512, backend="test",
    )

    def count_tokens(self, text, *, query=False):
        return len(text.split()) + 2

    def embed_documents(self, texts):
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, query):
        return self._vector(query)

    def _vector(self, text):
        lowered = text.casefold()
        vector = np.asarray(
            [
                lowered.count("flood"),
                lowered.count("damage"),
                lowered.count("claim"),
                lowered.count("hospital"),
                lowered.count("war"),
                lowered.count("exclude"),
            ],
            dtype=np.float32,
        )
        if not vector.any():
            vector[int(hashlib.sha256(text.encode()).hexdigest(), 16) % len(vector)] = 1
        return vector / np.linalg.norm(vector)


def _record(index, text, document="doc-a", section="Coverage"):
    chunk = Chunk(
        chunk_id=f"{document}-structure-{index:04d}",
        document_id=document,
        chunk_index=index,
        text=text,
        page_numbers=[index + 1],
        section_path=[section],
        source_refs=[f"#/texts/{index}"],
        strategy="structure-aware",
        token_count=10,
        metadata={
            "source_file": f"{document}.pdf",
            "source_sha256": document,
            "validation_status": "PASS",
        },
    )
    return IndexRecord.from_chunk(chunk)


def test_deterministic_point_id_and_content_hash():
    first = _record(0, "Flood damage is covered.")
    second = _record(0, "Flood damage is covered.")
    assert first.point_id == second.point_id
    assert first.content_sha256 == second.content_sha256


def test_index_validation_retrieval_and_metadata_filtering():
    embedder = KeywordEmbedder()
    records = [
        _record(0, "Flood damage is covered."),
        _record(1, "War damage is excluded.", section="Exclusions"),
        _record(0, "Hospital expenses are covered.", document="doc-b"),
    ]
    store = QdrantVectorStore(
        QdrantClient(":memory:"),
        "test_structure_chunks",
        embedder.model_name,
        embedder.dimension,
        local_mode=True,
    )
    store.ensure_collection(configuration=embedder.configuration)
    store.upsert(records, embedder.embed_documents([record.text for record in records]))

    assert store.validate(records)["status"] == "PASS"
    retriever = DenseRetriever(embedder, store)
    hits = retriever.retrieve("Is flood damage covered?", top_k=2)
    assert hits[0].chunk_id == records[0].chunk_id
    filtered = retriever.retrieve(
        "What is covered?", top_k=3, filters={"document_id": "doc-b"}
    )
    assert [hit.document_id for hit in filtered] == ["doc-b"]
    store.close()


def test_store_rejects_wrong_vector_shape():
    embedder = KeywordEmbedder()
    record = _record(0, "Flood damage is covered.")
    store = QdrantVectorStore(
        QdrantClient(":memory:"),
        "shape_test",
        embedder.model_name,
        6,
        local_mode=True,
    )
    store.ensure_collection()
    with pytest.raises(ValueError, match="vectors have shape"):
        store.upsert([record], np.zeros((1, 5), dtype=np.float32))
    store.close()
