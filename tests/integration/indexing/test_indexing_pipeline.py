import json

from qdrant_client import QdrantClient

from src.chunking.models import Chunk
from src.indexing.pipeline import run_embedding_and_indexing_pipeline
from src.indexing.qdrant_store import QdrantVectorStore
from tests.unit.indexing.test_qdrant_index import KeywordEmbedder


def _write_chunks(root):
    directory = root / "motor-policy"
    directory.mkdir(parents=True)
    chunks = [
        Chunk(
            chunk_id="hash-a-structure-0000",
            document_id="hash-a",
            chunk_index=0,
            text="Flood damage to the insured vehicle is covered.",
            page_numbers=[3],
            section_path=["Coverage"],
            source_refs=["#/texts/1"],
            strategy="structure-aware",
            token_count=12,
            metadata={
                "source_file": "motor-policy.pdf",
                "source_sha256": "hash-a",
                "validation_status": "PASS",
            },
        ),
        Chunk(
            chunk_id="hash-a-structure-0001",
            document_id="hash-a",
            chunk_index=1,
            text="War damage is excluded from the policy.",
            page_numbers=[8],
            section_path=["Exclusions"],
            source_refs=["#/texts/2"],
            strategy="structure-aware",
            token_count=10,
            metadata={
                "source_file": "motor-policy.pdf",
                "source_sha256": "hash-a",
                "validation_status": "PASS",
            },
        ),
    ]
    (directory / "structure_chunks.json").write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in chunks]),
        encoding="utf-8",
    )


def test_complete_embedding_indexing_and_retrieval_pipeline(tmp_path):
    _write_chunks(tmp_path)
    embedder = KeywordEmbedder()
    store = QdrantVectorStore(
        QdrantClient(":memory:"),
        "pipeline_test",
        embedder.model_name,
        embedder.dimension,
        local_mode=True,
    )
    result = run_embedding_and_indexing_pipeline(
        processed_directory=tmp_path,
        embedder=embedder,
        vector_store=store,
        example_questions=["Is flood damage covered?"],
        top_k=2,
    )

    assert result["documents_indexed"] == 1
    assert result["chunks_indexed"] == 2
    assert result["vectors_normalized"] is True
    assert result["validation"]["status"] == "PASS"
    assert result["example_queries"][0]["results"][0]["section_path"] == [
        "Coverage"
    ]
    assert (tmp_path / "dense_indexing_summary.json").exists()
    assert (tmp_path / "dense_indexing_summary.md").exists()
    store.close()
