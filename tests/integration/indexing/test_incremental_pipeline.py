import json
import sqlite3

import numpy as np
import pytest
from qdrant_client import QdrantClient
from filelock import FileLock, Timeout

from src.chunking.models import Chunk
from src.indexing.pipeline import run_embedding_and_indexing_pipeline
from src.indexing.qdrant_store import QdrantVectorStore
from src.retrieval.dense import DenseRetriever
from tests.unit.indexing.test_qdrant_index import KeywordEmbedder


class CountingEmbedder(KeywordEmbedder):
    def __init__(self):
        self.inputs = []
        self.fail = False

    def embed_documents(self, texts):
        if self.fail:
            raise RuntimeError("simulated embedding failure")
        self.inputs.extend(texts)
        return super().embed_documents(texts)


def write_document(root, name="motor", version="v1", texts=None, page=1):
    path = root / name / "structure_chunks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    texts = texts if texts is not None else ["Flood damage is covered.", "War damage is excluded."]
    chunks = [Chunk(
        chunk_id=f"{name}-{version}-{index}", document_id=version, chunk_index=index,
        text=text, strategy="structure-aware", token_count=len(text.split()),
        section_path=["Coverage"], page_numbers=[page], source_refs=[f"#/texts/{index}"],
        metadata={"source_file": f"{name}.pdf", "source_sha256": version, "validation_status": "PASS"},
    ) for index, text in enumerate(texts)]
    path.write_text(json.dumps([chunk.model_dump() for chunk in chunks]), encoding="utf-8")
    return path


@pytest.fixture
def setup(tmp_path):
    embedder = CountingEmbedder()
    store = QdrantVectorStore(QdrantClient(":memory:"), "incremental_test", embedder.model_name, embedder.dimension, local_mode=True)
    write_document(tmp_path)
    yield tmp_path, embedder, store
    store.close()


def run(setup, **kwargs):
    root, embedder, store = setup
    return run_embedding_and_indexing_pipeline(
        processed_directory=root, embedder=embedder, vector_store=store,
        example_questions=[], **kwargs,
    )


def test_unchanged_run_skips_embedding_and_writes(setup):
    first = run(setup)
    second = run(setup)
    assert len(setup[1].inputs) == 2
    assert first["incremental_updates"]["unique_texts_embedded"] == 2
    assert second["incremental_updates"]["points_unchanged"] == 2
    assert second["incremental_updates"]["points_upserted"] == 0
    assert second["incremental_updates"]["documents_unchanged"] == 1


def test_changed_policy_reuses_unchanged_text_and_removes_old_version(setup):
    root, embedder, store = setup
    run(setup)
    old_ids = set(store.snapshot())
    write_document(root, version="v2", texts=["Flood damage is covered.", "Hospital expenses are covered."])
    result = run(setup)
    assert len(embedder.inputs) == 3
    assert result["incremental_updates"]["cache_hits"] == 1
    assert result["incremental_updates"]["points_deleted"] == 2
    assert result["incremental_updates"]["documents_updated"] == 1
    assert old_ids.isdisjoint(store.snapshot())
    assert store.client.count(store.collection_name).count == 2


def test_metadata_only_update_does_not_reembed(setup):
    root, embedder, store = setup
    run(setup)
    write_document(root, page=9)
    result = run(setup)
    assert len(embedder.inputs) == 2
    assert result["incremental_updates"]["cache_hits"] == 2
    assert all(point["payload"]["page_numbers"] == [9] for point in store.snapshot().values())


def test_duplicate_text_is_embedded_once_and_keeps_both_sources(setup):
    root, embedder, store = setup
    write_document(root, name="other", texts=["Flood damage is covered."])
    result = run(setup)
    assert len(embedder.inputs) == 2
    assert result["chunks_indexed"] == 3
    assert {point["payload"]["source_file"] for point in store.snapshot().values()} == {"motor.pdf", "other.pdf"}


def test_removed_document_is_deleted_after_successful_sync(setup):
    root, embedder, store = setup
    path = write_document(root, name="other", texts=["Hospital expenses are covered."])
    run(setup)
    path.unlink()
    result = run(setup)
    assert result["incremental_updates"]["documents_removed"] == 1
    assert result["incremental_updates"]["points_deleted"] == 1
    assert len(embedder.inputs) == 3
    assert {point["payload"]["source_file"] for point in store.snapshot().values()} == {"motor.pdf"}


def test_failure_keeps_old_points_and_resumes_cached_batch(setup, monkeypatch):
    root, embedder, store = setup
    run(setup)
    old_ids = set(store.snapshot())
    write_document(root, version="v2", texts=["Flood claims require proof.", "Hospital claims require receipts."])
    original_upsert = store.upsert
    calls = 0

    def interrupted(records, vectors, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated index failure")
        return original_upsert(records, vectors, **kwargs)

    monkeypatch.setattr(store, "upsert", interrupted)
    with pytest.raises(RuntimeError, match="simulated index failure"):
        run(setup, batch_size=1)
    assert old_ids.issubset(store.snapshot())
    assert len(embedder.inputs) == 4
    with sqlite3.connect(root / ".indexing" / "state.sqlite3") as connection:
        run_id, status = connection.execute("SELECT run_id, status FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
    assert status == "FAILED"
    monkeypatch.setattr(store, "upsert", original_upsert)
    resumed = run(setup, batch_size=1)
    assert resumed["run_id"] == run_id
    assert resumed["incremental_updates"]["resumed_checkpoint_points"] == 1
    assert resumed["incremental_updates"]["unique_texts_embedded"] == 0
    assert resumed["incremental_updates"]["points_upserted"] == 1
    assert resumed["incremental_updates"]["points_deleted"] == 2
    assert len(embedder.inputs) == 4
    assert old_ids.isdisjoint(store.snapshot())


def test_deleted_index_point_is_restored_from_cache(setup):
    root, embedder, store = setup
    run(setup)
    store.delete_points([next(iter(store.snapshot()))])
    result = run(setup)
    assert len(embedder.inputs) == 2
    assert result["incremental_updates"]["points_upserted"] == 1
    assert result["incremental_updates"]["cache_hits"] == 1
    assert result["validation"]["status"] == "PASS"


def test_embedding_failure_does_not_remove_old_policy(setup):
    root, embedder, store = setup
    run(setup)
    old_ids = set(store.snapshot())
    write_document(root, version="v2", texts=["Hospital damage is covered."])
    embedder.fail = True
    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        run(setup)
    assert set(store.snapshot()) == old_ids


def test_source_change_cancels_cleanup(setup):
    root, embedder, store = setup
    run(setup)
    old_ids = set(store.snapshot())
    write_document(root, version="v2", texts=["Hospital damage is covered."])

    def change_source(progress):
        write_document(root, version="v3", texts=["Claim documents must include receipts."])

    with pytest.raises(RuntimeError, match="source corpus changed"):
        run(setup, progress_callback=change_source)
    assert old_ids.issubset(store.snapshot())
    result = run(setup)
    assert result["validation"]["status"] == "PASS"
    assert store.client.count(store.collection_name).count == 1


def test_empty_corpus_requires_explicit_cleanup_flag(setup):
    root, embedder, store = setup
    run(setup)
    (root / "motor" / "structure_chunks.json").unlink()
    with pytest.raises(FileNotFoundError):
        run(setup)
    assert store.client.count(store.collection_name).count == 2
    result = run(setup, allow_empty_corpus=True)
    assert result["chunks_indexed"] == 0
    assert result["incremental_updates"]["points_deleted"] == 2


def test_empty_artifact_cannot_accidentally_delete_index(setup):
    root, embedder, store = setup
    run(setup)
    (root / "motor" / "structure_chunks.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no chunks"):
        run(setup)
    assert store.client.count(store.collection_name).count == 2


def test_model_revision_mismatch_is_rejected_for_indexing_and_queries(setup):
    root, embedder, store = setup
    run(setup)
    old_ids = set(store.snapshot())
    embedder.configuration = embedder.configuration.model_copy(update={"model_revision": "test-v2"})
    with pytest.raises(ValueError, match="configuration differs"):
        run(setup)
    with pytest.raises(ValueError, match="configuration differs"):
        DenseRetriever(embedder, store)
    assert set(store.snapshot()) == old_ids


def test_new_model_collection_uses_new_cache_namespace(setup):
    root, embedder, store = setup
    run(setup)
    embedder.configuration = embedder.configuration.model_copy(update={"model_revision": "test-v2"})
    other = QdrantVectorStore(store.client, "other_model", embedder.model_name, embedder.dimension, local_mode=True)
    result = run_embedding_and_indexing_pipeline(processed_directory=root, embedder=embedder, vector_store=other, example_questions=[])
    assert len(embedder.inputs) == 4
    assert result["incremental_updates"]["unique_texts_embedded"] == 2


def test_rebuild_restores_entire_index_from_durable_cache(setup):
    run(setup)
    rebuilt = run(setup, recreate_collection=True)
    assert len(setup[1].inputs) == 2
    assert rebuilt["incremental_updates"]["cache_hits"] == 2
    assert rebuilt["incremental_updates"]["points_upserted"] == 2


def test_wrong_corpus_cannot_prune_existing_collection(setup, tmp_path):
    root, embedder, store = setup
    run(setup)
    other_root = tmp_path / "another-corpus"
    write_document(other_root, name="other")
    with pytest.raises(ValueError, match="different corpus"):
        run_embedding_and_indexing_pipeline(processed_directory=other_root, embedder=embedder, vector_store=store, example_questions=[])
    assert store.client.count(store.collection_name).count == 2


def test_persisted_qdrant_and_cache_survive_restart(tmp_path):
    write_document(tmp_path)
    embedder = CountingEmbedder()
    index_path = tmp_path / "qdrant"
    run_embedding_and_indexing_pipeline(
        processed_directory=tmp_path, local_qdrant_path=index_path,
        embedder=embedder, example_questions=[],
    )
    restarted = CountingEmbedder()
    result = run_embedding_and_indexing_pipeline(
        processed_directory=tmp_path, local_qdrant_path=index_path,
        embedder=restarted, example_questions=[],
    )
    assert restarted.inputs == []
    assert result["incremental_updates"]["points_unchanged"] == 2


def test_corrupted_cache_is_recomputed_when_index_needs_repair(setup):
    root, embedder, store = setup
    run(setup)
    store.delete_points(list(store.snapshot()))
    with sqlite3.connect(root / ".indexing" / "state.sqlite3") as connection:
        connection.execute("UPDATE embeddings SET vector = ?", (b"bad",))
    result = run(setup)
    assert len(embedder.inputs) == 4
    assert result["incremental_updates"]["unique_texts_embedded"] == 2
    assert result["validation"]["status"] == "PASS"


def test_second_writer_cannot_modify_index(setup):
    root, embedder, store = setup
    run(setup)
    old_ids = set(store.snapshot())
    with FileLock(str(root / ".indexing" / "state.sqlite3") + ".lock"):
        with pytest.raises(Timeout):
            run(setup)
    assert set(store.snapshot()) == old_ids


@pytest.mark.parametrize("bad_value", [0.0, np.nan, np.inf])
def test_invalid_vectors_fail_before_index_cleanup(setup, monkeypatch, bad_value):
    root, embedder, store = setup
    run(setup)
    old_ids = set(store.snapshot())
    write_document(root, version="v2", texts=["Hospital damage is covered."])
    monkeypatch.setattr(embedder, "embed_documents", lambda texts: np.full((len(texts), 6), bad_value, dtype=np.float32))
    with pytest.raises(ValueError):
        run(setup)
    assert set(store.snapshot()) == old_ids
