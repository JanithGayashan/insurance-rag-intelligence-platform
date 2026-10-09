import json

from src.chunking.semantic.pipeline import run_semantic_chunking_pipeline
from tests.unit.semantic.test_semantic_chunkers import (
    DeterministicEmbedder,
    sample_document,
)


def _write_document(root, name, status):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "document.json").write_text(
        json.dumps(sample_document()), encoding="utf-8"
    )
    (directory / "validation_report.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    (directory / "metadata.json").write_text(
        json.dumps({"source": {"sha256": f"hash-{name}"}}), encoding="utf-8"
    )


def test_pipeline_writes_both_implementations_and_evaluations(tmp_path):
    _write_document(tmp_path, "accepted", "PASS")
    _write_document(tmp_path, "rejected", "FAIL")

    result = run_semantic_chunking_pipeline(
        tmp_path,
        embedder=DeterministicEmbedder(),
        min_chunk_size=8,
        target_chunk_size=20,
        max_chunk_size=45,
        breakpoint_percentile=75,
        buffer_size=1,
    )

    assert result["processed_documents"] == 1
    assert result["skipped_documents"] == 1
    assert result["failed_documents"] == 0
    assert result["corpus"]["custom"]["children_crossing_sections"] == 0
    assert result["corpus"]["llamaindex"]["oversized_chunks"] == 0
    output = tmp_path / "accepted" / "semantic_comparison"
    assert (output / "custom_semantic_chunks.json").exists()
    assert (output / "llamaindex_semantic_chunks.json").exists()
    assert (output / "custom_semantic_chunks.txt").exists()
    assert (output / "llamaindex_semantic_chunks.txt").exists()
    assert (output / "comparison.json").exists()
    assert (tmp_path / "semantic_chunking_summary.md").exists()
