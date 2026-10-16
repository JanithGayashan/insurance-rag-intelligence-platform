import json
from pathlib import Path

from src.chunking.structure_aware.pipeline import (
    discover_documents,
    run_structure_chunking_pipeline,
)


def _write_document(directory, status):
    directory.mkdir(parents=True)
    document = {
        "name": directory.name,
        "body": {
            "self_ref": "#/body",
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
            ],
        },
        "texts": [
            {
                "self_ref": "#/texts/0",
                "children": [],
                "label": "section_header",
                "level": 1,
                "text": "Coverage",
                "prov": [{"page_no": 1}],
            },
            {
                "self_ref": "#/texts/1",
                "children": [],
                "label": "text",
                "text": "Accidental damage is covered.",
                "prov": [{"page_no": 1}],
            },
        ],
        "groups": [],
        "pictures": [],
        "tables": [],
        "key_value_items": [],
        "form_items": [],
    }
    (directory / "document.json").write_text(
        json.dumps(document),
        encoding="utf-8",
    )
    (directory / "validation_report.json").write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "source": {
                    "relative_path": f"{directory.name}.pdf",
                    "sha256": f"sha-{directory.name}",
                }
            }
        ),
        encoding="utf-8",
    )


def test_structure_chunking_pipeline_processes_validated_documents(tmp_path):
    processed_directory = tmp_path / "processed"
    _write_document(processed_directory / "pass-policy", "PASS")
    _write_document(processed_directory / "review-policy", "REVIEW")
    _write_document(processed_directory / "failed-policy", "FAIL")

    assert len(discover_documents(processed_directory)) == 3
    summary = run_structure_chunking_pipeline(processed_directory)

    assert summary["total_documents"] == 3
    assert summary["processed_documents"] == 2
    assert summary["skipped_documents"] == 1
    assert summary["failed_documents"] == 0
    assert summary["total_chunks"] == 2

    pass_output = processed_directory / "pass-policy" / "structure_chunks.json"
    review_output = (
        processed_directory / "review-policy" / "structure_chunks.json"
    )
    failed_output = (
        processed_directory / "failed-policy" / "structure_chunks.json"
    )
    assert pass_output.exists()
    assert review_output.exists()
    assert not failed_output.exists()

    chunks = json.loads(pass_output.read_text(encoding="utf-8"))
    assert chunks[0]["document_id"] == "sha-pass-policy"
    assert chunks[0]["metadata"]["validation_status"] == "PASS"
    assert chunks[0]["page_numbers"] == [1]
    assert (
        processed_directory / "structure_chunking_summary.json"
    ).exists()


def test_structure_pipeline_discovers_nested_categories_and_preserves_classification(tmp_path):
    sources = ["motor/book.pdf", "general/travel/nested/book.pdf", "general/health/book.pdf"]
    for index, source in enumerate(sources):
        directory = tmp_path / Path(source).with_suffix("")
        _write_document(directory, "PASS")
        (directory / "metadata.json").write_text(json.dumps({"source": {
            "relative_path": source, "sha256": str(index) * 64,
        }}), encoding="utf-8")
    summary = run_structure_chunking_pipeline(tmp_path)
    assert summary["processed_documents"] == 3 and summary["failed_documents"] == 0
    for source in sources:
        directory = tmp_path / Path(source).with_suffix("")
        chunk = json.loads((directory / "structure_chunks.json").read_text(encoding="utf-8"))[0]
        assert chunk["metadata"]["insurer"] == "slic"
        assert chunk["metadata"]["source_file"] == source
        expected_category = source.split("/")[1] if source.startswith("general/") else "motor"
        assert chunk["metadata"]["category_key"] == f"general/{expected_category}"


def test_failed_validation_removes_active_chunks_and_preserves_history(tmp_path):
    from src.indexing.loader import load_structure_records, discover_structure_chunk_files
    directory = tmp_path / "motor" / "policy"
    _write_document(directory, "PASS")
    run_structure_chunking_pipeline(tmp_path)
    original = (directory / "structure_chunks.json").read_bytes()
    (directory / "validation_report.json").write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    summary = run_structure_chunking_pipeline(tmp_path)
    assert summary["skipped_documents"] == 1
    assert not (directory / "structure_chunks.json").exists()
    assert any(path.read_bytes() == original for path in (directory / ".chunk_history").rglob("structure_chunks.json"))
    assert discover_structure_chunk_files(tmp_path) == []
    assert load_structure_records(tmp_path, allow_empty=True)[0] == []


def test_index_loader_rejects_changed_validation_and_parsed_input(tmp_path):
    import pytest
    from src.indexing.loader import load_structure_records
    directory = tmp_path / "policy"
    _write_document(directory, "PASS")
    run_structure_chunking_pipeline(tmp_path)
    (directory / "validation_report.json").write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        load_structure_records(tmp_path)
    (directory / "validation_report.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    document = json.loads((directory / "document.json").read_text(encoding="utf-8"))
    document["texts"][1]["text"] = "The policy terms changed."
    (directory / "document.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="parsed document changed"):
        load_structure_records(tmp_path)


def test_chunking_failure_invalidates_previous_output(tmp_path):
    directory = tmp_path / "policy"
    _write_document(directory, "PASS")
    run_structure_chunking_pipeline(tmp_path)
    (directory / "document.json").write_text(json.dumps({"name": "broken"}), encoding="utf-8")
    result = run_structure_chunking_pipeline(tmp_path)
    assert result["failed_documents"] == 1
    assert not (directory / "structure_chunks.json").exists()
