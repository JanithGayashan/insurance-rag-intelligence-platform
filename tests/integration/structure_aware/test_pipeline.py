import json

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
