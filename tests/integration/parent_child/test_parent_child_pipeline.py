import json

from src.chunking.parent_child.pipeline import run_parent_child_pipeline


def _write_document(root, name, status):
    document_directory = root / name
    document_directory.mkdir(parents=True)
    text = "\n\n".join(
        " ".join(
            f"Insurance sentence {section}-{sentence} describes coverage."
            for sentence in range(12)
        )
        for section in range(5)
    )
    (document_directory / "document.txt").write_text(text, encoding="utf-8")
    docling_texts = [
        {
            "self_ref": "#/texts/0",
            "children": [],
            "label": "section_header",
            "text": "Coverage",
            "prov": [{"page_no": 1}],
        },
        {
            "self_ref": "#/texts/1",
            "children": [],
            "label": "text",
            "text": text,
            "prov": [{"page_no": 1}],
        },
    ]
    (document_directory / "document.json").write_text(
        json.dumps(
            {
                "name": name,
                "body": {
                    "self_ref": "#/body",
                    "children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}],
                },
                "texts": docling_texts,
                "groups": [],
                "tables": [],
                "pictures": [],
                "key_value_items": [],
                "form_items": [],
            }
        ),
        encoding="utf-8",
    )
    (document_directory / "validation_report.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    (document_directory / "metadata.json").write_text(
        json.dumps({"source": {"sha256": f"hash-{name}"}}), encoding="utf-8"
    )


def test_pipeline_processes_valid_documents_and_skips_failed_validation(tmp_path):
    _write_document(tmp_path, "accepted", "PASS")
    _write_document(tmp_path, "review", "REVIEW")
    _write_document(tmp_path, "rejected", "FAIL")

    result = run_parent_child_pipeline(
        tmp_path,
        parent_size=180,
        child_size=70,
        child_overlap=10,
    )

    assert result["total_documents"] == 3
    assert result["processed_documents"] == 2
    assert result["skipped_documents"] == 1
    assert result["failed_documents"] == 0
    assert result["corpus"]["custom"]["relationship_errors"] == 0
    assert result["corpus"]["llamaindex"]["relationship_errors"] == 0
    assert result["corpus"]["industry"]["relationship_errors"] == 0
    assert result["corpus"]["industry"]["children_crossing_sections"] == 0
    assert (tmp_path / "parent_child_summary.json").exists()
    assert (tmp_path / "parent_child_summary.md").exists()
    assert (
        tmp_path / "accepted" / "parent_child_comparison" / "comparison.md"
    ).exists()
