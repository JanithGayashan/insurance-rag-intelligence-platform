import json

import pytest

from src.chunking.structure_aware.chunker import (
    StructureAwareChunker,
    write_chunks,
)


def _text(
    index: int,
    text: str,
    label: str = "text",
    page: int = 1,
    **extra,
):
    return {
        "self_ref": f"#/texts/{index}",
        "children": [],
        "label": label,
        "text": text,
        "prov": [{"page_no": page}],
        **extra,
    }


def _document():
    texts = [
        _text(0, "Motor Insurance Policy", "title"),
        _text(1, "Section 1: Coverage", "section_header"),
        _text(2, "Damage caused by an accident is covered.", page=2),
        _text(3, "Fire damage is covered.", "list_item", page=2),
        _text(4, "Flood damage is covered.", "list_item", page=2),
        _text(5, "Confidential footer", "page_footer", page=2),
        _text(6, "1. Exclusions", "section_header", page=3),
        _text(7, "Wear and tear is not covered.", page=3),
    ]
    return {
        "name": "motor-policy",
        "body": {
            "self_ref": "#/body",
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"},
                {"$ref": "#/groups/0"},
                {"$ref": "#/tables/0"},
                {"$ref": "#/texts/5"},
                {"$ref": "#/texts/6"},
                {"$ref": "#/texts/7"},
            ],
        },
        "texts": texts,
        "groups": [
            {
                "self_ref": "#/groups/0",
                "label": "list",
                "children": [
                    {"$ref": "#/texts/3"},
                    {"$ref": "#/texts/4"},
                ],
            }
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "children": [],
                "prov": [{"page_no": 2}],
                "data": {
                    "grid": [
                        [{"text": "Benefit"}, {"text": "Limit"}],
                        [{"text": "Towing"}, {"text": "10,000"}],
                    ]
                },
            }
        ],
        "pictures": [],
        "key_value_items": [],
        "form_items": [],
    }


def test_structure_aware_chunking_preserves_document_structure():
    chunker = StructureAwareChunker(chunk_size=100, overlap=10)

    chunks = chunker.chunk_document(
        _document(),
        document_id="motor-policy",
        document_metadata={"insurer": "Example Insurance"},
    )

    assert chunks
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.token_count <= 100 for chunk in chunks)
    assert all(chunk.strategy == "structure-aware" for chunk in chunks)
    assert all("Confidential footer" not in chunk.text for chunk in chunks)

    coverage_chunks = [
        chunk
        for chunk in chunks
        if "Damage caused by an accident" in chunk.text
    ]
    assert len(coverage_chunks) == 1
    coverage = coverage_chunks[0]
    assert coverage.section_path == [
        "Motor Insurance Policy",
        "Section 1: Coverage",
    ]
    assert coverage.page_numbers == [2]
    assert "- Fire damage is covered." in coverage.text
    assert "#/texts/1" in coverage.source_refs
    assert coverage.metadata["insurer"] == "Example Insurance"


def test_tables_are_kept_as_standalone_chunks():
    chunks = StructureAwareChunker(chunk_size=100).chunk_document(
        _document(),
        document_id="motor-policy",
    )

    table_chunks = [
        chunk for chunk in chunks if chunk.metadata["contains_table"]
    ]
    assert len(table_chunks) == 1
    table = table_chunks[0]
    assert table.metadata["block_count"] == 1
    assert "Benefit | Limit" in table.text
    assert "Towing | 10,000" in table.text
    assert table.page_numbers == [2]
    assert "#/tables/0" in table.source_refs


def test_oversized_blocks_are_split_with_metadata():
    document = _document()
    document["texts"][2]["text"] = " ".join(
        f"coverage-{index}" for index in range(120)
    )
    chunker = StructureAwareChunker(chunk_size=40, overlap=5)

    chunks = chunker.chunk_document(document, document_id="motor-policy")
    split_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("oversized_block_split")
    ]

    assert len(split_chunks) > 1
    assert all(chunk.token_count <= 40 for chunk in chunks)
    assert [chunk.metadata["split_part"] for chunk in split_chunks] == list(
        range(len(split_chunks))
    )


def test_chunk_file_and_write_chunks(tmp_path):
    document_path = tmp_path / "policy" / "document.json"
    document_path.parent.mkdir()
    document_path.write_text(json.dumps(_document()), encoding="utf-8")
    chunker = StructureAwareChunker(chunk_size=100)

    chunks = chunker.chunk_file(document_path)
    output_path = tmp_path / "chunks.json"
    write_chunks(chunks, output_path)
    stored_chunks = json.loads(output_path.read_text(encoding="utf-8"))

    assert chunks[0].document_id == "policy"
    assert stored_chunks[0]["strategy"] == "structure-aware"
    assert stored_chunks[0]["chunk_id"].startswith("policy-structure-")


def test_structure_aware_chunker_validates_input():
    with pytest.raises(ValueError, match="chunk_size"):
        StructureAwareChunker(chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        StructureAwareChunker(chunk_size=20, overlap=20)

    chunker = StructureAwareChunker()
    with pytest.raises(ValueError, match="document_id"):
        chunker.chunk_document(_document(), document_id=" ")
    with pytest.raises(ValueError, match="body"):
        chunker.chunk_document({}, document_id="policy")
