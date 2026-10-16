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


def test_consecutive_same_level_headings_retain_clause_context():
    document = _document()
    document["texts"][0].update(label="section_header", level=1, text="CLAUSE 8-PASSENGER RISK")
    document["texts"][1].update(level=1, text="(Applicable to Commercial Vehicles)")
    chunks = StructureAwareChunker().chunk_document(document, "policy")
    coverage = next(chunk for chunk in chunks if "Damage caused" in chunk.text)
    assert coverage.section_path == ["CLAUSE 8-PASSENGER RISK", "(Applicable to Commercial Vehicles)"]
    assert "CLAUSE 8-PASSENGER RISK" in coverage.text
    assert coverage.metadata["heading_context_inferred"] is True
    assert {"#/texts/0", "#/texts/1"}.issubset(coverage.source_refs)


def test_original_numbered_and_lettered_list_markers_are_preserved():
    document = _document()
    document["texts"][3].update(marker="1.", enumerated=True)
    document["texts"][4].update(marker="a.", enumerated=True)
    chunks = StructureAwareChunker().chunk_document(document, "policy")
    text = "\n".join(chunk.text for chunk in chunks)
    assert "1. Fire damage is covered." in text
    assert "a. Flood damage is covered." in text
    assert any(chunk.metadata["list_markers"].get("#/texts/4") == "a." for chunk in chunks)


@pytest.mark.parametrize("overlap", [0, 5])
def test_unicode_fallback_uses_original_character_spans(overlap):
    document = _document()
    text = ("රක්ෂණය සහ හානි ආවරණය. " * 35).strip()
    document["texts"][2]["text"] = text
    chunker = StructureAwareChunker(chunk_size=36, overlap=overlap)
    chunks = chunker.chunk_document(document, "policy")
    pieces = [chunk for chunk in chunks if chunk.metadata.get("source_spans") and "#/texts/2" in chunk.source_refs]
    assert pieces
    covered = set()
    for chunk in pieces:
        assert "\ufffd" not in chunk.text
        assert chunk.token_count <= 36
        for span in chunk.metadata["source_spans"]:
            assert chunk.metadata["body_text"] == text[span["start"]:span["end"]]
            covered.update(range(span["start"], span["end"]))
    assert covered == set(range(len(text)))


@pytest.mark.parametrize("limit", [1, 4, 8])
def test_small_budgets_do_not_overflow_with_context(limit):
    document = _document()
    document["texts"][2]["text"] = "Motor insurance coverage. " * 12
    chunks = StructureAwareChunker(chunk_size=limit, overlap=0).chunk_document(document, "policy")
    assert chunks and all(chunk.token_count <= limit for chunk in chunks)


def test_heading_only_sections_stay_in_document_order():
    document = _document()
    document["texts"].append(_text(8, "Final heading without body", "section_header", level=1))
    document["body"]["children"].append({"$ref": "#/texts/8"})
    chunks = StructureAwareChunker().chunk_document(document, "policy")
    assert "Final heading without body" in chunks[-1].text


def test_impossible_unicode_budget_raises_without_corrupting_text():
    from src.chunking.structure_aware.text import split_text
    import tiktoken
    with pytest.raises(ValueError, match="Unicode"):
        split_text("ර", tiktoken.get_encoding("cl100k_base"), 1)
