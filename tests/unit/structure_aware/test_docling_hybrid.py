import json

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument

from src.chunking.structure_aware.comparison import compare_document
from src.chunking.structure_aware.docling_hybrid import (
    DoclingHybridStructureChunker,
)


def _document() -> dict:
    document = DoclingDocument(name="motor-policy")
    document.add_title("Motor Insurance Policy")
    document.add_heading("Coverage", level=1)
    document.add_text(
        DocItemLabel.TEXT,
        "Accidental damage is covered by this policy.",
    )
    document.add_heading("Exclusions", level=1)
    document.add_text(
        DocItemLabel.TEXT,
        "Wear and tear is not covered by this policy.",
    )
    return document.export_to_dict()


def test_docling_hybrid_chunker_preserves_native_headings():
    chunks = DoclingHybridStructureChunker(chunk_size=100).chunk_document(
        _document(),
        document_id="motor-policy",
        document_metadata={"insurer": "Example Insurance"},
    )

    assert len(chunks) == 2
    assert chunks[0].strategy == "structure-aware-docling-hybrid"
    assert chunks[0].section_path == ["Motor Insurance Policy", "Coverage"]
    assert chunks[0].text.startswith("Motor Insurance Policy\nCoverage")
    assert chunks[0].metadata["chunker"] == "docling.HybridChunker"
    assert chunks[0].metadata["insurer"] == "Example Insurance"
    assert chunks[0].source_refs == ["#/texts/2"]


def test_structure_aware_comparison_writes_both_actual_outputs(tmp_path):
    document_path = tmp_path / "policy" / "document.json"
    document_path.parent.mkdir()
    document_path.write_text(json.dumps(_document()), encoding="utf-8")

    result = compare_document(
        document_path=document_path,
        chunk_size=100,
        overlap=10,
    )
    output_directory = document_path.parent / "structure_aware_comparison"

    assert result["custom"]["chunk_count"] > 0
    assert result["docling_hybrid"]["chunk_count"] > 0
    assert (output_directory / "custom_chunks.json").exists()
    assert (output_directory / "docling_hybrid_chunks.json").exists()
    assert (output_directory / "comparison.json").exists()
    report = (output_directory / "comparison.md").read_text(encoding="utf-8")
    assert "Custom implementation: actual chunk samples" in report
    assert "Docling HybridChunker: actual chunk samples" in report


def test_docling_hybrid_chunker_validates_configuration():
    with pytest.raises(ValueError, match="chunk_size"):
        DoclingHybridStructureChunker(chunk_size=0)

    chunker = DoclingHybridStructureChunker()
    with pytest.raises(ValueError, match="document_id"):
        chunker.chunk_document(_document(), document_id=" ")


def test_hybrid_preserves_consecutive_context_and_does_not_modify_parser_output():
    from copy import deepcopy
    document = _document()
    document["texts"][0]["label"] = "section_header"
    document["texts"][0]["level"] = 1
    document["texts"][0]["text"] = "CLAUSE 8-PASSENGER RISK"
    document["texts"][1]["text"] = "(Applicable to Commercial Vehicles)"
    original = deepcopy(document)
    chunks = DoclingHybridStructureChunker(chunk_size=100).chunk_document(document, "policy")
    assert document == original
    assert chunks[0].section_path == ["CLAUSE 8-PASSENGER RISK", "(Applicable to Commercial Vehicles)"]
    assert "CLAUSE 8-PASSENGER RISK" in chunks[0].text


def test_hybrid_enforces_limit_after_contextualization():
    from types import SimpleNamespace
    from docling_core.transforms.chunker import DocChunk, DocMeta
    parsed = DoclingDocument.model_validate(_document())
    native = DocChunk(text="Covered damage. " * 40, meta=DocMeta(doc_items=[parsed.texts[2]], headings=["Coverage heading " * 15]))
    chunker = DoclingHybridStructureChunker(chunk_size=50)
    chunker.native_chunker = SimpleNamespace(chunk=lambda **kwargs: [native], contextualize=lambda chunk: "\n".join(chunk.meta.headings) + "\n" + chunk.text)
    chunks = chunker.chunk_document(_document(), "policy")
    assert len(chunks) > 1
    assert all(chunk.token_count <= 50 for chunk in chunks)
    assert all(chunk.metadata["fallback_reason"] == "contextualized-native-chunk-over-budget" for chunk in chunks)
    assert "".join(chunk.metadata["body_text"] for chunk in chunks) == native.text


def test_hybrid_preserves_original_list_markers():
    document = DoclingDocument(name="list-policy")
    document.add_heading("Coverage")
    group = document.add_list_group()
    document.add_list_item(text="Flood damage is covered.", enumerated=True, marker="a.", parent=group)
    chunks = DoclingHybridStructureChunker(chunk_size=100).chunk_document(document.export_to_dict(), "policy")
    assert any("a. Flood damage is covered." in chunk.text for chunk in chunks)


def test_hybrid_keeps_extracted_text_nested_in_pictures_and_section_boundaries():
    from src.chunking.structure_aware.evaluation import evaluate_chunks
    document = DoclingDocument(name="graphic-policy")
    document.add_heading("Coverage")
    document.add_text(DocItemLabel.TEXT, "Ordinary coverage text.")
    picture = document.add_picture()
    document.add_heading("Important notice", parent=picture)
    document.add_text(DocItemLabel.TEXT, "Read the terms before making a claim.", parent=picture)
    document.add_heading("Exclusions")
    document.add_text(DocItemLabel.TEXT, "War damage is excluded.")
    source = document.export_to_dict()
    chunks = DoclingHybridStructureChunker(chunk_size=100).chunk_document(source, "policy")
    assert any("Important notice" in chunk.text and "Read the terms" in chunk.text for chunk in chunks)
    quality = evaluate_chunks(source, chunks, 100)
    assert quality["missing_source_refs"] == []
    assert quality["missing_heading_refs"] == []
    assert quality["section_crossing_chunks"] == []


def test_repeated_heading_names_keep_distinct_provenance_and_do_not_merge_sections():
    from src.chunking.structure_aware.evaluation import evaluate_chunks
    document = DoclingDocument(name="repeated-sections")
    document.add_heading("Provided always that:")
    document.add_text(DocItemLabel.TEXT, "The first clause requires prior approval.")
    document.add_heading("Provided always that:")
    document.add_text(DocItemLabel.TEXT, "The second clause requires written notice.")
    source = document.export_to_dict()
    chunks = DoclingHybridStructureChunker(chunk_size=100).chunk_document(source, "policy")
    result = evaluate_chunks(source, chunks, 100)
    assert result["missing_source_refs"] == []
    assert result["missing_heading_refs"] == []
    assert result["section_crossing_chunks"] == []
    assert len(chunks) == 2
