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
