from src.chunking.structure_aware.chunker import StructureAwareChunker
from src.chunking.structure_aware.evaluation import evaluate_chunks
from tests.unit.structure_aware.test_chunker import _document


def test_diagnostics_detect_missing_text_markers_and_unknown_references():
    document = _document()
    document["texts"][3]["marker"] = "a."
    chunks = StructureAwareChunker().chunk_document(document, "policy")
    baseline = evaluate_chunks(document, chunks, 500)
    assert baseline["status"] == "PASS"
    chunks[0].text = "\ufffd Lost content."
    chunks[0].source_refs.append("#/texts/9999")
    chunks[0].token_count = 900
    result = evaluate_chunks(document, chunks, 500)
    assert result["status"] == "REVIEW"
    assert result["missing_source_words"] > 0
    assert result["missing_list_marker_refs"] == ["#/texts/3"]
    assert result["invalid_source_refs"] == ["#/texts/9999"]
    assert result["replacement_character_chunks"]
    assert result["oversized_chunks"]
    assert result["token_count_mismatches"]


def test_empty_output_and_section_crossings_are_not_reported_as_success():
    document = _document()
    missing = evaluate_chunks(document, [], 500)
    assert missing["status"] == "REVIEW"
    assert missing["source_element_coverage"] == 0
    chunks = StructureAwareChunker().chunk_document(document, "policy")
    chunks[0].metadata["body_source_refs"].append("#/texts/7")
    assert evaluate_chunks(document, chunks, 500)["section_crossing_chunks"] == [chunks[0].chunk_id]
