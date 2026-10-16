from src.ingestion.comparison.evaluation import edit_rate, evaluate_result, token_agreement


def output(**overrides):
    return {"success": True, "text": "Flood damage is covered.",
            "pages": [{"page_number": 1, "text": "Flood damage is covered."}],
            "elements": [{"type": "section_header", "text": "Coverage", "page_number": 1}],
            "tables": [{"page_number": 1, "rows": [["Excess", "Rs. 500"]]}], **overrides}


def test_diagnostics_do_not_become_accuracy_labels():
    result = evaluate_result(output(), [{"page_number": 1, "text": "", "has_images": True},
                                       {"page_number": 2, "text": "", "has_images": True}])
    assert result["diagnostics"]["page_coverage"] == 0.5
    assert result["diagnostics"]["missing_pages"] == [2]
    assert result["reference"]["word_error_rate"] is None
    assert result["reference"]["status"] == "NOT_PROVIDED"
    assert result["pages"][0]["native_text_agreement"] is None


def test_reference_checks_enforce_page_and_structure_not_just_text_presence():
    checks = [{"id": "a", "kind": "heading", "page_number": 1, "text": "Coverage"},
              {"id": "b", "kind": "passage", "page_number": 2, "text": "Flood damage"},
              {"id": "c", "kind": "table_cell", "page_number": 1, "text": "Rs. 500"},
              {"id": "d", "kind": "heading", "page_number": 1, "text": "Flood damage"}]
    result = evaluate_result(output(), [{"page_number": 1, "text": "", "has_images": False}], {"checks": checks})
    assert [row["matched"] for row in result["reference"]["checks"]] == [True, False, True, False]
    assert result["reference"]["by_kind"]["table_cell"]["coverage"] == 1


def test_word_and_character_error_rates_only_use_complete_reference_pages():
    result = evaluate_result(output(), [], {"full_pages": [{"page_number": 1, "text": "Flood damage is excluded."}]})
    assert result["reference"]["word_error_rate"] == 0.25
    assert result["reference"]["character_error_rate"] > 0
    assert edit_rate([], ["extra"]) is None
    assert edit_rate(["word"], ["word", "extra", "extra"]) == 2


def test_token_recall_is_bounded_with_duplicate_extraction():
    assert token_agreement("flood damage", "flood flood damage")["token_recall"] == 1
    assert token_agreement("flood damage", "flood flood damage")["token_precision"] == 2 / 3


def test_failed_parser_does_not_get_zero_accuracy():
    result = evaluate_result(output(success=False), [])
    assert result["status"] == "FAILED"
    assert result["diagnostics"] is None and result["reference"] is None


def test_markdown_table_formatting_is_not_counted_as_text_recognition_errors():
    data=output(pages=[{"page_number":1,"text":"| Benefit | Limit |\n|---|---|\n| Glass | 1500 |"}])
    reference={"full_pages":[{"page_number":1,"text":"Benefit Limit\nGlass 1500"}]}
    score=evaluate_result(data,[],reference)["reference"]
    assert score["word_error_rate"] == 0
    assert score["character_error_rate"] == 0
