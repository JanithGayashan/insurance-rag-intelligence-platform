from types import SimpleNamespace

import pytest

from src.ingestion.validator import (
    ValidationConfig,
    calculate_char_ratio,
    calculate_replacement_char_ratio,
    calculate_token_coverage,
    determine_document_status,
    determine_document_type,
    get_docling_table_counts,
    normalize_text,
    profile_pdfplumber_page,
    safe_mean,
    tokenize,
    validate_page,
)


def make_reference_profile(
    *,
    page_type: str = "DIGITAL",
    usable_text_layer: bool = True,
    extracted_text: str = "Insurance policy coverage details.",
    pdf_character_objects: int = 100,
    image_count: int = 0,
    largest_image_coverage: float = 0.0,
    total_image_coverage: float = 0.0,
) -> dict:
    """
    Build a reference profile with the same structure
    returned by profile_pdfplumber_page().
    """

    return {
        "page_type": page_type,
        "usable_text_layer": usable_text_layer,
        "extracted_text": extracted_text,
        "text_characters": len(extracted_text),
        "pdf_character_objects": pdf_character_objects,
        "image_count": image_count,
        "largest_image_coverage": largest_image_coverage,
        "total_image_coverage": total_image_coverage,
    }


# ============================================================
# TEXT NORMALIZATION
# ============================================================


def test_normalize_text():
    text = "  Insurance   POLICY\nCoverage  "

    result = normalize_text(text)

    assert result == "insurance policy coverage"


def test_normalize_text_unicode():
    text = "ＡＢＣ Insurance"

    result = normalize_text(text)

    assert result == "abc insurance"


# ============================================================
# TOKENIZATION
# ============================================================


def test_tokenize():
    text = "Insurance Policy 123/ABC"

    result = tokenize(text)

    assert result == [
        "insurance",
        "policy",
        "123/abc",
    ]


# ============================================================
# TOKEN COVERAGE
# ============================================================


def test_calculate_token_coverage_full_match():
    result = calculate_token_coverage(
        "insurance policy coverage",
        "insurance policy coverage",
    )

    assert result == 1.0


def test_calculate_token_coverage_partial_match():
    result = calculate_token_coverage(
        "insurance policy",
        "insurance policy coverage",
    )

    assert result == pytest.approx(
        2 / 3
    )


def test_calculate_token_coverage_empty_reference():
    result = calculate_token_coverage(
        "insurance policy",
        "",
    )

    assert result is None


# ============================================================
# CHARACTER RATIO
# ============================================================


def test_calculate_char_ratio():
    result = calculate_char_ratio(
        "abcd",
        "abcdef",
    )

    assert result == pytest.approx(
        4 / 6
    )


def test_calculate_char_ratio_empty():
    result = calculate_char_ratio(
        "",
        "",
    )

    assert result is None


# ============================================================
# REPLACEMENT CHARACTER RATIO
# ============================================================


def test_calculate_replacement_char_ratio():
    result = calculate_replacement_char_ratio(
        "abc\ufffd"
    )

    assert result == pytest.approx(
        1 / 4
    )


def test_calculate_replacement_char_ratio_empty():
    result = calculate_replacement_char_ratio("")

    assert result == 0.0


# ============================================================
# DOCLING TABLE COUNTS
# ============================================================


def test_get_docling_table_counts():
    table_page_2 = SimpleNamespace(
        prov=[
            SimpleNamespace(page_no=2)
        ]
    )

    table_page_4 = SimpleNamespace(
        prov=[
            SimpleNamespace(page_no=4)
        ]
    )

    table_without_provenance = SimpleNamespace(
        prov=None
    )

    document = SimpleNamespace(
        tables=[
            table_page_2,
            table_page_2,
            table_page_4,
            table_without_provenance,
        ]
    )

    table_counts, unassigned = (
        get_docling_table_counts(
            document
        )
    )

    assert table_counts == {
        2: 2,
        4: 1,
    }

    assert unassigned == 1


# ============================================================
# PDF PAGE PROFILING
# ============================================================


class FakePage:
    def __init__(
        self,
        text: str,
        character_count: int,
        images: list[dict],
        width: float = 100.0,
        height: float = 100.0,
    ):
        self._text = text
        self.chars = [
            {}
            for _ in range(character_count)
        ]
        self.images = images
        self.width = width
        self.height = height

    def dedupe_chars(self):
        return self

    def extract_text(self):
        return self._text


def test_profile_pdfplumber_page_digital():
    page = FakePage(
        text="A" * 100,
        character_count=100,
        images=[],
    )

    config = ValidationConfig()

    result = profile_pdfplumber_page(
        page,
        config,
    )

    assert result["page_type"] == "DIGITAL"
    assert result["usable_text_layer"] is True
    assert result["image_count"] == 0


def test_profile_pdfplumber_page_scanned():
    page = FakePage(
        text="",
        character_count=0,
        images=[
            {
                "x0": 0,
                "x1": 100,
                "top": 0,
                "bottom": 100,
            }
        ],
    )

    config = ValidationConfig()

    result = profile_pdfplumber_page(
        page,
        config,
    )

    assert (
        result["page_type"]
        == "SCANNED_OR_IMAGE_BASED"
    )

    assert result["usable_text_layer"] is False
    assert result["largest_image_coverage"] == 1.0


def test_profile_pdfplumber_page_ocr_text_over_image():
    page = FakePage(
        text="A" * 100,
        character_count=100,
        images=[
            {
                "x0": 0,
                "x1": 100,
                "top": 0,
                "bottom": 100,
            }
        ],
    )

    config = ValidationConfig()

    result = profile_pdfplumber_page(
        page,
        config,
    )

    assert (
        result["page_type"]
        == "OCR_TEXT_OVER_IMAGE"
    )

    assert result["usable_text_layer"] is True


# ============================================================
# PAGE VALIDATION
# ============================================================


def test_validate_page_pass():
    config = ValidationConfig()

    text = (
        "Insurance policy coverage details."
    )

    profile = make_reference_profile(
        extracted_text=text
    )

    result = validate_page(
        page_number=1,
        docling_text=text,
        reference_profile=profile,
        docling_table_count=0,
        pdfplumber_table_count=0,
        config=config,
    )

    assert result["status"] == "PASS"
    assert result["cross_validation"] == "AVAILABLE"
    assert result["metrics"]["token_coverage"] == 1.0


def test_validate_page_low_token_coverage_fails():
    config = ValidationConfig()

    reference_text = (
        "insurance policy coverage "
        "vehicle damage accident claim "
    ) * 10

    docling_text = "insurance"

    profile = make_reference_profile(
        extracted_text=reference_text
    )

    result = validate_page(
        page_number=1,
        docling_text=docling_text,
        reference_profile=profile,
        docling_table_count=0,
        pdfplumber_table_count=0,
        config=config,
    )

    assert result["status"] == "FAIL"

    issue_codes = {
        issue["code"]
        for issue in result["issues"]
    }

    assert (
        "LOW_TOKEN_COVERAGE"
        in issue_codes
    )


def test_validate_page_scanned():
    config = ValidationConfig()

    profile = make_reference_profile(
        page_type="SCANNED_OR_IMAGE_BASED",
        usable_text_layer=False,
        extracted_text="",
        pdf_character_objects=0,
        image_count=1,
        largest_image_coverage=1.0,
        total_image_coverage=1.0,
    )

    result = validate_page(
        page_number=1,
        docling_text="OCR extracted insurance text",
        reference_profile=profile,
        docling_table_count=0,
        pdfplumber_table_count=0,
        config=config,
    )

    assert result["status"] == "REVIEW"

    assert (
        result["cross_validation"]
        == "UNAVAILABLE"
    )


def test_validate_page_low_text_reference():
    config = ValidationConfig()

    profile = make_reference_profile(
        page_type="LOW_TEXT_OR_UNKNOWN",
        usable_text_layer=False,
        extracted_text="",
        pdf_character_objects=5,
    )

    result = validate_page(
        page_number=1,
        docling_text="Some Docling text",
        reference_profile=profile,
        docling_table_count=0,
        pdfplumber_table_count=0,
        config=config,
    )

    assert result["status"] == "REVIEW"

    assert (
        result["cross_validation"]
        == "INSUFFICIENT_REFERENCE"
    )


def test_validate_page_table_mismatch():
    config = ValidationConfig()

    text = "Insurance policy coverage details."

    profile = make_reference_profile(
        extracted_text=text
    )

    result = validate_page(
        page_number=1,
        docling_text=text,
        reference_profile=profile,
        docling_table_count=0,
        pdfplumber_table_count=1,
        config=config,
    )

    assert result["status"] == "REVIEW"

    issue_codes = {
        issue["code"]
        for issue in result["issues"]
    }

    assert (
        "TABLE_MISSED_BY_DOCLING"
        in issue_codes
    )


# ============================================================
# DOCUMENT STATUS
# ============================================================


def test_determine_document_status_pass():
    config = ValidationConfig()

    pages = [
        {"status": "PASS"},
        {"status": "PASS"},
        {"status": "PASS"},
    ]

    result = determine_document_status(
        page_results=pages,
        page_count_match=True,
        config=config,
    )

    assert result == "PASS"


def test_determine_document_status_review():
    config = ValidationConfig()

    pages = [
        {"status": "PASS"},
        {"status": "REVIEW"},
        {"status": "PASS"},
    ]

    result = determine_document_status(
        page_results=pages,
        page_count_match=True,
        config=config,
    )

    assert result == "REVIEW"


def test_determine_document_status_page_count_mismatch():
    config = ValidationConfig()

    pages = [
        {"status": "PASS"},
        {"status": "PASS"},
    ]

    result = determine_document_status(
        page_results=pages,
        page_count_match=False,
        config=config,
    )

    assert result == "REVIEW"


def test_determine_document_status_failure():
    config = ValidationConfig()

    pages = [
        {"status": "FAIL"},
        {"status": "FAIL"},
        {"status": "PASS"},
        {"status": "PASS"},
    ]

    result = determine_document_status(
        page_results=pages,
        page_count_match=True,
        config=config,
    )

    assert result == "FAIL"


def test_determine_document_status_no_pages():
    config = ValidationConfig()

    result = determine_document_status(
        page_results=[],
        page_count_match=True,
        config=config,
    )

    assert result == "FAIL"


# ============================================================
# DOCUMENT TYPE
# ============================================================


def test_determine_document_type_digital():
    result = determine_document_type(
        [
            {"page_type": "DIGITAL"},
            {"page_type": "DIGITAL"},
        ]
    )

    assert result == "DIGITAL"


def test_determine_document_type_scanned():
    result = determine_document_type(
        [
            {
                "page_type":
                    "SCANNED_OR_IMAGE_BASED"
            },
            {
                "page_type":
                    "SCANNED_OR_IMAGE_BASED"
            },
        ]
    )

    assert (
        result
        == "SCANNED_OR_IMAGE_BASED"
    )


def test_determine_document_type_mixed():
    result = determine_document_type(
        [
            {"page_type": "DIGITAL"},
            {"page_type": "MIXED"},
        ]
    )

    assert result == "MIXED"


def test_determine_document_type_unknown():
    result = determine_document_type(
        [
            {
                "page_type":
                    "LOW_TEXT_OR_UNKNOWN"
            }
        ]
    )

    assert result == "UNKNOWN"


# ============================================================
# SAFE MEAN
# ============================================================


def test_safe_mean():
    result = safe_mean(
        [0.8, 0.9, None, 1.0]
    )

    assert result == 0.9


def test_safe_mean_all_none():
    result = safe_mean(
        [None, None]
    )

    assert result is None