from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata

import pdfplumber
from docling_core.types.doc.document import DoclingDocument


# ============================================================
# PATHS
# ============================================================

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

SUMMARY_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "validation_summary.json"
)


# ============================================================
# VALIDATION CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ValidationConfig:
    """
    Configuration for document parsing quality validation.

    These are initial heuristic thresholds.
    We will calibrate them later using manually
    validated insurance-document samples.
    """

    # ==================================================
    # PDF TYPE / TEXT-LAYER DETECTION
    # ==================================================

    # Minimum reconstructed native-text characters
    # required before pdfplumber text is considered usable.
    min_usable_text_chars: int = 80

    # Minimum number of native PDF character objects.
    min_pdf_character_objects: int = 20

    # If one image covers >= 80% of the page,
    # it is a strong signal that the page may be scanned.
    scanned_image_coverage_threshold: float = 0.80

    # Used to distinguish meaningful embedded-image content
    # from tiny logos/icons.
    mixed_image_coverage_threshold: float = 0.20


    # ==================================================
    # CROSS-PARSER TEXT VALIDATION
    # ==================================================

    token_coverage_review_threshold: float = 0.85
    token_coverage_fail_threshold: float = 0.60

    char_ratio_review_threshold: float = 0.70

    empty_docling_reference_chars: int = 100


    # ==================================================
    # TEXT QUALITY
    # ==================================================

    max_replacement_char_ratio: float = 0.005


    # ==================================================
    # TABLE VALIDATION
    # ==================================================

    check_tables: bool = True


    # ==================================================
    # DOCUMENT-LEVEL STATUS
    # ==================================================

    max_critical_page_fraction: float = 0.20

    minimum_critical_pages_for_document_failure: int = 2


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text before comparing parsers.

    We do NOT compare raw strings because Docling and
    pdfplumber can legitimately differ in whitespace,
    line breaks, Unicode representation, etc.
    """

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text: str) -> list[str]:
    """
    Convert normalized text into tokens.

    Token comparison is less sensitive to reading-order
    and whitespace differences than raw string comparison.
    """

    normalized = normalize_text(text)

    return re.findall(
        r"\b[\w]+(?:[./%-][\w]+)*\b",
        normalized,
        flags=re.UNICODE,
    )


# ============================================================
# TEXT COMPARISON METRICS
# ============================================================

def calculate_token_coverage(
    docling_text: str,
    reference_text: str,
) -> float | None:
    """
    Measures how much of the pdfplumber token content
    can also be found in Docling output.

    pdfplumber is being used as an independent reference
    signal — NOT as absolute ground truth.

    Returns:
        1.0  -> very high agreement
        0.0  -> no token agreement
        None -> reference had no tokens
    """

    docling_tokens = Counter(
        tokenize(docling_text)
    )

    reference_tokens = Counter(
        tokenize(reference_text)
    )

    reference_total = sum(
        reference_tokens.values()
    )

    if reference_total == 0:
        return None

    matching_tokens = sum(
        min(
            docling_tokens[token],
            count,
        )
        for token, count
        in reference_tokens.items()
    )

    return (
        matching_tokens
        / reference_total
    )


def calculate_char_ratio(
    docling_text: str,
    reference_text: str,
) -> float | None:
    """
    Compare normalized text lengths.

    Example:

        Docling      = 950 chars
        pdfplumber   = 1000 chars

        ratio = 950 / 1000
              = 0.95
    """

    docling_text = normalize_text(
        docling_text
    )

    reference_text = normalize_text(
        reference_text
    )

    docling_length = len(
        docling_text
    )

    reference_length = len(
        reference_text
    )

    if (
        docling_length == 0
        and reference_length == 0
    ):
        return None

    largest = max(
        docling_length,
        reference_length,
    )

    smallest = min(
        docling_length,
        reference_length,
    )

    return smallest / largest


def calculate_replacement_char_ratio(
    text: str,
) -> float:
    """
    Detect suspicious Unicode replacement characters:

        �

    Too many may indicate extraction/encoding problems.
    """

    if not text:
        return 0.0

    replacement_count = text.count(
        "\ufffd"
    )

    return replacement_count / len(text)


# ============================================================
# DOCLING STRUCTURE INFORMATION
# ============================================================

def get_docling_table_counts(
    document: DoclingDocument,
) -> tuple[dict[int, int], int]:
    """
    Count Docling tables by source page.

    Returns:

        {
            4: 2,
            7: 1
        }

    plus number of tables without page provenance.
    """

    table_counts: Counter[int] = Counter()

    without_provenance = 0

    for table in document.tables:

        provenance = getattr(
            table,
            "prov",
            None,
        )

        if not provenance:
            without_provenance += 1
            continue

        page_number = getattr(
            provenance[0],
            "page_no",
            None,
        )

        if page_number is None:
            without_provenance += 1
            continue

        table_counts[
            int(page_number)
        ] += 1

    return (
        dict(table_counts),
        without_provenance,
    )

def profile_pdfplumber_page(
    page,
    config: ValidationConfig,
) -> dict:
    """
    Inspect the PDF page and estimate whether it is:

        DIGITAL
        MIXED
        OCR_TEXT_OVER_IMAGE
        SCANNED_OR_IMAGE_BASED
        LOW_TEXT_OR_UNKNOWN

    Signals used:

        1. Native PDF character objects
        2. Reconstructed native text
        3. Image coverage of the page

    pdfplumber does NOT perform OCR here.
    """

    # ==================================================
    # 1. EXTRACT NATIVE PDF TEXT
    # ==================================================

    try:
        deduped_page = page.dedupe_chars()

        extracted_text = (
            deduped_page.extract_text()
            or ""
        )

    except Exception:
        extracted_text = ""

    normalized_text = normalize_text(
        extracted_text
    )

    text_character_count = len(
        normalized_text
    )

    pdf_character_objects = len(
        page.chars
    )


    # ==================================================
    # 2. PAGE AREA
    # ==================================================

    page_width = float(
        page.width
    )

    page_height = float(
        page.height
    )

    page_area = (
        page_width
        * page_height
    )


    # ==================================================
    # 3. IMAGE COVERAGE
    # ==================================================

    image_coverages = []

    for image in page.images:

        try:
            x0 = float(
                image["x0"]
            )

            x1 = float(
                image["x1"]
            )

            top = float(
                image["top"]
            )

            bottom = float(
                image["bottom"]
            )

            image_width = max(
                0.0,
                x1 - x0,
            )

            image_height = max(
                0.0,
                bottom - top,
            )

            image_area = (
                image_width
                * image_height
            )

            if page_area > 0:

                coverage = (
                    image_area
                    / page_area
                )

                # Protect against unusual PDF geometry.
                coverage = min(
                    max(coverage, 0.0),
                    1.0,
                )

                image_coverages.append(
                    coverage
                )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue


    # Largest image is especially useful for detecting
    # full-page scanned images.
    largest_image_coverage = (
        max(image_coverages)
        if image_coverages
        else 0.0
    )

    # Approximate total coverage.
    #
    # Note:
    # Images can overlap, so this is only a diagnostic
    # signal rather than an exact geometric union.
    total_image_coverage = min(
        sum(image_coverages),
        1.0,
    )


    # ==================================================
    # 4. DETERMINE WHETHER A USABLE TEXT LAYER EXISTS
    # ==================================================

    usable_text_layer = (
        text_character_count
        >= config.min_usable_text_chars

        and

        pdf_character_objects
        >= config.min_pdf_character_objects
    )


    # ==================================================
    # 5. PAGE CLASSIFICATION
    # ==================================================

    large_page_image = (
        largest_image_coverage
        >= config.scanned_image_coverage_threshold
    )

    meaningful_image_content = (
        largest_image_coverage
        >= config.mixed_image_coverage_threshold
    )


    # --------------------------------------------------
    # CASE A:
    # Good text layer + full-page image
    #
    # Often a scanned page that already has an OCR
    # text layer embedded in the PDF.
    # --------------------------------------------------

    if (
        usable_text_layer
        and large_page_image
    ):

        page_type = (
            "OCR_TEXT_OVER_IMAGE"
        )


    # --------------------------------------------------
    # CASE B:
    # Good native text + meaningful images
    # --------------------------------------------------

    elif (
        usable_text_layer
        and meaningful_image_content
    ):

        page_type = "MIXED"


    # --------------------------------------------------
    # CASE C:
    # Good text + only small images/logos
    # --------------------------------------------------

    elif usable_text_layer:

        page_type = "DIGITAL"


    # --------------------------------------------------
    # CASE D:
    # No useful text + nearly full-page image
    #
    # Strong scanned-page signal.
    # --------------------------------------------------

    elif large_page_image:

        page_type = (
            "SCANNED_OR_IMAGE_BASED"
        )


    # --------------------------------------------------
    # CASE E:
    # Not enough evidence to confidently classify.
    # --------------------------------------------------

    else:

        page_type = (
            "LOW_TEXT_OR_UNKNOWN"
        )


    return {
        "page_type":
            page_type,

        "usable_text_layer":
            usable_text_layer,

        "extracted_text":
            extracted_text,

        "text_characters":
            text_character_count,

        "pdf_character_objects":
            pdf_character_objects,

        "image_count":
            len(page.images),

        "largest_image_coverage":
            round(
                largest_image_coverage,
                4,
            ),

        "total_image_coverage":
            round(
                total_image_coverage,
                4,
            ),
    }

# ============================================================
# PAGE VALIDATION
# ============================================================

def validate_page(
    page_number: int,
    docling_text: str,
    reference_profile: dict,
    docling_table_count: int,
    pdfplumber_table_count: int,
    config: ValidationConfig,
) -> dict:

    page_type = (
        reference_profile[
            "page_type"
        ]
    )

    usable_text_layer = (
        reference_profile[
            "usable_text_layer"
        ]
    )

    pdfplumber_text = (
        reference_profile[
            "extracted_text"
        ]
    )

    normalized_docling = normalize_text(
        docling_text
    )

    normalized_reference = normalize_text(
        pdfplumber_text
    )

    docling_chars = len(
        normalized_docling
    )

    reference_chars = len(
        normalized_reference
    )

    issues = []

    # =====================================================
    # CASE 1:
    # SCANNED / IMAGE-BASED PAGE
    # =====================================================

    if page_type == "SCANNED_OR_IMAGE_BASED":

        issues.append(
            {
                "severity": "warning",

                "code":
                    "OCR_CROSS_VALIDATION_UNAVAILABLE",

                "message": (
                    "The page appears to be scanned or "
                    "image-based. pdfplumber does not "
                    "provide a usable native text layer, "
                    "so Docling OCR output cannot be "
                    "cross-validated with pdfplumber."
                ),
            }
        )

        if docling_chars == 0:

            issues.append(
                {
                    "severity": "warning",

                    "code":
                        "DOCLING_OCR_OUTPUT_EMPTY",

                    "message": (
                        "Docling produced no text for an "
                        "image-based page. Manual review "
                        "may be required."
                    ),
                }
            )

        return {
            "page_number": page_number,

            "page_type": page_type,

            "cross_validation":
                "UNAVAILABLE",

            "status": "REVIEW",

            "metrics": {
                "docling_characters":
                    docling_chars,

                "pdfplumber_characters":
                    reference_chars,

                "pdf_character_objects":
                    reference_profile[
                        "pdf_character_objects"
                    ],

                "image_count":
                    reference_profile[
                        "image_count"
                    ],

                "largest_image_coverage":
                    reference_profile[
                        "largest_image_coverage"
                    ],

                "total_image_coverage":
                    reference_profile[
                        "total_image_coverage"
                    ],

                "token_coverage":
                    None,

                "character_length_ratio":
                    None,

                "docling_tables":
                    docling_table_count,

                "pdfplumber_tables":
                    None,
            },

            "issues": issues,
        }

    # =====================================================
    # CASE 2:
    # VERY LOW TEXT / UNKNOWN PAGE
    # =====================================================

    if not usable_text_layer:

        status = "REVIEW"

        issues.append(
            {
                "severity": "warning",

                "code":
                    "REFERENCE_TEXT_INSUFFICIENT",

                "message": (
                    "The page does not contain enough native "
                    "PDF text for reliable pdfplumber "
                    "cross-validation. Manual review may "
                    "be required."
                ),
            }
        )

        return {
            "page_number": page_number,

            "page_type": page_type,

            "cross_validation":
                "INSUFFICIENT_REFERENCE",

            "status": status,

            "metrics": {
                "docling_characters":
                    docling_chars,

                "pdfplumber_characters":
                    reference_chars,

                "pdf_character_objects":
                    reference_profile[
                        "pdf_character_objects"
                    ],

                "image_count":
                    reference_profile[
                        "image_count"
                    ],

                "largest_image_coverage":
                    reference_profile[
                        "largest_image_coverage"
                    ],

                "total_image_coverage":
                    reference_profile[
                        "total_image_coverage"
                    ],

                "token_coverage":
                    None,

                "character_length_ratio":
                    None,

                "docling_tables":
                    docling_table_count,

                "pdfplumber_tables":
                    None,
            },

            "issues": issues,
        }

    # =====================================================
    # CASE 3:
    # DIGITAL / MIXED PAGE
    #
    # pdfplumber contains usable native text.
    # Cross-parser comparison is valid.
    # =====================================================

    token_coverage = (
        calculate_token_coverage(
            docling_text,
            pdfplumber_text,
        )
    )

    char_ratio = (
        calculate_char_ratio(
            docling_text,
            pdfplumber_text,
        )
    )

    replacement_ratio = (
        calculate_replacement_char_ratio(
            docling_text
        )
    )

    # ----------------------------------------------
    # Missing Docling text
    # ----------------------------------------------

    if (
        reference_chars
        >= config.empty_docling_reference_chars
        and docling_chars == 0
    ):

        issues.append(
            {
                "severity": "critical",

                "code":
                    "DOCLING_EMPTY_REFERENCE_NONEMPTY",

                "message": (
                    "Docling extracted no text "
                    "while pdfplumber found "
                    f"{reference_chars} characters."
                ),
            }
        )

    # ----------------------------------------------
    # Token agreement
    # ----------------------------------------------

    if token_coverage is not None:

        if (
            token_coverage
            < config.token_coverage_fail_threshold
        ):

            issues.append(
                {
                    "severity": "critical",

                    "code":
                        "LOW_TOKEN_COVERAGE",

                    "message": (
                        "Low token agreement between "
                        "Docling and pdfplumber."
                    ),
                }
            )

        elif (
            token_coverage
            < config.token_coverage_review_threshold
        ):

            issues.append(
                {
                    "severity": "warning",

                    "code":
                        "MODERATE_TOKEN_COVERAGE",

                    "message": (
                        "Moderate token disagreement "
                        "between Docling and pdfplumber."
                    ),
                }
            )

    # ----------------------------------------------
    # Text length disagreement
    # ----------------------------------------------

    if (
        char_ratio is not None

        and char_ratio
        < config.char_ratio_review_threshold
    ):

        issues.append(
            {
                "severity": "warning",

                "code":
                    "TEXT_LENGTH_MISMATCH",

                "message": (
                    "Large difference between "
                    "Docling and pdfplumber text lengths."
                ),
            }
        )

    # ----------------------------------------------
    # Unicode corruption
    # ----------------------------------------------

    if (
        replacement_ratio
        > config.max_replacement_char_ratio
    ):

        issues.append(
            {
                "severity": "warning",

                "code":
                    "REPLACEMENT_CHARACTERS",

                "message": (
                    "Docling output contains suspicious "
                    "Unicode replacement characters."
                ),
            }
        )

    # ----------------------------------------------
    # Tables
    # ----------------------------------------------

    if (
        config.check_tables

        and pdfplumber_table_count > 0

        and docling_table_count == 0
    ):

        issues.append(
            {
                "severity": "warning",

                "code":
                    "TABLE_MISSED_BY_DOCLING",

                "message": (
                    "pdfplumber detected "
                    f"{pdfplumber_table_count} table(s), "
                    "while Docling detected none."
                ),
            }
        )

    # ----------------------------------------------
    # Final page status
    # ----------------------------------------------

    severities = {
        issue["severity"]
        for issue in issues
    }

    if "critical" in severities:
        status = "FAIL"

    elif "warning" in severities:
        status = "REVIEW"

    else:
        status = "PASS"

    return {
        "page_number": page_number,

        "page_type": page_type,

        "cross_validation":
            "AVAILABLE",

        "status": status,

        "metrics": {
            "docling_characters":
                docling_chars,

            "pdfplumber_characters":
                reference_chars,

            "pdf_character_objects":
                reference_profile[
                    "pdf_character_objects"
                ],

            "image_count":
                reference_profile[
                    "image_count"
                ],

            "largest_image_coverage":
                reference_profile[
                    "largest_image_coverage"
                ],

            "total_image_coverage":
                reference_profile[
                    "total_image_coverage"
                ],

            "token_coverage":
                (
                    round(
                        token_coverage,
                        4,
                    )
                    if token_coverage
                    is not None
                    else None
                ),

            "character_length_ratio":
                (
                    round(
                        char_ratio,
                        4,
                    )
                    if char_ratio
                    is not None
                    else None
                ),

            "replacement_character_ratio":
                round(
                    replacement_ratio,
                    6,
                ),

            "docling_tables":
                docling_table_count,

            "pdfplumber_tables":
                pdfplumber_table_count,
        },

        "issues": issues,
    }


# ============================================================
# DOCUMENT STATUS
# ============================================================

def determine_document_status(
    page_results: list[dict],
    page_count_match: bool,
    config: ValidationConfig,
) -> str:

    critical_pages = sum(
        1
        for page in page_results
        if page["status"] == "FAIL"
    )

    review_pages = sum(
        1
        for page in page_results
        if page["status"] == "REVIEW"
    )

    total_pages = len(
        page_results
    )

    # No comparable Docling pages is a serious problem.
    if total_pages == 0:
        return "FAIL"

    failure_limit = max(
        config.minimum_critical_pages_for_document_failure,

        math.ceil(
            total_pages
            * config.max_critical_page_fraction
        ),
    )

    # Enough page-level evidence of extraction failure.
    if critical_pages >= failure_limit:
        return "FAIL"

    # Cross-parser page-count disagreement requires review,
    # but does not by itself prove that Docling failed.
    if not page_count_match:
        return "REVIEW"

    if (
        critical_pages > 0
        or review_pages > 0
    ):
        return "REVIEW"

    return "PASS"


# ============================================================
# SAFE MEAN
# ============================================================

def safe_mean(
    values: list[float | None],
) -> float | None:

    valid_values = [
        value
        for value in values
        if value is not None
    ]

    if not valid_values:
        return None

    return round(
        statistics.mean(valid_values),
        4,
    )

def determine_document_type(
    page_results: list[dict],
) -> str:
    """
    Determine overall PDF type from page-level classifications.

    LOW_TEXT_OR_UNKNOWN pages are treated as neutral because
    blank pages, covers, or low-content pages should not
    automatically change the entire document type.
    """

    page_types = {
        page["page_type"]
        for page in page_results
    }

    informative_types = (
        page_types
        - {"LOW_TEXT_OR_UNKNOWN"}
    )

    if not informative_types:
        return "UNKNOWN"

    if informative_types == {"DIGITAL"}:
        return "DIGITAL"

    if informative_types == {
        "OCR_TEXT_OVER_IMAGE"
    }:
        return "OCR_TEXT_OVER_IMAGE"

    if informative_types.issubset(
        {
            "SCANNED_OR_IMAGE_BASED",
            "OCR_TEXT_OVER_IMAGE",
        }
    ):
        return "SCANNED_OR_IMAGE_BASED"

    if (
        "MIXED" in informative_types
        or len(informative_types) > 1
    ):
        return "MIXED"

    return "UNKNOWN"

def save_validation_report(
    processed_directory: Path,
    report: dict,
) -> None:

    report_path = (
        processed_directory
        / "validation_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

# ============================================================
# VALIDATE ONE DOCUMENT
# ============================================================

def validate_document(
    processed_directory: Path,
    config: ValidationConfig,
) -> dict:

    metadata_path = (
        processed_directory
        / "metadata.json"
    )

    document_json_path = (
        processed_directory
        / "document.json"
    )

    print(
        f"Validating: {processed_directory}"
    )

    # --------------------------------------------------------
    # Load ingestion metadata
    # --------------------------------------------------------

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        ingestion_metadata = (
            json.load(file)
        )

    relative_source_path = Path(
        ingestion_metadata[
            "source"
        ][
            "relative_path"
        ]
    )

    source_pdf_path = (
        RAW_DATA_DIR
        / relative_source_path
    )

    # --------------------------------------------------------
    # Basic existence checks
    # --------------------------------------------------------

    if not source_pdf_path.exists():

        return {
            "document":
                str(relative_source_path),

            "status":
                "FAIL",

            "error":
                (
                    "Original source PDF "
                    "could not be found."
                ),
        }

    if not document_json_path.exists():

        return {
            "document":
                str(relative_source_path),

            "status":
                "FAIL",

            "error":
                (
                    "Docling document.json "
                    "could not be found."
                ),
        }

    # --------------------------------------------------------
    # Reload Docling's native structured document
    #
    # IMPORTANT:
    # We are NOT running Docling parsing again.
    # --------------------------------------------------------

    document = (
        DoclingDocument.load_from_json(
            document_json_path
        )
    )

    docling_page_count = (
        document.num_pages()
    )

    docling_table_counts, unassigned_tables = (
        get_docling_table_counts(
            document
        )
    )

    # --------------------------------------------------------
    # Open original PDF independently using pdfplumber
    # --------------------------------------------------------

    page_results = []

    with pdfplumber.open(
        source_pdf_path
    ) as pdf:

        pdfplumber_page_count = len(
            pdf.pages
        )

        if pdfplumber_page_count == 0:

            report = {
                "document": {
                    "source_file":
                        str(relative_source_path),

                    "processed_directory":
                        str(processed_directory),
                },

                "status": "REVIEW",

                "document_type":
                    "UNKNOWN",

                "cross_parser": {
                    "name":
                        "pdfplumber",

                    "available":
                        False,

                    "reason":
                        "PDFPLUMBER_RETURNED_ZERO_PAGES",
                },

                "page_counts": {
                    "docling":
                        docling_page_count,

                    "pdfplumber":
                        0,

                    "match":
                        False,
                },

                "summary": {
                    "pass_pages": 0,
                    "review_pages": 0,
                    "failed_pages": 0,

                    "average_token_coverage":
                        None,

                    "average_character_length_ratio":
                        None,

                    "docling_total_tables":
                        len(document.tables),

                    "docling_tables_without_page_provenance":
                        unassigned_tables,
                },

                "issues": [
                    {
                        "severity":
                            "warning",

                        "code":
                            "CROSS_PARSER_UNAVAILABLE",

                        "message": (
                            "pdfplumber could not enumerate "
                            "the PDF pages. Docling output "
                            "cannot be cross-validated using "
                            "pdfplumber for this document."
                        ),
                    }
                ],

                "pages": [],
            }

            save_validation_report(
                processed_directory,
                report,
            )

            return report

        page_count_match = (
            docling_page_count
            == pdfplumber_page_count
        )

        pages_to_compare = min(
            docling_page_count,
            pdfplumber_page_count,
        )

        for page_index in range(
            pages_to_compare
        ):

            page_number = (
                page_index + 1
            )

            # ----------------------------------------------
            # DOCLING TEXT
            # ----------------------------------------------

            docling_text = (
                document.export_to_text(
                    page_no=page_number
                )
                or ""
            )

            # ----------------------------------------------
            # PDFPLUMBER TEXT
            # ----------------------------------------------

            pdf_page = (
                pdf.pages[
                    page_index
                ]
            )

            reference_profile = (
                profile_pdfplumber_page(
                    pdf_page,
                    config,
                )
            )

            # ----------------------------------------------
            # TABLE COUNTS
            # ----------------------------------------------

            docling_tables = (
                docling_table_counts.get(
                    page_number,
                    0,
                )
            )

            pdfplumber_tables = 0

            if (
                config.check_tables

                and reference_profile[
                    "usable_text_layer"
                ]
            ):

                try:

                    pdfplumber_tables = len(
                        pdf_page.find_tables()
                    )

                except Exception as error:

                    print(
                        f"Table validation warning "
                        f"on page {page_number}: "
                        f"{error}"
                    )

            # ----------------------------------------------
            # VALIDATE PAGE
            # ----------------------------------------------

            result = validate_page(
                page_number=page_number,

                docling_text=docling_text,

                reference_profile=
                    reference_profile,

                docling_table_count=
                    docling_tables,

                pdfplumber_table_count=
                    pdfplumber_tables,

                config=config,
            )

            page_results.append(
                result
            )
    
    document_type = determine_document_type(
        page_results
    )

    # --------------------------------------------------------
    # Aggregate document metrics
    # --------------------------------------------------------

    token_coverages = [
        page["metrics"][
            "token_coverage"
        ]
        for page in page_results
    ]

    char_ratios = [
        page["metrics"][
            "character_length_ratio"
        ]
        for page in page_results
    ]

    pass_pages = sum(
        page["status"] == "PASS"
        for page in page_results
    )

    review_pages = sum(
        page["status"] == "REVIEW"
        for page in page_results
    )

    failed_pages = sum(
        page["status"] == "FAIL"
        for page in page_results
    )

    overall_status = (
        determine_document_status(
            page_results,
            page_count_match,
            config,
        )
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = {
        "document": {
            "source_file":
                str(relative_source_path),

            "processed_directory":
                str(processed_directory),
        },

        "document_type": document_type,

        "status":
            overall_status,

        "page_counts": {
            "docling":
                docling_page_count,

            "pdfplumber":
                pdfplumber_page_count,

            "match":
                page_count_match,
        },

        "summary": {
            "pass_pages":
                pass_pages,

            "review_pages":
                review_pages,

            "failed_pages":
                failed_pages,

            "average_token_coverage":
                safe_mean(
                    token_coverages
                ),

            "average_character_length_ratio":
                safe_mean(
                    char_ratios
                ),

            "docling_total_tables":
                len(document.tables),

            "docling_tables_without_page_provenance":
                unassigned_tables,
        },

        "thresholds":
            asdict(config),

        "pages":
            page_results,
    }

    # --------------------------------------------------------
    # Save report beside processed document
    # --------------------------------------------------------

    save_validation_report(
        processed_directory,
        report,
    )

    return report


# ============================================================
# DISCOVER ALL PROCESSED DOCUMENTS
# ============================================================

def discover_processed_documents(
) -> list[Path]:

    metadata_files = sorted(
        PROCESSED_DATA_DIR.rglob(
            "metadata.json"
        )
    )

    return [
        metadata_file.parent
        for metadata_file
        in metadata_files
    ]


# ============================================================
# RUN FULL VALIDATION PIPELINE
# ============================================================

def run_validation_pipeline():

    print(
        "Starting parsing validation"
    )

    print("=" * 70)

    config = ValidationConfig()

    processed_documents = (
        discover_processed_documents()
    )

    print(
        f"Found "
        f"{len(processed_documents)} "
        f"processed documents"
    )

    if not processed_documents:

        print(
            "No processed documents found."
        )

        return

    summary_documents = []

    pass_count = 0
    review_count = 0
    fail_count = 0

    for processed_directory in (
        processed_documents
    ):

        try:

            report = validate_document(
                processed_directory=
                    processed_directory,

                config=config,
            )

            status = report.get(
                "status",
                "FAIL",
            )

        except Exception as error:

            status = "FAIL"

            report = {
                "document":
                    str(processed_directory),

                "status":
                    "FAIL",

                "error": {
                    "type":
                        type(error).__name__,

                    "message":
                        str(error),
                },
            }

        if status == "PASS":
            pass_count += 1

        elif status == "REVIEW":
            review_count += 1

        else:
            fail_count += 1

        summary_documents.append(
            {
                "document":
                    report.get(
                        "document"
                    ),

                "status":
                    status,

                "summary":
                    report.get(
                        "summary"
                    ),
            }
        )

        print(
            f"  → {status}"
        )

    # --------------------------------------------------------
    # Collection-level report
    # --------------------------------------------------------

    collection_report = {
        "total_documents":
            len(processed_documents),

        "pass":
            pass_count,

        "review":
            review_count,

        "fail":
            fail_count,

        "documents":
            summary_documents,
    }

    with SUMMARY_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            collection_report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("Validation completed")
    print(f"PASS:   {pass_count}")
    print(f"REVIEW: {review_count}")
    print(f"FAIL:   {fail_count}")

    print(
        "\nSummary saved to:"
    )

    print(
        SUMMARY_OUTPUT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_validation_pipeline()