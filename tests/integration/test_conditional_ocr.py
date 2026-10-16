"""Exercise native, scanned, mixed, previously OCRed and rotated PDF pages."""
from io import BytesIO
import hashlib
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont
import pymupdf
import pytest

from src.ingestion.parsers import conditional_ocr as ocr
from src.ingestion.parsers import pdfplumber_parser, pymupdf_parser


def text_image():
    image = Image.new("RGB", (1500, 600), "white")
    font_path = next((path for path in [Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")] if path.is_file()), None)
    if font_path is None:
        pytest.skip("A scalable test font is required for the real OCR fixture")
    font = ImageFont.truetype(str(font_path), 50)
    draw = ImageDraw.Draw(image)
    draw.text((60, 100), "Flood damage is covered.", fill="black", font=font)
    draw.text((60, 220), "Report the claim within seven days.", fill="black", font=font)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def sample_pdf(tmp_path):
    image = text_image()
    path = tmp_path / "mixed-policy.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_text((40, 50), "Native digital coverage", fontsize=16)
        page = document.new_page(width=600, height=800)
        page.insert_image(pymupdf.Rect(0, 0, 600, 240), stream=image)
        page = document.new_page(width=600, height=800)
        page.insert_text((40, 50), "Policy exclusions", fontsize=16)
        page.insert_image(pymupdf.Rect(30, 150, 570, 366), stream=image)
        document.new_page(width=600, height=800)  # A genuinely blank page.
        page = document.new_page(width=600, height=800)
        page.insert_image(pymupdf.Rect(0, 0, 600, 240), stream=image)
        page.set_rotation(90)
        document.save(path)
    return path


def test_native_blank_and_mixed_assessments(sample_pdf):
    with pymupdf.open(sample_pdf) as document:
        native = ocr.assess_page(document[0], document[0].get_text())
        mixed = ocr.assess_page(document[2], document[2].get_text())
        blank = ocr.assess_page(document[3], "")
    assert native["action"] == "NATIVE_TEXT" and not native["regions"]
    assert mixed["reasons"] == ["check_image_regions_for_missing_text"]
    assert len(mixed["regions"]) == 1
    assert blank["action"] == "NO_TEXT_CONTENT" and not blank["regions"]


@pytest.mark.parametrize("parser", [pymupdf_parser, pdfplumber_parser])
def test_real_conditional_ocr_preserves_native_text_and_source(sample_pdf, parser):
    if not ocr.runtime()[0]:
        pytest.skip("Real Tesseract runtime is optional in CI")
    fingerprint = hashlib.sha256(sample_pdf.read_bytes()).hexdigest()
    native = parser.parse_pdf(sample_pdf, ocr_mode="off")
    result = parser.parse_pdf(sample_pdf)
    assert result.success, result.error
    assert not result.metadata["ocr"]["failed_pages"], result.warnings
    assert result.pages[0].text == native.pages[0].text
    assert result.pages[3].text == ""
    for number in (2, 3, 5):
        text = " ".join(result.pages[number - 1].text.split()).casefold()
        assert "flood damage is covered" in text, (number, text)
        assert "report the claim within seven days" in text, (number, text)
        assert text.count("flood") == 1
    assert result.pages[2].text.count("Policy exclusions") == 1
    assert result.metadata["ocr"]["pages_ocr_applied"] == [2, 3, 5]
    assert hashlib.sha256(sample_pdf.read_bytes()).hexdigest() == fingerprint


def test_ocr_failure_preserves_native_text_and_flags_missing_content(sample_pdf, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Tesseract language data missing")
    monkeypatch.setattr(ocr, "run_tesseract", fail)
    result = pymupdf_parser.parse_pdf(sample_pdf)
    assert result.success
    assert "Policy exclusions" in result.pages[2].text
    assert result.metadata["ocr"]["failed_pages"] == [2, 3, 5]
    assert result.pages[2].metadata["ocr"]["review_required"]
    assert result.pages[2].metadata["ocr"]["action"] == "OCR_FAILED"
    assert any("language data missing" in warning for warning in result.warnings)


def test_timeout_and_photo_without_detected_text_are_reviewable(sample_pdf, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("tesseract", 1)
    monkeypatch.setattr(ocr, "run_tesseract", timeout)
    result = pdfplumber_parser.parse_pdf(sample_pdf)
    assert result.metadata["ocr"]["failed_pages"] == [2, 3, 5]
    monkeypatch.setattr(ocr, "run_tesseract", lambda *args, **kwargs:
        {"recognized_words": 0, "mean_word_confidence": None})
    result = pdfplumber_parser.parse_pdf(sample_pdf)
    assert result.pages[1].metadata["ocr"]["action"] == "NO_ADDITIONAL_TEXT"
    assert result.pages[1].metadata["ocr"]["review_required"]
    assert not result.pages[1].text.strip()


def test_existing_ocr_layer_is_masked_before_recognition(tmp_path, monkeypatch):
    path = tmp_path / "already-ocred.pdf"
    image = Image.new("RGB", (1200, 200), "white")
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_image(pymupdf.Rect(0, 0, 600, 100), stream=image_bytes.getvalue())
        page.insert_text((40, 50), "Already recognized text", fontsize=16, render_mode=3)
        document.save(path)
    monkeypatch.setattr(ocr, "run_tesseract", lambda *args, **kwargs: pytest.fail("Masked white image needs no OCR"))
    result = pymupdf_parser.parse_pdf(path)
    assert result.success and result.text.count("Already recognized text") == 1
    page_ocr = result.pages[0].metadata["ocr"]
    assert page_ocr["possible_existing_ocr_characters"] > 0
    assert page_ocr["regions"][0]["status"] == "NO_UNCOVERED_CONTENT"


@pytest.mark.parametrize("overlay_failure", [False, True])
@pytest.mark.parametrize("version_command", [None, "missing-tesseract-version-executable"])
def test_unreadable_native_text_is_replaced_only_after_successful_ocr(tmp_path, monkeypatch, overlay_failure, version_command):
    monkeypatch.setattr(ocr, "runtime", lambda: (version_command, {}, None))
    if version_command is None:
        monkeypatch.setattr(ocr.subprocess, "run", lambda *args, **kwargs: pytest.fail("A missing optional executable must not be invoked"))
    path = tmp_path / "broken-encoding.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_text((40, 50), "(cid:54) unreadable", fontsize=16)
        page.draw_line((40, 100), (500, 100))
        document.save(path)
    def recognize(image, output, **kwargs):
        with pymupdf.open() as document:
            page = document.new_page(width=600, height=800)
            page.insert_text((40, 50), "Recovered coverage", fontsize=16, render_mode=3)
            document.save(output.with_suffix(".pdf"))
        return {"recognized_words": 2, "mean_word_confidence": 99}
    monkeypatch.setattr(ocr, "run_tesseract", recognize)
    if overlay_failure:
        def fail_overlay(*args, **kwargs):
            raise RuntimeError("Unable to overlay OCR text")
        monkeypatch.setattr(pymupdf.Page, "show_pdf_page", fail_overlay)
    with ocr.prepare_pdf(path) as prepared:
        if overlay_failure:
            assert "cid:" in prepared.document[0].get_text()
            assert prepared.pages[0]["action"] == "OCR_FAILED"
            assert prepared.summary()["failed_pages"] == [1]
        else:
            assert "Recovered coverage" in prepared.document[0].get_text()
            assert "cid:" not in prepared.document[0].get_text()
            assert prepared.summary()["engine_version"] == "unavailable"
        assert prepared.document[0].get_drawings(), "Native table/rule graphics must survive"
        assert prepared.pages[0]["replace_native_text"]
    with pymupdf.open(path) as original:
        assert "cid:" in original[0].get_text()


@pytest.mark.parametrize("text,suspicious", [("\ufffd\ufffd\ufffd broken", True), ("(cid:54)", True),
    ("Coverage", False), ("රක්ෂණ தமிழ் insurance", False), ("", False)])
def test_text_quality_is_language_independent(text, suspicious):
    assert ocr.text_quality(text)["suspicious"] == suspicious


@pytest.mark.parametrize("options", [{"ocr_mode": "always"}, {"ocr_language": "../eng"},
    {"ocr_dpi": 0}, {"ocr_timeout": -1}])
def test_invalid_configuration_is_reported(sample_pdf, options):
    result = pymupdf_parser.parse_pdf(sample_pdf, **options)
    assert not result.success and "ValueError" in result.error
