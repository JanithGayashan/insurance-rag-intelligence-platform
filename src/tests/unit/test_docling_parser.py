import hashlib
from pathlib import Path
from types import SimpleNamespace

from src.ingestion import docling_parser


class FakeDocument:
    """
    Small fake Docling document used for unit tests.
    """

    def __init__(
        self,
        text: str = "Insurance policy test document.",
        page_count: int = 2,
        table_count: int = 1,
    ):
        self._text = text
        self._page_count = page_count
        self.tables = [
            object()
            for _ in range(table_count)
        ]

    def num_pages(self) -> int:
        return self._page_count

    def save_as_json(
        self,
        path: Path,
    ) -> None:
        path.write_text(
            '{"test": "json"}',
            encoding="utf-8",
        )

    def save_as_markdown(
        self,
        path: Path,
    ) -> None:
        path.write_text(
            "# Insurance Policy",
            encoding="utf-8",
        )

    def export_to_text(self) -> str:
        return self._text


class FakeConverter:
    """
    Fake converter that behaves like the part of
    Docling's DocumentConverter used by parse_pdf().
    """

    def __init__(
        self,
        document=None,
        status="success",
        error=None,
    ):
        self.document = (
            document
            or FakeDocument()
        )
        self.status = status
        self.error = error

    def convert(
        self,
        path: Path,
    ):
        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            document=self.document,
            status=SimpleNamespace(
                value=self.status
            ),
        )


# ============================================================
# SHA-256
# ============================================================


def test_calculate_sha256(tmp_path):
    file_path = (
        tmp_path / "test.pdf"
    )

    content = b"insurance test"

    file_path.write_bytes(content)

    expected = hashlib.sha256(
        content
    ).hexdigest()

    result = (
        docling_parser.calculate_sha256(
            file_path
        )
    )

    assert result == expected


# ============================================================
# PARSE SUCCESS
# ============================================================


def test_parse_pdf_success(tmp_path):
    raw_root = (
        tmp_path / "raw"
    )

    processed_root = (
        tmp_path / "processed"
    )

    pdf_path = (
        raw_root
        / "allianz"
        / "motor"
        / "policy.pdf"
    )

    pdf_path.parent.mkdir(
        parents=True
    )

    pdf_path.write_bytes(
        b"fake pdf content"
    )

    converter = FakeConverter()

    result = docling_parser.parse_pdf(
        pdf_path=pdf_path,
        raw_root=raw_root,
        processed_root=processed_root,
        converter=converter,
    )

    output_dir = (
        processed_root
        / "allianz"
        / "motor"
        / "policy"
    )

    assert (
        result["parser"]["name"]
        == "docling"
    )

    assert (
        result["parser"]["status"]
        == "success"
    )

    assert (
        result["document"]["page_count"]
        == 2
    )

    assert (
        result["document"]["table_count"]
        == 1
    )

    assert (
        output_dir
        / "document.json"
    ).exists()

    assert (
        output_dir
        / "document.md"
    ).exists()

    assert (
        output_dir
        / "document.txt"
    ).exists()

    assert (
        output_dir
        / "metadata.json"
    ).exists()


# ============================================================
# PARSE FAILURE
# ============================================================


def test_parse_pdf_failure(tmp_path):
    raw_root = (
        tmp_path / "raw"
    )

    processed_root = (
        tmp_path / "processed"
    )

    pdf_path = (
        raw_root / "broken.pdf"
    )

    raw_root.mkdir(
        parents=True
    )

    pdf_path.write_bytes(
        b"broken pdf"
    )

    converter = FakeConverter(
        error=RuntimeError(
            "Test conversion failure"
        )
    )

    result = docling_parser.parse_pdf(
        pdf_path=pdf_path,
        raw_root=raw_root,
        processed_root=processed_root,
        converter=converter,
    )

    output_dir = (
        processed_root / "broken"
    )

    assert (
        result["parser"]["name"]
        == "docling"
    )

    assert (
        result["parser"]["status"]
        == "failed"
    )

    assert (
        result["error"]["type"]
        == "RuntimeError"
    )

    assert (
        output_dir / "error.json"
    ).exists()


# ============================================================
# OUTPUT METADATA
# ============================================================


def test_parse_pdf_metadata_contains_sha256(
    tmp_path,
):
    raw_root = (
        tmp_path / "raw"
    )

    processed_root = (
        tmp_path / "processed"
    )

    pdf_path = (
        raw_root / "policy.pdf"
    )

    raw_root.mkdir(
        parents=True
    )

    content = b"insurance document"

    pdf_path.write_bytes(
        content
    )

    converter = FakeConverter()

    result = docling_parser.parse_pdf(
        pdf_path=pdf_path,
        raw_root=raw_root,
        processed_root=processed_root,
        converter=converter,
    )

    expected_hash = hashlib.sha256(
        content
    ).hexdigest()

    assert (
        result["source"]["sha256"]
        == expected_hash
    )

    assert (
        result["source"]["file_name"]
        == "policy.pdf"
    )