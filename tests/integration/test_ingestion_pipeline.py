from pathlib import Path
from types import SimpleNamespace

from src.ingestion import pipeline


class FakeDocument:
    """
    Fake Docling document for integration testing.
    """

    def __init__(self):
        self.tables = []

    def num_pages(self) -> int:
        return 1

    def save_as_json(
        self,
        path: Path,
    ) -> None:
        path.write_text(
            '{"document": "test"}',
            encoding="utf-8",
        )

    def save_as_markdown(
        self,
        path: Path,
    ) -> None:
        path.write_text(
            "# Test Document",
            encoding="utf-8",
        )

    def export_to_text(self) -> str:
        return (
            "Insurance test document."
        )


class FakeConverter:
    """
    Fake Docling converter.

    good.pdf succeeds.
    bad.pdf raises an exception.
    """

    def convert(
        self,
        path: Path,
    ):
        if path.name == "bad.pdf":
            raise RuntimeError(
                "Simulated parser failure"
            )

        return SimpleNamespace(
            document=FakeDocument(),
            status=SimpleNamespace(
                value="success"
            ),
        )


# ============================================================
# PDF DISCOVERY
# ============================================================


def test_discover_pdfs(tmp_path):
    raw_directory = (
        tmp_path / "raw"
    )

    first_pdf = (
        raw_directory
        / "policy_a.pdf"
    )

    second_pdf = (
        raw_directory
        / "allianz"
        / "motor"
        / "policy_b.pdf"
    )

    text_file = (
        raw_directory
        / "notes.txt"
    )

    first_pdf.parent.mkdir(
        parents=True
    )

    second_pdf.parent.mkdir(
        parents=True
    )

    first_pdf.write_bytes(
        b"pdf"
    )

    second_pdf.write_bytes(
        b"pdf"
    )

    text_file.write_text(
        "not a pdf",
        encoding="utf-8",
    )

    result = pipeline.discover_pdfs(
        raw_directory
    )

    assert result == [
        first_pdf,
        second_pdf,
    ]


# ============================================================
# FULL INGESTION PIPELINE
# ============================================================


def test_run_ingestion_pipeline(
    tmp_path,
    monkeypatch,
    capsys,
):
    raw_directory = (
        tmp_path / "raw"
    )

    processed_directory = (
        tmp_path / "processed"
    )

    good_pdf = (
        raw_directory
        / "good.pdf"
    )

    bad_pdf = (
        raw_directory
        / "bad.pdf"
    )

    raw_directory.mkdir(
        parents=True
    )

    good_pdf.write_bytes(
        b"good pdf"
    )

    bad_pdf.write_bytes(
        b"bad pdf"
    )

    # Replace the real project directories
    # with temporary test directories.
    monkeypatch.setattr(
        pipeline,
        "RAW_DATA_DIR",
        raw_directory,
    )

    monkeypatch.setattr(
        pipeline,
        "PROCESSED_DATA_DIR",
        processed_directory,
    )

    # Replace the real Docling converter.
    monkeypatch.setattr(
        pipeline,
        "create_converter",
        lambda: FakeConverter(),
    )

    # Run the real ingestion pipeline.
    pipeline.run_ingestion_pipeline()

    captured = capsys.readouterr()

    assert (
        "Found 2 PDF files"
        in captured.out
    )

    assert (
        "Successful: 1"
        in captured.out
    )

    assert (
        "Failed: 1"
        in captured.out
    )

    # --------------------------------
    # Successful document
    # --------------------------------

    good_output = (
        processed_directory
        / "good"
    )

    assert (
        good_output
        / "document.json"
    ).exists()

    assert (
        good_output
        / "document.md"
    ).exists()

    assert (
        good_output
        / "document.txt"
    ).exists()

    assert (
        good_output
        / "metadata.json"
    ).exists()

    # --------------------------------
    # Failed document
    # --------------------------------

    bad_output = (
        processed_directory
        / "bad"
    )

    assert (
        bad_output
        / "error.json"
    ).exists()