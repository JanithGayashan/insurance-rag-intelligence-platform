from pathlib import Path
from types import SimpleNamespace
import json
from dataclasses import replace

from src.ingestion import pipeline
from src.configuration.dataset import get_dataset_configuration


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


def test_ingestion_discovers_and_mirrors_all_slic_categories(tmp_path, monkeypatch):
    dataset = replace(get_dataset_configuration(), raw_root=tmp_path / "raw/slic",
                      processed_root=tmp_path / "processed/slic")
    sources = ["motor/policy.pdf", "general/travel/nested/book.PDF", "general/health/health.pdf"]
    for source in sources:
        file = dataset.raw_root / source
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(source.encode())
    other = dataset.raw_root.parent / "allianz/motor/other.pdf"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"preserve other insurer")
    monkeypatch.setattr(pipeline, "create_converter", lambda: FakeConverter())
    pipeline.run_ingestion_pipeline(dataset=dataset)
    for source in sources:
        output = dataset.processed_root / Path(source).with_suffix("")
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["source"]["insurer"] == "slic"
        assert metadata["source"]["category"] in {"motor", "travel", "health"}
        assert metadata["source"]["business_line"] == "general"
        assert len(metadata["source"]["sha256"]) == 64
    assert len(list(dataset.processed_root.rglob("document.json"))) == 3
    assert other.read_bytes() == b"preserve other insurer"
