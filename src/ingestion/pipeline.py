from pathlib import Path

from src.ingestion.docling_parser import (
    create_converter,
    parse_pdf,
)


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def discover_pdfs(
    raw_directory: Path,
) -> list[Path]:

    pdf_files = sorted(
        raw_directory.rglob("*.pdf")
    )

    return pdf_files


def run_ingestion_pipeline():

    print("Starting document ingestion")
    print("=" * 60)

    pdf_files = discover_pdfs(
        RAW_DATA_DIR
    )

    print(
        f"Found {len(pdf_files)} PDF files"
    )

    if not pdf_files:
        print(
            "No PDF files found."
        )
        return

    # Create once and reuse it.
    converter = create_converter()

    successful = 0
    failed = 0

    for pdf_path in pdf_files:

        result = parse_pdf(
            pdf_path=pdf_path,
            raw_root=RAW_DATA_DIR,
            processed_root=PROCESSED_DATA_DIR,
            converter=converter,
        )

        if (
            result.get("parser", {})
            .get("status")
            == "failed"
        ):
            failed += 1

        else:
            successful += 1

    print()
    print("=" * 60)
    print("Ingestion completed")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    run_ingestion_pipeline()