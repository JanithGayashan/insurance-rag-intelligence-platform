from pathlib import Path

from src.ingestion.docling_parser import (
    create_converter,
    parse_pdf,
)


RAW_DATA_DIR = Path("data/raw/allianz")
PROCESSED_DATA_DIR = Path("data/processed/allianz")


def discover_pdfs(
    raw_directory: Path,
) -> list[Path]:
    """
    Recursively discover all PDF files
    inside the raw data directory.
    """

    pdf_files = sorted(
        raw_directory.rglob("*.pdf")
    )

    return pdf_files


def run_ingestion_pipeline() -> None:
    """
    Run the document ingestion pipeline.

    The pipeline:
    1. Discovers all PDF files.
    2. Creates one reusable Docling converter.
    3. Parses each PDF.
    4. Counts successful and failed documents.
    """

    print("Starting document ingestion")
    print("=" * 60)

    # --------------------------------
    # 1. Discover PDF files
    # --------------------------------

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

    # --------------------------------
    # 2. Create Docling converter once
    # --------------------------------

    converter = create_converter()

    successful = 0
    failed = 0

    # --------------------------------
    # 3. Process each PDF
    # --------------------------------

    for pdf_path in pdf_files:

        result = parse_pdf(
            pdf_path=pdf_path,
            raw_root=RAW_DATA_DIR,
            processed_root=PROCESSED_DATA_DIR,
            converter=converter,
        )

        # --------------------------------
        # 4. Check parser status
        # --------------------------------

        parser_status = result.get(
            "parser",
            {}
        ).get("status")

        if parser_status == "failed":
            failed += 1
        else:
            successful += 1

    # --------------------------------
    # 5. Print summary
    # --------------------------------

    print()
    print("=" * 60)
    print("Ingestion completed")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    run_ingestion_pipeline()