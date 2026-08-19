from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)


def create_converter() -> DocumentConverter:
    """
    Create and configure the Docling PDF converter.

    We keep this function separate so that later we can
    experiment with OCR, table recognition, and other
    pipeline options without changing the ingestion logic.
    """

    pipeline_options = PdfPipelineOptions()

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        },
    )

    return converter


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate a SHA-256 fingerprint for the source PDF.

    This will later help us detect:
    - duplicate documents
    - changed documents
    - document versions
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def parse_pdf(
    pdf_path: Path,
    raw_root: Path,
    processed_root: Path,
    converter: DocumentConverter,
) -> dict:
    """
    Parse one PDF using Docling and store its outputs.

    Returns metadata describing the ingestion result.
    """

    start_time = perf_counter()

    pdf_path = pdf_path.resolve()
    raw_root = raw_root.resolve()
    processed_root = processed_root.resolve()

    # Example:
    #
    # raw:
    # data/raw/allianz/motor/policy.pdf
    #
    # output:
    # data/processed/allianz/motor/policy/

    relative_path = pdf_path.relative_to(raw_root)

    output_dir = (
        processed_root
        / relative_path.parent
        / pdf_path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Parsing: {relative_path}")

    try:
        # -----------------------------
        # 1. Convert PDF using Docling
        # -----------------------------

        conversion_result = converter.convert(
            pdf_path
        )

        document = conversion_result.document

        # -----------------------------
        # 2. Output paths
        # -----------------------------

        json_path = output_dir / "document.json"
        markdown_path = output_dir / "document.md"
        text_path = output_dir / "document.txt"
        metadata_path = output_dir / "metadata.json"

        # -----------------------------
        # 3. Preserve native Docling JSON
        # -----------------------------

        document.save_as_json(
            json_path
        )

        # -----------------------------
        # 4. Human-readable Markdown
        # -----------------------------

        document.save_as_markdown(
            markdown_path
        )

        # -----------------------------
        # 5. Plain text representation
        # -----------------------------

        text = document.export_to_text()

        text_path.write_text(
            text,
            encoding="utf-8",
        )

        # -----------------------------
        # 6. Document metadata
        # -----------------------------

        processing_time = (
            perf_counter() - start_time
        )

        status = getattr(
            conversion_result.status,
            "value",
            str(conversion_result.status),
        )

        metadata = {
            "source": {
                "file_name": pdf_path.name,
                "relative_path": str(relative_path),
                "file_size_bytes": pdf_path.stat().st_size,
                "sha256": calculate_sha256(pdf_path),
            },

            "document": {
                "page_count": document.num_pages(),
                "table_count": len(document.tables),
            },

            "parser": {
                "name": "docling",
                "conversion_status": status,
                "processing_time_seconds": round(
                    processing_time,
                    3,
                ),
            },

            "outputs": {
                "json": str(json_path),
                "markdown": str(markdown_path),
                "text": str(text_path),
            },
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            f"Completed: {relative_path} "
            f"({processing_time:.2f}s)"
        )

        return metadata

    except Exception as error:

        processing_time = (
            perf_counter() - start_time
        )

        error_metadata = {
            "source": {
                "file_name": pdf_path.name,
                "relative_path": str(relative_path),
            },

            "parser": {
                "name": "docling",
                "status": "failed",
                "processing_time_seconds": round(
                    processing_time,
                    3,
                ),
            },

            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }

        error_path = (
            output_dir / "error.json"
        )

        error_path.write_text(
            json.dumps(
                error_metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            f"FAILED: {relative_path}"
        )

        print(error)

        return error_metadata