from __future__ import annotations

from pathlib import Path
from time import perf_counter
from io import BytesIO

from src.ingestion.parsers.conditional_ocr import prepare_pdf

from src.ingestion.models import (
    ParseResult,
    ParsedElement,
    ParsedPage,
    ParsedTable,
)

from src.ingestion.utils import (
    normalize_rows,
    rows_to_markdown,
)


PARSER_NAME = "pdfplumber"


def parse_pdf(
    pdf_path: str | Path,
    *,
    ocr_mode: str = "auto",
    ocr_language: str = "eng",
    ocr_dpi: int = 300,
    ocr_timeout: float = 60,
) -> ParseResult:

    start = perf_counter()

    path = Path(pdf_path)

    warnings = []

    try:

        import pdfplumber

        pages = []
        elements = []
        tables = []

        with prepare_pdf(path, ocr_mode=ocr_mode, ocr_language=ocr_language,
                         ocr_dpi=ocr_dpi, ocr_timeout=ocr_timeout) as prepared, pdfplumber.open(
                             BytesIO(prepared.document.tobytes()) if prepared.changed else path) as pdf:

            pdf_metadata = dict(
                pdf.metadata or {}
            )

            for page_index, page in enumerate(
                pdf.pages
            ):

                page_number = (
                    page_index + 1
                )

                page_text = (
                    page.extract_text(
                        layout=True
                    )
                    or ""
                )

                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text=page_text,
                        metadata={
                            "width": float(
                                page.width
                            ),
                            "height": float(
                                page.height
                            ),
                            "ocr": prepared.pages[page_index],
                        }
                    )
                )

                # Keep low-level word positions.
                # Do not invent headings/sections.
                words = (
                    page.extract_words()
                    or []
                )

                for word in words:

                    elements.append(
                        ParsedElement(
                            type="word",
                            text=str(
                                word.get(
                                    "text",
                                    ""
                                )
                            ),
                            page_number=page_number,
                            bbox=[
                                float(
                                    word["x0"]
                                ),
                                float(
                                    word["top"]
                                ),
                                float(
                                    word["x1"]
                                ),
                                float(
                                    word["bottom"]
                                ),
                            ]
                        )
                    )

                try:

                    found_tables = (
                        page.find_tables()
                    )

                    for table_index, table in enumerate(
                        found_tables
                    ):

                        rows = normalize_rows(
                            table.extract()
                        )

                        bbox = None

                        if getattr(
                            table,
                            "bbox",
                            None
                        ) is not None:

                            bbox = [
                                float(value)
                                for value
                                in table.bbox
                            ]

                        tables.append(
                            ParsedTable(
                                rows=rows,
                                page_number=page_number,
                                bbox=bbox,
                                markdown=rows_to_markdown(
                                    rows
                                ),
                                metadata={
                                    "table_index_on_page":
                                        table_index
                                }
                            )
                        )

                except Exception as exc:

                    warnings.append(
                        f"Page {page_number}: "
                        f"table extraction failed: {exc}"
                    )

        warnings.extend(prepared.warnings)
        full_text = "\n\n".join(
            page.text
            for page in pages
        )

        return ParseResult(
            parser_name=PARSER_NAME,
            source_file=str(
                path.resolve()
            ),
            success=True,
            text=full_text,
            pages=pages,
            elements=elements,
            tables=tables,
            metadata={
                "page_count": len(pages),
                "pdf_metadata": pdf_metadata,
                "ocr": prepared.summary(),
            },
            processing_time_seconds=(
                perf_counter() - start
            ),
            warnings=warnings,
        )

    except Exception as exc:

        return ParseResult(
            parser_name=PARSER_NAME,
            source_file=str(path),
            success=False,
            processing_time_seconds=(
                perf_counter() - start
            ),
            warnings=warnings,
            error=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


if __name__ == "__main__":

    from src.ingestion.utils import (
        save_parse_result
    )

    result = parse_pdf(
        "data/raw/sample.pdf"
    )

    save_parse_result(
        result,
        "data/processed/pdfplumber/sample.json"
    )

    print(
        result.success,
        result.processing_time_seconds
    )
