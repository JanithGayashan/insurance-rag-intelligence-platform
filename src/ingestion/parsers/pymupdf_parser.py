from __future__ import annotations

from pathlib import Path
from time import perf_counter

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


PARSER_NAME = "pymupdf"


def parse_pdf(
    pdf_path: str | Path
) -> ParseResult:

    start = perf_counter()

    path = Path(pdf_path)

    warnings = []

    try:

        import pymupdf

        pages = []
        elements = []
        tables = []

        with pymupdf.open(path) as document:

            pdf_metadata = dict(
                document.metadata or {}
            )

            for page_index, page in enumerate(document):

                page_number = page_index + 1

                # Main page text
                page_text = (
                    page.get_text(
                        "text",
                        sort=True
                    )
                    or ""
                )

                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text=page_text,
                        metadata={
                            "width": float(
                                page.rect.width
                            ),
                            "height": float(
                                page.rect.height
                            ),
                            "image_count": len(
                                page.get_images(
                                    full=True
                                )
                            ),
                        }
                    )
                )

                # Extract text blocks
                blocks = page.get_text(
                    "blocks",
                    sort=True
                )

                for block in blocks:

                    if len(block) < 5:
                        continue

                    block_text = str(
                        block[4] or ""
                    ).strip()

                    if not block_text:
                        continue

                    block_metadata = {}

                    if len(block) > 5:
                        block_metadata[
                            "block_number"
                        ] = block[5]

                    if len(block) > 6:
                        block_metadata[
                            "block_type"
                        ] = block[6]

                    elements.append(
                        ParsedElement(
                            type="text_block",
                            text=block_text,
                            page_number=page_number,
                            bbox=[
                                float(value)
                                for value
                                in block[:4]
                            ],
                            metadata=block_metadata
                        )
                    )

                # Extract tables
                try:

                    table_finder = (
                        page.find_tables()
                    )

                    for table_index, table in enumerate(
                        table_finder.tables
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
        "data/processed/pymupdf/sample.json"
    )

    print(
        result.success,
        result.processing_time_seconds
    )