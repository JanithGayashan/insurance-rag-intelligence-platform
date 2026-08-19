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
)


PARSER_NAME = "docling"


def label_to_string(label) -> str:

    return str(
        getattr(
            label,
            "value",
            label
        )
    )


def parse_pdf(
    pdf_path: str | Path
) -> ParseResult:

    start = perf_counter()

    path = Path(pdf_path)

    warnings = []

    try:

        from docling.document_converter import (
            DocumentConverter
        )

        converter = (
            DocumentConverter()
        )

        conversion_result = (
            converter.convert(path)
        )

        document = (
            conversion_result.document
        )

        page_count = (
            document.num_pages()
        )

        pages = []
        elements = []
        tables = []

        # Page-level output
        for page_number in range(
            1,
            page_count + 1
        ):

            pages.append(
                ParsedPage(
                    page_number=page_number,

                    text=document.export_to_text(
                        page_no=page_number
                    ),

                    markdown=document.export_to_markdown(
                        page_no=page_number
                    ),
                )
            )

        # Semantic document elements
        for item, level in document.iterate_items():

            text = getattr(
                item,
                "text",
                None
            )

            if not text:
                continue

            provenance = None

            if getattr(
                item,
                "prov",
                None
            ):
                provenance = (
                    item.prov[0]
                )

            page_number = None
            bbox = None

            if provenance:

                page_number = (
                    provenance.page_no
                )

                if provenance.bbox:

                    bbox = [
                        float(value)
                        for value
                        in provenance.bbox.as_tuple()
                    ]

            elements.append(
                ParsedElement(
                    type=label_to_string(
                        getattr(
                            item,
                            "label",
                            type(item).__name__
                        )
                    ),
                    text=str(text),
                    page_number=page_number,
                    bbox=bbox,
                    metadata={
                        "hierarchy_level":
                            level
                    }
                )
            )

        # Structured tables
        for table_index, table in enumerate(
            document.tables
        ):

            provenance = None

            if getattr(
                table,
                "prov",
                None
            ):
                provenance = (
                    table.prov[0]
                )

            page_number = None
            bbox = None

            if provenance:

                page_number = (
                    provenance.page_no
                )

                if provenance.bbox:

                    bbox = [
                        float(value)
                        for value
                        in provenance.bbox.as_tuple()
                    ]

            try:

                dataframe = (
                    table.export_to_dataframe(
                        doc=document
                    )
                    .fillna("")
                )

                rows = [
                    list(
                        map(
                            str,
                            dataframe.columns
                        )
                    )
                ]

                rows.extend(
                    [
                        [
                            str(value)
                            for value
                            in row
                        ]
                        for row
                        in dataframe
                        .to_numpy()
                        .tolist()
                    ]
                )

                rows = normalize_rows(
                    rows
                )

            except Exception as exc:

                warnings.append(
                    f"Table {table_index}: "
                    f"DataFrame export failed: "
                    f"{exc}"
                )

                rows = []

            try:

                table_html = (
                    table.export_to_html(
                        doc=document
                    )
                )

            except Exception as exc:

                warnings.append(
                    f"Table {table_index}: "
                    f"HTML export failed: "
                    f"{exc}"
                )

                table_html = None

            try:

                table_markdown = (
                    table.export_to_markdown(
                        doc=document
                    )
                )

            except Exception as exc:

                warnings.append(
                    f"Table {table_index}: "
                    f"Markdown export failed: "
                    f"{exc}"
                )

                table_markdown = None

            tables.append(
                ParsedTable(
                    rows=rows,
                    page_number=page_number,
                    bbox=bbox,
                    markdown=table_markdown,
                    html=table_html,
                    metadata={
                        "table_index":
                            table_index
                    }
                )
            )

        return ParseResult(
            parser_name=PARSER_NAME,

            source_file=str(
                path.resolve()
            ),

            success=True,

            text=document.export_to_text(),

            markdown=(
                document.export_to_markdown()
            ),

            pages=pages,
            elements=elements,
            tables=tables,

            metadata={
                "page_count":
                    page_count,

                "document_name":
                    document.name,

                "conversion_status":
                    str(
                        getattr(
                            conversion_result,
                            "status",
                            ""
                        )
                    ),
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
        "data/processed/docling/sample.json"
    )

    print(
        result.success,
        result.processing_time_seconds
    )