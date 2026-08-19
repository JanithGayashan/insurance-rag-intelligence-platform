from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter

from src.ingestion.models import (
    ParseResult,
    ParsedElement,
    ParsedPage,
    ParsedTable,
)

from src.ingestion.utils import (
    html_table_to_rows,
    points_to_bbox,
    rows_to_markdown,
)


PARSER_NAME = "unstructured"


def parse_pdf(
    pdf_path: str | Path,
    strategy: str = "hi_res",
    languages: list[str] | None = None,
) -> ParseResult:

    start = perf_counter()

    path = Path(pdf_path)

    warnings = []

    languages = (
        languages or ["eng"]
    )

    try:

        from unstructured.partition.pdf import (
            partition_pdf
        )

        raw_elements = partition_pdf(
            filename=str(path),

            strategy=strategy,

            infer_table_structure=(
                strategy == "hi_res"
            ),

            languages=languages,

            include_page_breaks=False,
        )

        elements = []
        tables = []

        page_text_parts = defaultdict(
            list
        )

        max_page_number = 0

        for raw_element in raw_elements:

            element_type = (
                type(
                    raw_element
                ).__name__
            )

            text = str(
                getattr(
                    raw_element,
                    "text",
                    ""
                )
                or ""
            )

            metadata = (
                raw_element
                .metadata
                .to_dict()
            )

            page_number = metadata.get(
                "page_number"
            )

            if page_number is not None:

                page_number = int(
                    page_number
                )

                max_page_number = max(
                    max_page_number,
                    page_number
                )

                if text.strip():

                    page_text_parts[
                        page_number
                    ].append(text)

            coordinates = (
                metadata.get(
                    "coordinates"
                )
                or {}
            )

            bbox = points_to_bbox(
                coordinates.get(
                    "points"
                )
            )

            # Avoid storing very large
            # binary/table fields again.
            element_metadata = {
                key: value
                for key, value
                in metadata.items()

                if key not in {
                    "coordinates",
                    "text_as_html",
                    "image_base64",
                }
            }

            elements.append(
                ParsedElement(
                    type=element_type,
                    text=text,
                    page_number=page_number,
                    bbox=bbox,
                    metadata=element_metadata,
                )
            )

            # Table-specific handling
            if element_type == "Table":

                html = metadata.get(
                    "text_as_html"
                )

                rows = (
                    html_table_to_rows(
                        html
                    )
                )

                tables.append(
                    ParsedTable(
                        rows=rows,
                        page_number=page_number,
                        bbox=bbox,
                        markdown=rows_to_markdown(
                            rows
                        ),
                        html=html,
                    )
                )

        pages = [
            ParsedPage(
                page_number=page_number,

                text="\n\n".join(
                    page_text_parts.get(
                        page_number,
                        []
                    )
                ),
            )

            for page_number
            in range(
                1,
                max_page_number + 1
            )
        ]

        full_text = "\n\n".join(
            str(
                getattr(
                    element,
                    "text",
                    ""
                )
                or ""
            )

            for element
            in raw_elements

            if getattr(
                element,
                "text",
                None
            )
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
                "page_count":
                    max_page_number,

                "strategy":
                    strategy,

                "languages":
                    languages,

                "element_count":
                    len(elements),
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
        "data/raw/sample.pdf",
        strategy="hi_res"
    )

    save_parse_result(
        result,
        "data/processed/unstructured/sample.json"
    )

    print(
        result.success,
        result.processing_time_seconds
    )