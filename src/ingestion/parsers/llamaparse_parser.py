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
    safe_model_dump,
    xywh_to_bbox,
)


PARSER_NAME = "llamaparse"


def parse_pdf(
    pdf_path: str | Path,
    tier: str = "agentic",
) -> ParseResult:

    start = perf_counter()

    path = Path(pdf_path)

    warnings = []

    try:

        from dotenv import (
            load_dotenv
        )

        from llama_cloud import (
            LlamaCloud
        )

        load_dotenv()

        # Reads LLAMA_CLOUD_API_KEY
        # from environment/.env
        client = LlamaCloud()

        uploaded_file = (
            client.files.create(
                file=path,
                purpose="parse",
            )
        )

        result = (
            client.parsing.parse(

                file_id=uploaded_file.id,

                tier=tier,

                version="latest",

                output_options={
                    "markdown": {
                        "tables": {
                            "output_tables_as_markdown":
                                True
                        }
                    }
                },

                expand=[
                    "text",
                    "markdown",
                    "items",
                    "metadata",
                    "usage",
                ],
            )
        )

        # ----------------------
        # Page text
        # ----------------------

        text_pages = {
            int(page.page_number):
                page.text

            for page
            in result.text.pages
        }

        # ----------------------
        # Page markdown
        # ----------------------

        markdown_pages = {
            int(page.page_number):
                page.markdown

            for page
            in result.markdown.pages
        }

        # ----------------------
        # Page metadata
        # ----------------------

        page_metadata = {}

        if getattr(
            result,
            "metadata",
            None
        ) is not None:

            for page in result.metadata.pages:

                page_metadata[
                    int(page.page_number)
                ] = safe_model_dump(
                    page
                )

        # ----------------------
        # Structured items
        # ----------------------

        item_pages = {
            int(page.page_number):
                page

            for page
            in result.items.pages
        }

        all_page_numbers = sorted(
            set(text_pages)
            | set(markdown_pages)
            | set(item_pages)
            | set(page_metadata)
        )

        pages = []
        elements = []
        tables = []

        for page_number in all_page_numbers:

            pages.append(
                ParsedPage(
                    page_number=page_number,

                    text=text_pages.get(
                        page_number,
                        ""
                    ),

                    markdown=markdown_pages.get(
                        page_number
                    ),

                    metadata=page_metadata.get(
                        page_number,
                        {}
                    ),
                )
            )

            item_page = (
                item_pages.get(
                    page_number
                )
            )

            if item_page is None:
                continue

            for item_index, item in enumerate(
                item_page.items
            ):

                item_type = str(
                    getattr(
                        item,
                        "type",
                        type(item).__name__
                    )
                )

                item_markdown = getattr(
                    item,
                    "md",
                    None
                )

                item_value = getattr(
                    item,
                    "value",
                    None
                )

                item_text = str(
                    item_value
                    or item_markdown
                    or ""
                )

                bbox = xywh_to_bbox(
                    getattr(
                        item,
                        "bbox",
                        None
                    )
                )

                element_metadata = {
                    "item_index":
                        item_index
                }

                if getattr(
                    item,
                    "level",
                    None
                ) is not None:

                    element_metadata[
                        "level"
                    ] = item.level

                elements.append(
                    ParsedElement(
                        type=item_type,
                        text=item_text,
                        page_number=page_number,
                        bbox=bbox,
                        metadata=element_metadata,
                    )
                )

                # Structured table
                if item_type == "table":

                    rows = normalize_rows(
                        getattr(
                            item,
                            "rows",
                            None
                        )
                    )

                    tables.append(
                        ParsedTable(
                            rows=rows,

                            page_number=
                                page_number,

                            bbox=bbox,

                            markdown=getattr(
                                item,
                                "md",
                                None
                            ),

                            html=getattr(
                                item,
                                "html",
                                None
                            ),

                            metadata={
                                "item_index":
                                    item_index
                            },
                        )
                    )

        full_text = "\n\n".join(
            text_pages[
                page_number
            ]

            for page_number
            in sorted(text_pages)
        )

        full_markdown = "\n\n".join(
            markdown_pages[
                page_number
            ]

            for page_number
            in sorted(markdown_pages)
        )

        return ParseResult(
            parser_name=PARSER_NAME,

            source_file=str(
                path.resolve()
            ),

            success=True,

            text=full_text,

            markdown=full_markdown,

            pages=pages,
            elements=elements,
            tables=tables,

            metadata={
                "page_count":
                    len(pages),

                "tier":
                    tier,

                "uploaded_file_id":
                    uploaded_file.id,

                "job":
                    safe_model_dump(
                        getattr(
                            result,
                            "job",
                            None
                        )
                    ),

                "usage":
                    safe_model_dump(
                        getattr(
                            result,
                            "usage",
                            None
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
        "data/raw/sample.pdf",
        tier="agentic"
    )

    save_parse_result(
        result,
        "data/processed/llamaparse/sample.json"
    )

    print(
        result.success,
        result.processing_time_seconds
    )