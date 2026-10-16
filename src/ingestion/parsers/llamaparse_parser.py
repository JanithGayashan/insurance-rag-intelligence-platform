from __future__ import annotations

import json
import hashlib
from pathlib import Path
from time import perf_counter

from src.ingestion.models import ParseResult, ParsedElement, ParsedPage, ParsedTable
from src.ingestion.utils import normalize_rows, xywh_to_bbox

PARSER_NAME = "llamaparse"


def normalize_response(data: dict, source: Path, tier: str, elapsed: float) -> ParseResult:
    """Normalize the saved SDK response independently of the paid request."""
    job = data.get("job") or {}
    status = str(job.get("status", "completed")).lower()
    if status not in {"completed", "success", "succeeded"}:
        raise ValueError(f"LlamaParse job ended with status {status}")
    text_pages = {int(page["page_number"]): page.get("text", "") for page in (data.get("text") or {}).get("pages", [])}
    markdown_pages = {int(page["page_number"]): page.get("markdown", "") for page in (data.get("markdown") or {}).get("pages", [])}
    item_pages = {int(page["page_number"]): page for page in (data.get("items") or {}).get("pages", [])}
    metadata_pages = {int(page["page_number"]): page for page in (data.get("metadata") or {}).get("pages", [])}
    numbers = sorted(set(text_pages) | set(markdown_pages) | set(item_pages) | set(metadata_pages))
    elements, tables = [], []
    for number, page in item_pages.items():
        for index, item in enumerate(page.get("items", [])):
            kind = item.get("type", "unknown")
            value = item.get("value") or item.get("text") or item.get("md") or ""
            if isinstance(value, list):
                value = "\n".join(str(part) for part in value)
            bbox = xywh_to_bbox(item.get("bbox"))
            elements.append(ParsedElement(type=kind, text=str(value), page_number=number,
                bbox=bbox, metadata={"level": item.get("level"), "item_index": index}))
            if kind == "table":
                tables.append(ParsedTable(rows=normalize_rows(item.get("rows")), page_number=number,
                    bbox=bbox, markdown=item.get("md"), html=item.get("html")))
    return ParseResult(parser_name=PARSER_NAME, source_file=str(source), success=True,
        text="\n\n".join(text_pages.get(n, "") for n in numbers),
        markdown="\n\n".join(markdown_pages.get(n, "") for n in numbers),
        pages=[ParsedPage(page_number=n, text=text_pages.get(n, ""), markdown=markdown_pages.get(n),
                          metadata=metadata_pages.get(n, {})) for n in numbers],
        elements=elements, tables=tables, processing_time_seconds=elapsed,
        metadata={"page_count": len(numbers), "tier": tier, "job": job,
                  "usage": data.get("usage") or job.get("usage") or (data.get("metadata") or {}).get("usage")})


def parse_pdf(pdf_path: str | Path, tier: str = "agentic", version: str = "latest",
              native_output_path: Path | None = None) -> ParseResult:
    from src.configuration.dataset import get_dataset_configuration
    get_dataset_configuration()
    source = Path(pdf_path)
    started = perf_counter()
    try:
        from llama_cloud import LlamaCloud
        # Cache the raw paid response before normalization, so adapter repairs do not upload again.
        signature = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "tier": tier, "version": version}
        receipt = native_output_path.with_name("cloud_request.json") if native_output_path else None
        matches = receipt is not None and receipt.is_file() and json.loads(receipt.read_text(encoding="utf-8")) == signature
        if native_output_path is not None and native_output_path.is_file() and matches:
            data = json.loads(native_output_path.read_text(encoding="utf-8"))
            previous_output = native_output_path.with_name("result.json")
            previous_seconds = json.loads(previous_output.read_text(encoding="utf-8")).get("processing_time_seconds") if previous_output.is_file() else None
        else:
            with LlamaCloud(max_retries=0, timeout=600) as client:
                uploaded = client.files.create(file=source, purpose="parse")
                response = client.parsing.parse(file_id=uploaded.id, tier=tier, version=version,
                    timeout=900, expand=["text", "markdown", "items", "metadata"],
                    output_options={"markdown": {"tables": {"output_tables_as_markdown": True}}})
                data = response.model_dump(mode="json")
                if native_output_path is not None:
                    native_output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    receipt.write_text(json.dumps(signature), encoding="utf-8")
            previous_seconds = None
        result = normalize_response(data, source, tier, previous_seconds if previous_seconds is not None else perf_counter() - started)
        result.metadata.update(requested_version=version, raw_response_reused=bool(matches))
        return result
    except Exception as error:
        # SDK errors may include request bodies; never serialize credentials or authorization headers.
        status_code = getattr(error, "status_code", None)
        message = f"{type(error).__name__}" + (f" (HTTP {status_code})" if status_code else f": {str(error)[:300]}")
        import os
        secret = os.getenv("LLAMA_CLOUD_API_KEY", "")
        if secret:
            message = message.replace(secret, "[redacted]")
        return ParseResult(parser_name=PARSER_NAME, source_file=str(source), success=False,
                           processing_time_seconds=perf_counter() - started, error=message)
