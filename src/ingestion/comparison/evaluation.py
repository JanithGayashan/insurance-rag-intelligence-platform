from __future__ import annotations

from collections import Counter
import re
import unicodedata

from rapidfuzz.distance import Levenshtein


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", normalize(text))


def token_agreement(reference: str, actual: str) -> dict:
    expected, observed = Counter(tokens(reference)), Counter(tokens(actual))
    overlap = sum((expected & observed).values())
    recall = overlap / sum(expected.values()) if expected else None
    precision = overlap / sum(observed.values()) if observed else None
    return {"token_recall": recall, "token_precision": precision,
            "token_f1": 2 * precision * recall / (precision + recall) if precision and recall else 0.0 if expected else None}


def edit_rate(reference, actual) -> float | None:
    if not reference:
        return None
    return Levenshtein.distance(reference, actual) / len(reference)


def text_for_scoring(text: str) -> str:
    """Remove recognized Markdown table syntax, preserving words/punctuation.

    Some adapters export tables as Markdown even in their plain-text field.
    Formatting delimiters are not OCR character errors.
    """
    lines=text.splitlines()
    divider=lambda line:bool(re.fullmatch(r"\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*",line))
    cleaned=[]
    in_table=False
    for index,line in enumerate(lines):
        if divider(line):
            in_table=True
            continue
        table_row=line.strip().startswith("|") and line.strip().endswith("|")
        if table_row and (in_table or index+1<len(lines) and divider(lines[index+1])):
            cells=re.split(r"(?<!\\)\|",line.strip()[1:-1])
            line=" ".join(re.sub(r"<br\s*/?>"," ",cell,flags=re.I).replace(r"\|","|") for cell in cells)
        else:
            in_table=False
        cleaned.append(line)
    return normalize("\n".join(cleaned))


def evaluate_result(result: dict, source_pages: list[dict], reference: dict | None = None) -> dict:
    """Diagnostics are not ground truth. Unmeasured quality is null, never zero."""
    if not result["success"]:
        return {"status": "FAILED", "diagnostics": None, "reference": None, "pages": [],
                "standard": {"status": "FAILED"}}
    expected_numbers = {page["page_number"] for page in source_pages}
    pages = {page["page_number"]: page for page in result["pages"]}
    page_rows = []
    for source in source_pages:
        number = source["page_number"]
        text = pages.get(number, {}).get("text", "")
        page_rows.append({"page_number": number, "present": number in pages,
            "characters": len(text), "words": len(tokens(text)), "source_native_characters": len(source["text"]),
            "source_has_images": source["has_images"],
            "ocr": pages.get(number, {}).get("metadata", {}).get("ocr"),
            "native_text_agreement": token_agreement(source["text"], text) if len(tokens(source["text"])) >= 10 else None})
    text = result.get("text", "")
    nonempty = [normalize(line) for line in text.splitlines() if normalize(line)]
    duplicates = sum(count - 1 for line, count in Counter(nonempty).items() if len(line) >= 40 and count > 1)
    reference_rows = []
    if reference:
        for item in reference.get("checks", []):
            number = item["page_number"]
            candidates = [pages.get(number, {}).get("text", "")]
            if item["kind"] == "table_cell":
                candidates = [cell for table in result["tables"] if table.get("page_number") == number
                              for row in table["rows"] for cell in row]
            elif item["kind"] == "heading":
                candidates = [element["text"] for element in result["elements"]
                              if element.get("page_number") == number
                              and element["type"].lower() in {"title", "section_header", "heading", "header"}]
            matched = any(normalize(item["text"]) in normalize(candidate) for candidate in candidates)
            reference_rows.append({**item, "matched": matched})
    full_pages = []
    for item in (reference or {}).get("full_pages", []):
        actual = pages.get(item["page_number"], {}).get("text", "")
        expected_text,actual_text=text_for_scoring(item["text"]),text_for_scoring(actual)
        full_pages.append({"page_number": item["page_number"],
            "reference_words": len(tokens(expected_text)), "reference_characters": len(expected_text),
            "word_error_rate": edit_rate(tokens(expected_text), tokens(actual_text)),
            "character_error_rate": edit_rate(expected_text,actual_text),
            "normalized_edit_distance": Levenshtein.normalized_distance(expected_text,actual_text),
            "edit_distance_units": max(len(expected_text),len(actual_text))})
    kinds = {kind: [item for item in reference_rows if item["kind"] == kind]
             for kind in ("passage", "heading", "table_cell")}
    from src.ingestion.comparison.standard_metrics import standard_evaluation
    return {"status": "EVALUATED", "standard": standard_evaluation(result,source_pages,reference),
        "operations": {"processing_seconds": result.get("processing_time_seconds"),
            "seconds_per_page": result["processing_time_seconds"]/len(result["pages"]) if result.get("processing_time_seconds") is not None and result["pages"] else None,
            "warning_count": len(result.get("warnings",[])), "usage": result.get("metadata",{}).get("usage"),
            "ocr_failed_pages": result.get("metadata",{}).get("ocr",{}).get("failed_pages",[]),
            "peak_memory_mib": None, "cost_usd": None,
            "limitations": "Single-run wall time includes initialization. Memory/currency cost were not measured in the saved runs; null does not mean zero."},
        "diagnostics": {
        "page_coverage": len(expected_numbers & set(pages)) / len(expected_numbers) if expected_numbers else None,
        "missing_pages": sorted(expected_numbers - set(pages)), "unexpected_pages": sorted(set(pages) - expected_numbers),
        "empty_pages": [number for number, page in pages.items() if not page.get("text", "").strip()],
        "text_characters": len(text), "text_words": len(tokens(text)),
        "replacement_characters": text.count("\ufffd"), "repeated_long_lines": duplicates,
        "elements": len(result["elements"]), "tables": len(result["tables"]),
        "unlocated_elements": sum(item.get("page_number") is None for item in result["elements"]),
        "unlocated_tables": sum(item.get("page_number") is None for item in result["tables"]),
    }, "reference": {"status": (reference or {}).get("review_status", "NOT_PROVIDED"),
        "check_count": len(reference_rows), "matched_checks": sum(item["matched"] for item in reference_rows),
        "by_kind": {kind: {"matched": sum(item["matched"] for item in items), "total": len(items),
                                  "coverage": sum(item["matched"] for item in items) / len(items) if items else None}
                    for kind, items in kinds.items()},
        "checks": reference_rows, "full_page_error_rates": full_pages, "reference_page_count": len(full_pages),
        "error_rate_aggregation": "Total edits divided by total reference units on annotated pages only",
        "word_error_rate": sum(item["word_error_rate"] * item["reference_words"] for item in full_pages) / sum(item["reference_words"] for item in full_pages) if full_pages else None,
        "character_error_rate": sum(item["character_error_rate"] * item["reference_characters"] for item in full_pages) / sum(item["reference_characters"] for item in full_pages) if full_pages else None,
        "normalized_edit_distance": sum(item["normalized_edit_distance"]*item["edit_distance_units"] for item in full_pages)/sum(item["edit_distance_units"] for item in full_pages) if full_pages else None,
    }, "pages": page_rows}
