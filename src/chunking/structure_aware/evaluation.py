"""Reproducible chunk diagnostics against the parsed document, not answer accuracy."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import re
from typing import Any
import unicodedata

from src.chunking.models import Chunk
from src.chunking.structure_aware.chunker import StructureAwareChunker
from src.chunking.structure_aware.context import resolve_contexts


VERSION = "structure-aware-quality-v2"


def words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def evaluate_chunks(document: Mapping[str, Any], chunks: Sequence[Chunk], limit: int) -> dict[str, Any]:
    helper = StructureAwareChunker()
    items = [node for node, _ in helper._walk_children(document["body"], helper._build_registry(document), set())]
    contexts = resolve_contexts(items)
    expected = {node["self_ref"]: node for node in items if node.get("label") not in {"page_header", "page_footer"} and (node.get("text") or node.get("label") == "table")}
    associated: dict[str, list[Chunk]] = defaultdict(list)
    invalid_refs = set()
    registry = helper._build_registry(document)
    for chunk in chunks:
        for ref in set(chunk.source_refs) | set(chunk.metadata.get("heading_source_refs", [])):
            if ref not in registry:
                invalid_refs.add(ref)
            associated[ref].append(chunk)
    missing = [ref for ref in expected if not associated[ref]]
    heading_refs = [ref for ref, node in expected.items() if node.get("label") in {"title", "section_header"}]
    missing_headings, missing_markers, text_errors = [], [], []
    total_words = missing_words = sentences = fragmented = 0
    for ref, node in expected.items():
        source_text = str(node.get("text", ""))
        if node.get("label") == "table":
            cells = node.get("data", {}).get("table_cells", [])
            source_text = " ".join(str(cell.get("text", "")) for cell in cells) if cells else helper._render_table(node)[0].removeprefix("[Table]\n")
        source_words = words(source_text)
        candidate_texts = [" ".join(words(chunk.text)) for chunk in associated[ref]]
        candidate_words = words(" ".join(chunk.text for chunk in associated[ref]))
        expected_count = Counter(source_words)
        deficit = sum((expected_count - Counter(candidate_words)).values())
        total_words += len(source_words)
        missing_words += deficit
        if deficit:
            text_errors.append({"source_ref": ref, "label": node.get("label"), "missing_word_count": deficit, "source_word_count": len(source_words)})
        if ref in heading_refs and " ".join(source_words) not in " ".join(candidate_texts):
            missing_headings.append(ref)
        marker = str(node.get("marker") or "").strip()
        if node.get("label") == "list_item" and marker:
            # Check the actual rendered marker next to this item's first words,
            # not just the marker copied into metadata.
            lead = re.findall(r"\S+", source_text)[:4]
            pattern = re.escape(marker) + r"\s+" + r"\s+".join(re.escape(word) for word in lead)
            if not any(re.search(pattern, chunk.text, re.IGNORECASE) for chunk in associated[ref]):
                missing_markers.append(ref)
        if node.get("label") in {"text", "paragraph", "list_item"}:
            for sentence in re.split(r"(?<=[.!?])\s+", source_text):
                sentence_words = words(sentence)
                if len(sentence_words) < 3:
                    continue
                sentences += 1
                if not any(" ".join(sentence_words) in text for text in candidate_texts):
                    fragmented += 1
    crossings = []
    for chunk in chunks:
        refs = chunk.metadata.get("body_source_refs", chunk.source_refs)
        sections = {contexts[ref].refs[-1] if contexts[ref].refs else "introduction" for ref in refs if ref in expected and expected[ref].get("label") not in {"title", "section_header"}}
        if len(sections) > 1:
            crossings.append(chunk.chunk_id)
    bodies = [str(chunk.metadata.get("body_text", chunk.text)).strip() for chunk in chunks]
    duplicate_count = sum(count - 1 for text, count in Counter(bodies).items() if text and count > 1)
    pages = sorted({page for chunk in chunks for page in chunk.page_numbers})
    expected_pages = sorted({p["page_no"] for node in expected.values() for p in node.get("prov", []) if "page_no" in p})
    markers = sum(node.get("label") == "list_item" and bool(node.get("marker")) for node in expected.values())
    oversize = [chunk.chunk_id for chunk in chunks if (chunk.token_count or 0) > limit]
    bad_tokens = [chunk.chunk_id for chunk in chunks if chunk.token_count != len(helper.encoding.encode(chunk.text))]
    bad_unicode = [chunk.chunk_id for chunk in chunks if "\ufffd" in chunk.text]
    failures = len(missing) + len(missing_headings) + len(missing_markers) + len(oversize) + len(invalid_refs) + len(crossings) + len(bad_tokens) + len(bad_unicode) + missing_words
    return {
        "version": VERSION, "status": "PASS" if not failures else "REVIEW",
        "reference": "active-parsed-docling-document", "human_review_status": "NOT_HUMAN_REVIEWED",
        "source_elements": len(expected), "represented_source_elements": len(expected) - len(missing),
        "source_element_coverage": (len(expected) - len(missing)) / len(expected) if expected else None,
        "missing_source_refs": missing, "invalid_source_refs": sorted(invalid_refs),
        "headings": len(heading_refs), "missing_heading_refs": missing_headings,
        "list_markers": markers, "missing_list_marker_refs": missing_markers,
        "source_words": total_words, "missing_source_words": missing_words,
        "missing_word_rate": missing_words / total_words if total_words else None,
        "text_coverage_errors": text_errors,
        "sentences_checked": sentences, "fragmented_or_missing_sentences": fragmented,
        "sentence_fragmentation_rate": fragmented / sentences if sentences else None,
        "section_crossing_chunks": crossings, "oversized_chunks": oversize,
        "token_count_mismatches": bad_tokens, "replacement_character_chunks": bad_unicode,
        "tiny_chunks_under_20_tokens": sum((chunk.token_count or 0) < 20 for chunk in chunks),
        "duplicate_chunk_bodies": duplicate_count, "duplicate_body_rate": duplicate_count / len(chunks) if chunks else None,
        "fallback_chunks": sum(bool(chunk.metadata.get("oversized_block_split")) for chunk in chunks),
        "hard_token_fallback_chunks": sum(bool(chunk.metadata.get("fallback_token_split")) for chunk in chunks),
        "inferred_heading_context_chunks": sum(bool(chunk.metadata.get("heading_context_inferred")) for chunk in chunks),
        "table_source_refs": [ref for ref, node in expected.items() if node.get("label") == "table"],
        "pages_represented": pages, "missing_source_pages": sorted(set(expected_pages) - set(pages)),
        "limitations": [
            "Diagnostics compare chunks with the parsed Docling JSON; parsing errors remain outside this evaluation.",
            "Heading groups inferred from consecutive labels require human review; zero crossings proves consistency with these boundaries, not their semantic correctness.",
            "Missing-word rate uses per-element word multisets over associated chunks. Repeated context/overlap can mask omissions; inspect text and source references.",
            "Sentence fragmentation uses punctuation and normalized phrase containment. Abbreviations, tables and long sentences need separate review; it includes missing sentences.",
            "Duplicate/tiny chunks are diagnostics; repeated headings, table headers and overlap can be intentional.",
            "These checks do not measure retrieval Recall@K, answer accuracy or source faithfulness.",
        ],
    }
