"""Conservative heading context shared by the two structure-aware adapters."""
from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Mapping, Sequence
from typing import Any


@dataclass(frozen=True)
class HeadingContext:
    headings: tuple[str, ...]
    refs: tuple[str, ...]
    inferred: bool = False


def resolve_contexts(items: Sequence[Mapping[str, Any]]) -> dict[str, HeadingContext]:
    """Keep consecutive labels together without discarding either heading.

    Explicit levels are otherwise respected. A consecutive same-level run is
    context inferred from document order, not a claim of verified hierarchy.
    """
    stack: list[tuple[int, str, str, bool]] = []
    result: dict[str, HeadingContext] = {}
    previous_heading = False
    for item in items:
        label = item.get("label")
        ref = str(item.get("self_ref", ""))
        if label in {"page_header", "page_footer"}:
            continue
        if label in {"title", "section_header"}:
            text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
            if not text:
                continue
            raw_level = None
            for source in (item, item.get("metadata", {})):
                if isinstance(source, Mapping):
                    raw_level = next((source[key] for key in ("level", "heading_level") if isinstance(source.get(key), int) and source[key] > 0), None)
                    if raw_level is not None:
                        break
            level = raw_level if isinstance(raw_level, int) and raw_level > 0 else (0 if label == "title" else 1)
            if label == "title":
                level = 0
            inferred = bool(previous_heading and stack and level <= stack[-1][0] and label != "title")
            if inferred:
                # Treat an ambiguous heading run as a compound context.
                level = stack[-1][0] + 1
            else:
                stack = [heading for heading in stack if heading[0] < level]
            stack.append((level, text, ref, inferred))
            previous_heading = True
        elif item.get("text") or label == "table":
            previous_heading = False
        result[ref] = HeadingContext(tuple(h[1] for h in stack), tuple(h[2] for h in stack), any(h[3] for h in stack))
    return result
