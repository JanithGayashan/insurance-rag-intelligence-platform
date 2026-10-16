"""Token-budget helpers that never decode incomplete UTF-8 token slices."""
from __future__ import annotations

from dataclasses import dataclass
import re

import tiktoken


@dataclass(frozen=True)
class TextPart:
    text: str
    start: int
    end: int
    hard_split: bool


def render_text(prefix: str, body: str) -> str:
    return f"{prefix}\n\n{body}" if prefix else body


def fitting_prefix(text: str, encoding: tiktoken.Encoding, budget: int) -> str:
    """Return a character-aligned prefix within a token budget."""
    if budget <= 0:
        return ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if len(encoding.encode(text[:middle])) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def section_prefix(headings: list[str] | tuple[str, ...], encoding: tiktoken.Encoding, limit: int) -> str:
    # Leave room for at least one multi-token Unicode character and separators.
    budget = min(limit // 3, max(0, limit - 8))
    return fitting_prefix("Section: " + " > ".join(headings), encoding, budget) if headings else ""


def split_text(text: str, encoding: tiktoken.Encoding, limit: int, *, prefix: str = "", overlap: int = 0) -> list[TextPart]:
    """Split at character boundaries; prefer paragraphs/sentences when possible.

    Each *rendered* part is checked, including prefix/separators. An impossible
    budget raises rather than replacing a character or silently exceeding it.
    Offsets refer to the original string, including whitespace and overlap.
    """
    if limit <= 0 or overlap < 0:
        raise ValueError("invalid text splitting budget")
    parts: list[TextPart] = []
    start = 0
    while start < len(text):
        low, high = start, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if len(encoding.encode(render_text(prefix, text[start:middle]))) <= limit:
                low = middle
            else:
                high = middle - 1
        end = low
        if end == start:
            raise ValueError("chunk_size cannot fit a Unicode character with its context; increase the token limit")
        hard = end < len(text)
        if hard:
            boundaries = list(re.finditer(r"\n\s*\n|(?<=[.!?])\s+", text[start:end]))
            if boundaries:
                end = start + boundaries[-1].end()
                hard = False
        segment = text[start:end]
        assert len(encoding.encode(render_text(prefix, segment))) <= limit
        parts.append(TextPart(segment, start, end, hard))
        if end == len(text):
            break
        next_start = end
        if overlap:
            # Retain an exact original suffix, never a partial encoded character.
            low, high = start + 1, end
            while low < high:
                middle = (low + high) // 2
                if len(encoding.encode(text[middle:end])) <= overlap:
                    high = middle
                else:
                    low = middle + 1
            next_start = low
        start = max(start + 1, next_start)
    return parts
