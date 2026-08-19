from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup

from src.ingestion.models import ParseResult


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(
        str(value)
        .replace("\n", " ")
        .split()
    )


def normalize_rows(
    rows: Iterable[Iterable[Any]] | None
) -> list[list[str]]:

    if rows is None:
        return []

    normalized = []

    for row in rows:
        if row is None:
            continue

        normalized.append(
            [normalize_cell(cell) for cell in row]
        )

    return normalized


def rows_to_markdown(
    rows: list[list[str]]
) -> str | None:

    if not rows:
        return None

    width = max(len(row) for row in rows)

    padded_rows = [
        row + [""] * (width - len(row))
        for row in rows
    ]

    def escape(value: str) -> str:
        return value.replace("|", r"\|")

    header = (
        "| "
        + " | ".join(escape(value) for value in padded_rows[0])
        + " |"
    )

    separator = (
        "| "
        + " | ".join("---" for _ in range(width))
        + " |"
    )

    body = [
        "| "
        + " | ".join(escape(value) for value in row)
        + " |"
        for row in padded_rows[1:]
    ]

    return "\n".join(
        [header, separator, *body]
    )


def html_table_to_rows(
    html: str | None
) -> list[list[str]]:

    if not html:
        return []

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    rows = []

    for tr in soup.find_all("tr"):

        cells = tr.find_all(
            ["th", "td"]
        )

        if cells:
            rows.append(
                [
                    normalize_cell(
                        cell.get_text(
                            " ",
                            strip=True
                        )
                    )
                    for cell in cells
                ]
            )

    return rows


def points_to_bbox(
    points: Any
) -> list[float] | None:

    if not points:
        return None

    try:

        xs = [
            float(point[0])
            for point in points
        ]

        ys = [
            float(point[1])
            for point in points
        ]

        return [
            min(xs),
            min(ys),
            max(xs),
            max(ys)
        ]

    except (
        TypeError,
        ValueError,
        IndexError
    ):
        return None


def xywh_to_bbox(
    bbox: Any
) -> list[float] | None:

    if bbox is None:
        return None

    if hasattr(bbox, "model_dump"):
        bbox = bbox.model_dump()

    if isinstance(bbox, dict):

        try:

            x = float(bbox["x"])
            y = float(bbox["y"])
            w = float(bbox["w"])
            h = float(bbox["h"])

            return [
                x,
                y,
                x + w,
                y + h
            ]

        except (
            KeyError,
            TypeError,
            ValueError
        ):
            return None

    return None


def safe_model_dump(value: Any) -> Any:

    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump(
            mode="json"
        )

    if hasattr(value, "to_dict"):
        return value.to_dict()

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            list,
            dict
        )
    ):
        return value

    return str(value)


def save_parse_result(
    result: ParseResult,
    output_path: str | Path
) -> None:

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result.to_dict(),
            file,
            ensure_ascii=False,
            indent=2
        )