from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from src.chunking.fixed.fixed import FixedTokenChunker


PROCESSED_DATA_DIR = Path("data/processed/allianz")

DOCUMENT_NAME = "Allianz_Motor_Fle_ (Modular Product)"

CHUNK_SIZE = 300
OVERLAP = 50


def find_document() -> Path:
    """
    Find the processed document.txt for the
    selected insurance document.
    """

    candidates = list(
        PROCESSED_DATA_DIR.rglob("document.txt")
    )

    for path in candidates:
        if path.parent.name == DOCUMENT_NAME:
            return path

    raise FileNotFoundError(
        f"Could not find document: {DOCUMENT_NAME}"
    )


def evaluate_fixed_chunking() -> None:
    """
    Evaluate fixed-token chunking on one real
    insurance document.
    """

    document_path = find_document()

    text = document_path.read_text(
        encoding="utf-8"
    )

    chunker = FixedTokenChunker(
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )

    chunks = chunker.chunk_text(
        text=text,
        document_id=DOCUMENT_NAME,
    )

    token_counts = [
        chunk.token_count
        for chunk in chunks
        if chunk.token_count is not None
    ]

    overlap_results = []

    for index in range(len(chunks) - 1):

        current_tokens = chunker.encoding.encode(
            chunks[index].text
        )

        next_tokens = chunker.encoding.encode(
            chunks[index + 1].text
        )

        overlap_matches = (
            current_tokens[-OVERLAP:]
            == next_tokens[:OVERLAP]
        )

        overlap_results.append(
            overlap_matches
        )

    evaluation = {
        "document": DOCUMENT_NAME,
        "document_path": str(document_path),
        "chunking": {
            "strategy": "fixed",
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
        },
        "statistics": {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "minimum_chunk_tokens": min(token_counts),
            "maximum_chunk_tokens": max(token_counts),
            "average_chunk_tokens": round(
                mean(token_counts),
                2,
            ),
        },
        "overlap_validation": {
            "boundaries_checked": len(
                overlap_results
            ),
            "correct_boundaries": sum(
                overlap_results
            ),
            "all_correct": all(
                overlap_results
            ),
        },
    }

    output_path = (
        document_path.parent
        / "fixed_chunk_evaluation.json"
    )

    output_path.write_text(
        json.dumps(
            evaluation,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("Fixed Token Chunking Evaluation")
    print("=" * 60)

    print(f"Document: {DOCUMENT_NAME}")
    print(f"Chunks: {len(chunks)}")
    print(f"Total tokens: {sum(token_counts)}")
    print(
        f"Minimum tokens: {min(token_counts)}"
    )
    print(
        f"Maximum tokens: {max(token_counts)}"
    )
    print(
        f"Average tokens: "
        f"{mean(token_counts):.2f}"
    )

    print()
    print("Overlap validation")
    print("-" * 60)

    print(
        f"Boundaries checked: "
        f"{len(overlap_results)}"
    )

    print(
        f"Correct boundaries: "
        f"{sum(overlap_results)}"
    )

    print(
        f"All correct: "
        f"{all(overlap_results)}"
    )

    print()
    print(
        f"Evaluation saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    evaluate_fixed_chunking()
