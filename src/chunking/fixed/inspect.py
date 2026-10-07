from __future__ import annotations

from pathlib import Path

from src.chunking.fixed.fixed import FixedTokenChunker


PROCESSED_DATA_DIR = Path("data/processed")

DOCUMENT_NAME = "Allianz_Motor_Fle_ (Modular Product)"

CHUNK_SIZE = 300
OVERLAP = 50


def find_document() -> Path:
    """Find the selected document.txt."""

    candidates = list(
        PROCESSED_DATA_DIR.rglob("document.txt")
    )

    for path in candidates:
        if path.parent.name == DOCUMENT_NAME:
            return path

    raise FileNotFoundError(
        f"Could not find document: {DOCUMENT_NAME}"
    )


def inspect_chunks() -> None:
    """Print the complete text of every fixed chunk."""

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

    output = []

    output.append("=" * 80)
    output.append("FIXED CHUNK INSPECTION")
    output.append("=" * 80)
    output.append(f"Document: {DOCUMENT_NAME}")
    output.append(f"Total chunks: {len(chunks)}")
    output.append(f"Chunk size: {CHUNK_SIZE}")
    output.append(f"Overlap: {OVERLAP}")
    output.append("")

    for chunk in chunks:

        output.append("=" * 80)
        output.append(
            f"CHUNK {chunk.chunk_index}"
        )
        output.append(
            f"Token count: {chunk.token_count}"
        )
        output.append("-" * 80)

        output.append(chunk.text)

        output.append("")
        output.append("")

    output_text = "\n".join(output)

    output_path = (
        document_path.parent
        / "fixed_chunk_full_inspection.txt"
    )

    output_path.write_text(
        output_text,
        encoding="utf-8",
    )

    print(output_text)

    print("=" * 80)
    print(
        f"Inspection saved to:\n{output_path}"
    )


if __name__ == "__main__":
    inspect_chunks()
