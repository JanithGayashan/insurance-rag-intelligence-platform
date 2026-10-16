from __future__ import annotations

from pathlib import Path

from src.configuration.dataset import get_dataset_configuration

from src.chunking.fixed.fixed import FixedTokenChunker


PROCESSED_DATA_DIR = get_dataset_configuration().processed_root


CHUNK_SIZE = 300
OVERLAP = 50


def find_documents() -> list[Path]:
    """Discover every parsed document in the active dataset."""
    return sorted(PROCESSED_DATA_DIR.rglob("document.txt"))


def inspect_chunks(document_path: Path | None = None) -> None:
    """Print the complete text of every fixed chunk."""

    if document_path is None:
        for candidate in find_documents():
            inspect_chunks(candidate)
        return
    document_name = document_path.parent.name

    text = document_path.read_text(
        encoding="utf-8"
    )

    chunker = FixedTokenChunker(
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )

    chunks = chunker.chunk_text(
        text=text,
        document_id=document_name,
    )

    output = []

    output.append("=" * 80)
    output.append("FIXED CHUNK INSPECTION")
    output.append("=" * 80)
    output.append(f"Document: {document_name}")
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
