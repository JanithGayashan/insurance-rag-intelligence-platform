"""Invalidate generated chunk artifacts while retaining recoverable snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def archive_artifact(path: Path) -> None:
    if path.is_file():
        content = path.read_bytes()
        archive = path.parent / ".chunk_history" / hashlib.sha256(content).hexdigest() / path.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(content)


def invalidate_chunks(directory: Path, reason: str, *, active: bool = True) -> None:
    directory = Path(directory)
    for path in [*([directory / "structure_chunks.json"] if active else []), *sorted((directory / "structure_aware_comparison").glob("*"))]:
        if path.is_file():
            archive_artifact(path)
            path.unlink()
    if active:
        (directory / "chunking_invalidated.json").write_text(json.dumps({"status": "INVALIDATED", "reason": reason}, indent=2), encoding="utf-8")


def clear_invalidation(directory: Path) -> None:
    (Path(directory) / "chunking_invalidated.json").unlink(missing_ok=True)
