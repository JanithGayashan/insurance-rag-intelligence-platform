from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.indexing.configuration import EmbeddingConfiguration, json_sha256
from src.indexing.models import IndexRecord


class IndexingState:
    """Durable vector cache, run checkpoints, and committed document registry.

    The pipeline holds a file lock for the lifetime of this connection. Each
    cache batch and checkpoint is its own committed SQLite transaction.
    """

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                corpus_signature TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TEXT,
                error TEXT,
                summary TEXT
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                point_id TEXT NOT NULL,
                record_hash TEXT NOT NULL,
                PRIMARY KEY (run_id, point_id)
            );
            CREATE TABLE IF NOT EXISTS documents (
                scope TEXT NOT NULL,
                logical_id TEXT NOT NULL,
                version TEXT NOT NULL,
                point_ids TEXT NOT NULL,
                PRIMARY KEY (scope, logical_id)
            );
        """)

    def close(self) -> None:
        self.connection.close()

    def get_vector(self, cache_key: str, configuration: EmbeddingConfiguration) -> np.ndarray | None:
        row = self.connection.execute(
            "SELECT * FROM embeddings WHERE cache_key = ?", (cache_key,),
        ).fetchone()
        if row is None:
            return None
        try:
            vector = np.frombuffer(row["vector"], dtype="<f4").copy()
        except ValueError:
            vector = np.empty(0, dtype=np.float32)
        if (
            row["fingerprint"] != configuration.fingerprint
            or row["dimension"] != configuration.dimension
            or vector.shape != (configuration.dimension,)
            or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4)
        ):
            with self.connection:
                self.connection.execute("DELETE FROM embeddings WHERE cache_key = ?", (cache_key,))
            return None
        return vector

    def put_vectors(self, keys: Sequence[str], vectors: np.ndarray, configuration: EmbeddingConfiguration) -> None:
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                [(key, configuration.fingerprint, configuration.dimension,
                  vector.astype("<f4").tobytes()) for key, vector in zip(keys, vectors, strict=True)],
            )

    def begin_run(self, scope: str, signature: str, fingerprint: str) -> tuple[str, int]:
        row = self.connection.execute(
            "SELECT run_id FROM runs WHERE scope = ? AND corpus_signature = ? "
            "AND fingerprint = ? AND status != 'COMPLETED' ORDER BY started_at DESC, rowid DESC LIMIT 1",
            (scope, signature, fingerprint),
        ).fetchone()
        with self.connection:
            if row:
                run_id = row["run_id"]
                self.connection.execute(
                    "UPDATE runs SET status = 'RUNNING', error = NULL, finished_at = NULL WHERE run_id = ?", (run_id,),
                )
            else:
                run_id = str(uuid.uuid4())
                self.connection.execute(
                    "INSERT INTO runs (run_id, scope, corpus_signature, fingerprint, status) VALUES (?, ?, ?, ?, 'RUNNING')",
                    (run_id, scope, signature, fingerprint),
                )
        count = self.connection.execute("SELECT COUNT(*) FROM checkpoints WHERE run_id = ?", (run_id,)).fetchone()[0]
        return run_id, count

    def checkpoint(self, run_id: str, records: Sequence[IndexRecord]) -> None:
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?)",
                [(run_id, record.point_id, json_sha256(record.model_dump())) for record in records],
            )

    def document_registry(self, scope: str) -> dict[str, str]:
        return {row["logical_id"]: row["version"] for row in self.connection.execute(
            "SELECT logical_id, version FROM documents WHERE scope = ?", (scope,),
        )}

    def complete(self, run_id: str, scope: str, documents: dict[str, dict[str, Any]], summary: dict[str, Any]) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM documents WHERE scope = ?", (scope,))
            self.connection.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?)",
                [(scope, key, value["version"], json.dumps(value["point_ids"])) for key, value in documents.items()],
            )
            self.connection.execute(
                "UPDATE runs SET status = 'COMPLETED', finished_at = CURRENT_TIMESTAMP, summary = ? WHERE run_id = ?",
                (json.dumps(summary, ensure_ascii=False), run_id),
            )

    def fail(self, run_id: str, error: BaseException) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE runs SET status = 'FAILED', error = ?, finished_at = CURRENT_TIMESTAMP WHERE run_id = ?",
                (f"{type(error).__name__}: {error}", run_id),
            )
