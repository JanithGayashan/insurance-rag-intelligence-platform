from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from src.indexing.models import IndexRecord, RetrievalHit
from src.indexing.configuration import EmbeddingConfiguration, json_sha256


class QdrantVectorStore:
    """Qdrant collection lifecycle, indexing, validation, and search."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_model: str,
        vector_dimension: int,
        local_mode: bool = False,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")
        if vector_dimension <= 0:
            raise ValueError("vector_dimension must be greater than 0")
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_dimension = vector_dimension
        self.local_mode = local_mode
        self.embedding_fingerprint: str | None = None

    @classmethod
    def local(
        cls,
        path: Path,
        collection_name: str,
        embedding_model: str,
        vector_dimension: int,
    ) -> "QdrantVectorStore":
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return cls(
            QdrantClient(path=str(path)),
            collection_name,
            embedding_model,
            vector_dimension,
            local_mode=True,
        )

    @classmethod
    def remote(
        cls,
        url: str,
        collection_name: str,
        embedding_model: str,
        vector_dimension: int,
        api_key: str | None = None,
    ) -> "QdrantVectorStore":
        return cls(
            QdrantClient(url=url, api_key=api_key),
            collection_name,
            embedding_model,
            vector_dimension,
            local_mode=False,
        )

    def ensure_collection(
        self, recreate: bool = False,
        configuration: EmbeddingConfiguration | None = None,
        ingestion_scope: str | None = None,
    ) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dimension,
                    distance=models.Distance.COSINE,
                ),
                metadata={
                    "embedding_model": self.embedding_model,
                    **({"embedding_fingerprint": configuration.fingerprint,
                        "embedding_configuration": configuration.model_dump(),
                        "pipeline_version": 2,
                        "ingestion_scope": ingestion_scope} if configuration else {}),
                },
            )
        else:
            information = self.client.get_collection(self.collection_name)
            vector_config = information.config.params.vectors
            if not isinstance(vector_config, models.VectorParams):
                raise ValueError("named-vector Qdrant collections are not supported")
            if vector_config.size != self.vector_dimension:
                raise ValueError(
                    "existing Qdrant vector dimension "
                    f"{vector_config.size} does not match {self.vector_dimension}"
                )
            if vector_config.distance != models.Distance.COSINE:
                raise ValueError("existing Qdrant collection does not use cosine distance")
        if configuration:
            self.check_configuration(configuration)
            information = self.client.get_collection(self.collection_name)
            if ingestion_scope and information.config.metadata.get("ingestion_scope") != ingestion_scope:
                raise ValueError("collection belongs to a different corpus; use a separate collection")
        if not self.local_mode:
            for field in (
                "document_id",
                "source_sha256",
                "source_file",
                "validation_status",
                "strategy",
                "embedding_model",
            ):
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )

    def check_configuration(self, configuration: EmbeddingConfiguration) -> None:
        information = self.client.get_collection(self.collection_name)
        metadata = information.config.metadata or {}
        if metadata.get("embedding_fingerprint") != configuration.fingerprint:
            raise ValueError("embedding configuration differs from the indexed collection; build a new collection")
        self.embedding_fingerprint = configuration.fingerprint

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Read current payloads and vectors so checkpoints never hide index loss."""
        result = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name, limit=256, offset=offset,
                with_payload=True, with_vectors=True,
            )
            for point in points:
                result[str(point.id)] = {"payload": dict(point.payload or {}), "vector": point.vector}
            if offset is None:
                return result

    def delete_points(self, point_ids: Sequence[str]) -> None:
        for start in range(0, len(point_ids), 256):
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=list(point_ids[start:start + 256])),
                wait=True,
            )

    def upsert(
        self,
        records: Sequence[IndexRecord],
        vectors: np.ndarray,
        batch_size: int = 64,
    ) -> None:
        if vectors.shape != (len(records), self.vector_dimension):
            raise ValueError(
                f"vectors have shape {vectors.shape}; expected "
                f"({len(records)}, {self.vector_dimension})"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            points = [
                models.PointStruct(
                    id=record.point_id,
                    vector=vector.tolist(),
                    payload={
                        **record.to_payload(self.embedding_model),
                        "embedding_fingerprint": self.embedding_fingerprint,
                        "record_sha256": json_sha256(record.to_payload(self.embedding_model)),
                    },
                )
                for record, vector in zip(batch_records, batch_vectors)
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def validate(self, expected_records: Sequence[IndexRecord], *, allow_unexpected: bool = False) -> dict[str, Any]:
        count = self.client.count(self.collection_name, exact=True).count
        expected_ids = {record.point_id for record in expected_records}
        stored_ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            stored_ids.update(str(point.id) for point in points)
            if offset is None:
                break
        missing = expected_ids - stored_ids
        unexpected = stored_ids - expected_ids
        return {
            "status": "PASS" if not missing and (allow_unexpected or not unexpected) else "FAIL",
            "expected_points": len(expected_records),
            "stored_points": count,
            "missing_points": len(missing),
            "unexpected_points": len(unexpected),
            "duplicate_chunk_ids": len(expected_records)
            - len({record.chunk_id for record in expected_records}),
            "vector_dimension": self.vector_dimension,
            "distance": "Cosine",
            "embedding_model": self.embedding_model,
        }

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        filters: Mapping[str, str | int | bool] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalHit]:
        if query_vector.shape != (self.vector_dimension,):
            raise ValueError(
                f"query vector has shape {query_vector.shape}; expected "
                f"({self.vector_dimension},)"
            )
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        query_filter = None
        if filters:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                    for key, value in filters.items()
                ]
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )
        hits: list[RetrievalHit] = []
        for rank, point in enumerate(response.points, 1):
            payload = dict(point.payload or {})
            hits.append(
                RetrievalHit(
                    rank=rank,
                    score=float(point.score),
                    point_id=str(point.id),
                    chunk_id=str(payload.get("chunk_id", "")),
                    document_id=str(payload.get("document_id", "")),
                    text=str(payload.get("text", "")),
                    section_path=list(payload.get("section_path", [])),
                    page_numbers=list(payload.get("page_numbers", [])),
                    source_refs=list(payload.get("source_refs", [])),
                    source_file=payload.get("source_file"),
                    validation_status=payload.get("validation_status"),
                    token_count=payload.get("token_count"),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
        return hits

    def close(self) -> None:
        self.client.close()
