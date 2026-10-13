from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from src.chunking.semantic.embeddings import DEFAULT_EMBEDDING_MODEL
from src.indexing.embeddings import DenseEmbedder, DenseEmbeddingService
from src.indexing.loader import load_structure_records
from src.indexing.qdrant_store import QdrantVectorStore
from src.retrieval.dense import DenseRetriever


PROCESSED_DATA_DIR = Path("data/processed/allianz")
LOCAL_QDRANT_PATH = Path("data/indexes/qdrant")
DEFAULT_COLLECTION = "insurance_structure_bge_small_en_v1_5"
DEFAULT_EXAMPLE_QUESTIONS = (
    "What accidental damage is covered by motor insurance?",
    "Which documents are required when submitting a claim?",
    "What losses caused by war are excluded?",
    "What hospital and medical expenses are covered?",
)


def run_embedding_and_indexing_pipeline(
    processed_directory: Path = PROCESSED_DATA_DIR,
    local_qdrant_path: Path = LOCAL_QDRANT_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedder: DenseEmbedder | None = None,
    vector_store: QdrantVectorStore | None = None,
    recreate_collection: bool = True,
    example_questions: Sequence[str] = DEFAULT_EXAMPLE_QUESTIONS,
    top_k: int = 5,
) -> dict[str, Any]:
    """Embed structure-aware chunks, index them, validate, and query them.

    The generated collection is replaced by default so removed or renamed
    source chunks cannot survive as stale Qdrant points. Point IDs remain
    deterministic for stable provenance and idempotent upserts.
    """

    load_dotenv()
    started = time.perf_counter()
    records, document_counts = load_structure_records(processed_directory)
    resolved_embedder = embedder or DenseEmbeddingService(embedding_model)
    embedding_started = time.perf_counter()
    vectors = resolved_embedder.embed_documents([record.text for record in records])
    embedding_seconds = time.perf_counter() - embedding_started
    _validate_vectors(vectors, len(records), resolved_embedder.dimension)

    owns_store = vector_store is None
    store = vector_store or _create_store(
        local_qdrant_path,
        collection_name,
        resolved_embedder,
    )
    indexing_started = time.perf_counter()
    store.ensure_collection(recreate=recreate_collection)
    store.upsert(records, vectors)
    indexing_seconds = time.perf_counter() - indexing_started
    validation = store.validate(records)
    if validation["status"] != "PASS":
        if owns_store:
            store.close()
        raise RuntimeError(f"Qdrant index validation failed: {validation}")

    retriever = DenseRetriever(resolved_embedder, store)
    retrieval_started = time.perf_counter()
    examples = []
    for question in example_questions:
        hits = retriever.retrieve(question, top_k=top_k)
        examples.append(
            {
                "question": question,
                "top_k": top_k,
                "results": [hit.model_dump(mode="json") for hit in hits],
            }
        )
    retrieval_seconds = time.perf_counter() - retrieval_started
    vector_norms = np.linalg.norm(vectors, axis=1)
    summary = {
        "status": "COMPLETED",
        "source_chunking_strategy": "structure-aware",
        "processed_directory": str(processed_directory),
        "collection_name": store.collection_name,
        "embedding_model": resolved_embedder.model_name,
        "vector_dimension": resolved_embedder.dimension,
        "distance": "Cosine",
        "documents_indexed": len(document_counts),
        "chunks_indexed": len(records),
        "document_chunk_counts": document_counts,
        "deterministic_point_ids": True,
        "vectors_normalized": bool(np.allclose(vector_norms, 1.0, atol=1e-4)),
        "vector_norm": {
            "minimum": round(float(vector_norms.min()), 6),
            "maximum": round(float(vector_norms.max()), 6),
            "average": round(float(vector_norms.mean()), 6),
        },
        "timing_seconds": {
            "embedding": round(embedding_seconds, 3),
            "indexing": round(indexing_seconds, 3),
            "example_retrieval": round(retrieval_seconds, 3),
            "total": round(time.perf_counter() - started, 3),
        },
        "validation": validation,
        "example_queries": examples,
        "evaluation_status": {
            "automatic_index_validation": "COMPLETED",
            "example_query_smoke_test": "COMPLETED",
            "reviewed_insurance_question_evaluation": "NOT_RUN",
            "reason": (
                "Recall@K, MRR, and nDCG require reviewed relevance labels and "
                "will be implemented in the retrieval-evaluation phase."
            ),
        },
    }
    output_directory = Path(processed_directory)
    json_path = output_directory / "dense_indexing_summary.json"
    markdown_path = output_directory / "dense_indexing_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown_path.write_text(_render_report(summary), encoding="utf-8")
    if owns_store:
        store.close()
    return summary


def _create_store(
    local_path: Path,
    collection_name: str,
    embedder: DenseEmbedder,
) -> QdrantVectorStore:
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    if qdrant_url:
        return QdrantVectorStore.remote(
            qdrant_url,
            collection_name,
            embedder.model_name,
            embedder.dimension,
            api_key=os.getenv("QDRANT_API_KEY") or None,
        )
    return QdrantVectorStore.local(
        local_path,
        collection_name,
        embedder.model_name,
        embedder.dimension,
    )


def _validate_vectors(vectors: np.ndarray, count: int, dimension: int) -> None:
    if vectors.shape != (count, dimension):
        raise ValueError(
            f"embedding output has shape {vectors.shape}; expected ({count}, {dimension})"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("embedding output contains NaN or infinite values")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("embedding output contains zero-length vectors")


def _render_report(summary: dict[str, Any]) -> str:
    validation = summary["validation"]
    lines = [
        "# Dense embedding and Qdrant indexing summary",
        "",
        f"- Status: **{summary['status']}**",
        f"- Source chunks: **{summary['source_chunking_strategy']}**",
        f"- Documents indexed: **{summary['documents_indexed']}**",
        f"- Chunks indexed: **{summary['chunks_indexed']}**",
        f"- Embedding model: `{summary['embedding_model']}`",
        f"- Vector dimension: **{summary['vector_dimension']}**",
        f"- Qdrant collection: `{summary['collection_name']}`",
        f"- Distance: **{summary['distance']}**",
        f"- Vectors normalized: **{summary['vectors_normalized']}**",
        "",
        "## Index validation",
        "",
        f"- Result: **{validation['status']}**",
        f"- Expected points: {validation['expected_points']}",
        f"- Stored points: {validation['stored_points']}",
        f"- Missing points: {validation['missing_points']}",
        f"- Unexpected points: {validation['unexpected_points']}",
        f"- Duplicate chunk IDs: {validation['duplicate_chunk_ids']}",
        "",
        "## Example dense retrieval",
        "",
    ]
    for example in summary["example_queries"]:
        lines.extend([f"### {example['question']}", ""])
        for hit in example["results"]:
            section = " > ".join(hit["section_path"]) or "Document introduction"
            lines.append(
                f"{hit['rank']}. score={hit['score']:.4f}; "
                f"source=`{hit['source_file']}`; section=`{section}`; "
                f"pages={hit['page_numbers']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Current evaluation boundary",
            "",
            summary["evaluation_status"]["reason"],
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_embedding_and_indexing_pipeline()
    print(
        "Dense indexing completed: "
        f"{result['documents_indexed']} documents, "
        f"{result['chunks_indexed']} chunks, "
        f"validation {result['validation']['status']}"
    )
