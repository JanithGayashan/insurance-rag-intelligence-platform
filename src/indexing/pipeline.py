from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from filelock import FileLock

from src.chunking.semantic.embeddings import DEFAULT_EMBEDDING_MODEL
from src.indexing.configuration import json_sha256
from src.indexing.embeddings import DenseEmbedder, DenseEmbeddingService
from src.indexing.loader import load_structure_records
from src.indexing.models import IndexRecord
from src.indexing.preparation import prepare_embedding_records
from src.indexing.qdrant_store import QdrantVectorStore
from src.indexing.state import IndexingState
from src.retrieval.dense import DenseRetriever


PROCESSED_DATA_DIR = Path("data/processed/allianz")
LOCAL_QDRANT_PATH = Path("data/indexes/qdrant")
DEFAULT_COLLECTION = "insurance_structure_bge_small_en_v1_5_v2"
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
    recreate_collection: bool = False,
    example_questions: Sequence[str] = DEFAULT_EXAMPLE_QUESTIONS,
    top_k: int = 5,
    state_path: Path | None = None,
    batch_size: int = 32,
    allow_empty_corpus: bool = False,
    revision: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Incrementally sync an authoritative chunk corpus with a dedicated index.

    Cached embeddings and confirmed batches survive interruption. Outdated
    vectors are removed only after replacements exist and the source snapshot
    is rechecked. During updates, old/new points can temporarily coexist; this
    single-writer pipeline does not provide atomic collection-wide cutover.
    """
    if batch_size <= 0 or top_k <= 0:
        raise ValueError("batch_size and top_k must be greater than 0")
    load_dotenv()
    started = time.perf_counter()
    root = Path(processed_directory).resolve()
    source_records, source_counts = load_structure_records(root, allow_empty=allow_empty_corpus)
    source_signature = _signature(source_records)
    service = embedder or DenseEmbeddingService(embedding_model, revision=revision)
    config = service.configuration
    if not config.normalize or config.precision != "float32":
        raise ValueError("pipeline v2 requires normalized float32 embeddings")
    if config.dimension != service.dimension or config.model_name != service.model_name:
        raise ValueError("embedder identity does not match its configuration")
    records, tokens = prepare_embedding_records(source_records, service)
    if len({record.point_id for record in records}) != len(records):
        raise ValueError("prepared chunks contain duplicate point IDs")
    documents = _documents(records)
    state_path = Path(state_path) if state_path else root / ".indexing" / "state.sqlite3"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    target = os.getenv("QDRANT_URL", "").strip() or str(Path(local_qdrant_path).resolve())
    scope = json_sha256({"corpus": str(root), "target": target,
                         "collection": vector_store.collection_name if vector_store else collection_name})
    with FileLock(str(state_path) + ".lock", timeout=0):
        state = IndexingState(state_path)
        store = None
        run_id = None
        try:
            store = vector_store or _create_store(local_qdrant_path, collection_name, service)
            if store.embedding_model != service.model_name or store.vector_dimension != service.dimension:
                raise ValueError("vector store and embedder identities must match")
            store.ensure_collection(recreate=recreate_collection, configuration=config, ingestion_scope=scope)
            run_id, resumed_points = state.begin_run(scope, source_signature, config.fingerprint)
            previous = state.document_registry(scope)
            live = store.snapshot()
            pending, norms = [], []
            for record in records:
                existing = live.get(record.point_id)
                if existing and _point_matches(record, existing, service):
                    norms.append(float(np.linalg.norm(existing["vector"])))
                else:
                    pending.append(record)
            stats = {
                "points_unchanged": len(records) - len(pending),
                "points_upserted": 0, "points_deleted": 0,
                "unique_texts_embedded": 0, "cache_hits": 0,
                "duplicate_inputs_reused": 0, "completed_batches": 0,
                "resumed_checkpoint_points": resumed_points,
                "documents_added": len(documents.keys() - previous.keys()),
                "documents_removed": len(previous.keys() - documents.keys()),
                "documents_updated": sum(previous[key] != value["version"] for key, value in documents.items() if key in previous),
                "documents_unchanged": sum(previous.get(key) == value["version"] for key, value in documents.items()),
            }
            embedding_seconds = indexing_seconds = 0.0
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                keys = [_cache_key(record, config.fingerprint) for record in batch]
                vectors_by_key, missing = {}, {}
                for key, record in zip(keys, batch, strict=True):
                    if key in vectors_by_key or key in missing:
                        stats["duplicate_inputs_reused"] += 1
                        continue
                    cached = state.get_vector(key, config)
                    if cached is None:
                        missing[key] = record.text
                    else:
                        vectors_by_key[key] = cached
                        stats["cache_hits"] += 1
                if missing:
                    began = time.perf_counter()
                    computed = service.embed_documents(list(missing.values()))
                    embedding_seconds += time.perf_counter() - began
                    _validate_vectors(computed, len(missing), service.dimension)
                    state.put_vectors(list(missing), computed, config)
                    vectors_by_key.update(zip(missing, computed, strict=True))
                    stats["unique_texts_embedded"] += len(missing)
                vectors = np.stack([vectors_by_key[key] for key in keys]).astype(np.float32)
                _validate_vectors(vectors, len(batch), service.dimension)
                began = time.perf_counter()
                store.upsert(batch, vectors)
                indexing_seconds += time.perf_counter() - began
                state.checkpoint(run_id, batch)
                norms.extend(float(value) for value in np.linalg.norm(vectors, axis=1))
                stats["points_upserted"] += len(batch)
                stats["completed_batches"] += 1
                if progress_callback:
                    progress_callback({"stage": "indexing", **stats, "pending_points": len(pending)})
            if store.validate(records, allow_unexpected=True)["status"] != "PASS":
                raise RuntimeError("desired points are missing; outdated-vector cleanup was cancelled")
            current_records, _ = load_structure_records(root, allow_empty=allow_empty_corpus)
            if _signature(current_records) != source_signature:
                raise RuntimeError("source corpus changed during indexing; cleanup was cancelled; rerun the pipeline")
            obsolete = sorted(live.keys() - {record.point_id for record in records})
            store.delete_points(obsolete)
            stats["points_deleted"] = len(obsolete)
            validation = store.validate(records)
            if validation["status"] != "PASS":
                raise RuntimeError(f"Qdrant index validation failed: {validation}")
            began = time.perf_counter()
            retriever = DenseRetriever(service, store)
            examples = [{"question": question, "top_k": top_k,
                         "results": [hit.model_dump(mode="json") for hit in retriever.retrieve(question, top_k=top_k)]}
                        for question in example_questions]
            summary = {
                "status": "COMPLETED", "pipeline_version": 2, "run_id": run_id,
                "source_chunking_strategy": "structure-aware", "processed_directory": str(root),
                "state_path": str(state_path.resolve()), "collection_name": store.collection_name,
                "embedding_model": service.model_name, "embedding_configuration": config.model_dump(),
                "embedding_fingerprint": config.fingerprint, "vector_dimension": service.dimension,
                "distance": "Cosine", "documents_indexed": len(documents),
                "source_chunks": len(source_records), "chunks_indexed": len(records),
                "document_chunk_counts": source_counts, "deterministic_point_ids": True,
                "vectors_normalized": True,
                "vector_norm": {"minimum": round(min(norms), 6) if norms else None,
                                "maximum": round(max(norms), 6) if norms else None,
                                "average": round(sum(norms) / len(norms), 6) if norms else None},
                "incremental_updates": stats, "token_validation": tokens,
                "timing_seconds": {"embedding": round(embedding_seconds, 3), "indexing": round(indexing_seconds, 3),
                                   "example_retrieval": round(time.perf_counter() - began, 3),
                                   "total": round(time.perf_counter() - started, 3)},
                "validation": validation, "example_queries": examples,
                "evaluation_status": {
                    "automatic_index_validation": "COMPLETED", "example_query_smoke_test": "COMPLETED",
                    "reviewed_insurance_question_evaluation": "NOT_RUN",
                    "reason": "Recall@K, MRR, and nDCG require reviewed relevance labels; index validation does not establish retrieval accuracy.",
                },
            }
            (root / "dense_indexing_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            (root / "dense_indexing_summary.md").write_text(_render_report(summary), encoding="utf-8")
            state.complete(run_id, scope, documents, summary)
            return summary
        except BaseException as error:
            if run_id:
                state.fail(run_id, error)
            raise
        finally:
            state.close()
            if vector_store is None and store is not None:
                store.close()


def _signature(records: Sequence[IndexRecord]) -> str:
    return json_sha256([record.model_dump() for record in records])


def _documents(records: Sequence[IndexRecord]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[IndexRecord]] = {}
    for record in records:
        key = (record.source_file or record.document_id).replace("\\", "/")
        grouped.setdefault(key, []).append(record)
    return {key: {"version": _signature(items), "point_ids": [item.point_id for item in items]}
            for key, items in grouped.items()}


def _cache_key(record: IndexRecord, fingerprint: str) -> str:
    return json_sha256({"text_sha256": record.content_sha256, "embedding_fingerprint": fingerprint, "task": "document"})


def _point_matches(record: IndexRecord, existing: dict[str, Any], embedder: DenseEmbedder) -> bool:
    payload = existing["payload"]
    if payload.get("embedding_fingerprint") != embedder.configuration.fingerprint:
        return False
    if payload.get("record_sha256") != json_sha256(record.to_payload(embedder.model_name)):
        return False
    try:
        vector = np.asarray(existing["vector"], dtype=np.float32)
        _validate_vectors(vector.reshape(1, -1), 1, embedder.dimension)
    except (ValueError, TypeError):
        return False
    return True


def _create_store(local_path: Path, collection_name: str, embedder: DenseEmbedder) -> QdrantVectorStore:
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    if qdrant_url:
        return QdrantVectorStore.remote(qdrant_url, collection_name, embedder.model_name, embedder.dimension,
                                        api_key=os.getenv("QDRANT_API_KEY") or None)
    return QdrantVectorStore.local(local_path, collection_name, embedder.model_name, embedder.dimension)


def _validate_vectors(vectors: np.ndarray, count: int, dimension: int) -> None:
    if vectors.shape != (count, dimension):
        raise ValueError(f"embedding output has shape {vectors.shape}; expected ({count}, {dimension})")
    if not np.isfinite(vectors).all():
        raise ValueError("embedding output contains NaN or infinite values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
        raise ValueError("embedding output must contain normalized, nonzero vectors")


def _render_report(summary: dict[str, Any]) -> str:
    stats, validation, tokens = summary["incremental_updates"], summary["validation"], summary["token_validation"]
    lines = ["# Embedding and indexing pipeline v2", "",
             f"- Status: **{summary['status']}**", f"- Documents: **{summary['documents_indexed']}**",
             f"- Source chunks: **{summary['source_chunks']}**", f"- Indexed chunks: **{summary['chunks_indexed']}**",
             f"- Model: `{summary['embedding_model']}`", f"- Revision: `{summary['embedding_configuration']['model_revision']}`",
             f"- Dimensions: **{summary['vector_dimension']}**", f"- Collection: `{summary['collection_name']}`",
             "", "## Incremental processing", ""]
    lines.extend(f"- {key.replace('_', ' ')}: **{value}**" for key, value in stats.items())
    lines.extend(["", "## Token validation", "",
                  f"- Model limit including special tokens: **{tokens['model_token_limit']}**",
                  f"- Oversized source chunks: **{tokens['oversized_source_chunks']}**",
                  f"- Marked fallback parts: **{tokens['fallback_parts']}**",
                  f"- Maximum prepared input tokens: **{tokens['maximum_prepared_tokens']}**",
                  "- Silently truncated inputs: **0**", "", "## Index validation", "",
                  f"- Result: **{validation['status']}**", f"- Missing points: {validation['missing_points']}",
                  f"- Unexpected points: {validation['unexpected_points']}", "", "## Example dense retrieval", ""])
    for example in summary["example_queries"]:
        lines.extend([f"### {example['question']}", ""])
        for hit in example["results"]:
            section = " > ".join(hit["section_path"]) or "Document introduction"
            lines.append(f"{hit['rank']}. score={hit['score']:.4f}; source=`{hit['source_file']}`; section=`{section}`; pages={hit['page_numbers']}")
        lines.append("")
    lines.extend(["## Evaluation boundary", "", summary["evaluation_status"]["reason"], ""])
    return "\n".join(lines)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally embed and synchronize structure-aware chunks.")
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--qdrant-path", type=Path, default=LOCAL_QDRANT_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--rebuild", action="store_true", help="Explicitly recreate the selected collection, reusing the embedding cache.")
    parser.add_argument("--allow-empty-corpus", action="store_true", help="Explicitly allow removing all points when no chunk artifacts remain.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    result = run_embedding_and_indexing_pipeline(
        processed_directory=arguments.processed_directory, local_qdrant_path=arguments.qdrant_path,
        collection_name=arguments.collection, state_path=arguments.state_path, batch_size=arguments.batch_size,
        embedding_model=arguments.embedding_model, revision=arguments.revision,
        recreate_collection=arguments.rebuild, allow_empty_corpus=arguments.allow_empty_corpus,
    )
    print(f"Pipeline v2: {result['documents_indexed']} documents, {result['chunks_indexed']} chunks, validation {result['validation']['status']}")
