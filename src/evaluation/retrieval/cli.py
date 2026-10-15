from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.retrieval.runner import evaluate_dense_retrieval
from src.evaluation.retrieval.dataset import load_dataset, select_questions
from src.indexing.embeddings import DenseEmbeddingService
from src.indexing.pipeline import DEFAULT_COLLECTION, LOCAL_QDRANT_PATH
from src.indexing.qdrant_store import QdrantVectorStore
from src.retrieval.dense import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense, BM25 and hybrid insurance retrieval with source evidence and judgment-aware pools.")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/datasets/allianz_questions_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/allianz"))
    parser.add_argument("--qdrant-path", type=Path, default=LOCAL_QDRANT_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--include-source-checked", action="store_true", help="Allow a clearly labeled provisional baseline before human approval.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/allianz"))
    parser.add_argument("--mode", choices=["unfiltered", "policy_scoped", "both"], default="both")
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid", "all"], default="all")
    parser.add_argument("--pool-depth", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--judgments", type=Path, help="Snapshot-bound pooled judgments to apply.")
    parser.add_argument("--outside-pool-sample", type=int, default=2)
    parser.add_argument("--require-judged-top5", action="store_true")
    parser.add_argument("--legacy-dense", action="store_true", help="Reproduce the previous positive-only baseline; requires --retriever dense.")
    args = parser.parse_args()
    if args.legacy_dense and (args.retriever != "dense" or args.judgments or args.require_judged_top5):
        parser.error("--legacy-dense requires --retriever dense without pooled judgment options")
    # Reject unreviewed-only datasets before loading the embedding model.
    select_questions(load_dataset(args.dataset), args.include_source_checked)
    load_dotenv()
    embedder = DenseEmbeddingService()
    url = os.getenv("QDRANT_URL", "").strip()
    store = QdrantVectorStore.remote(url, args.collection, embedder.model_name, embedder.dimension, api_key=os.getenv("QDRANT_API_KEY") or None) if url else QdrantVectorStore.local(args.qdrant_path, args.collection, embedder.model_name, embedder.dimension)
    try:
        if not args.legacy_dense:
            from src.evaluation.retrieval.comparison import evaluate_retrieval_comparison
            summary = evaluate_retrieval_comparison(
                args.dataset, DenseRetriever(embedder, store), args.output,
                methods=("dense", "bm25", "hybrid") if args.retriever == "all" else (args.retriever,),
                pool_depth=args.pool_depth, candidate_k=args.candidate_k, k_values=args.k,
                include_source_checked=args.include_source_checked,
                modes=("unfiltered", "policy_scoped") if args.mode == "both" else (args.mode,),
                raw_root=args.raw_root, judgments_path=args.judgments,
                outside_pool_sample=args.outside_pool_sample, require_judged_top5=args.require_judged_top5,
            )
        else:
            summary = evaluate_dense_retrieval(
                args.dataset, DenseRetriever(embedder, store), args.output, k_values=args.k,
                include_source_checked=args.include_source_checked,
                modes=("unfiltered", "policy_scoped") if args.mode == "both" else (args.mode,),
                raw_root=args.raw_root,
            )
        print(json.dumps(summary, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
