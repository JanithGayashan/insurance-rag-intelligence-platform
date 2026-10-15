from __future__ import annotations

from collections.abc import Sequence
import time

from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import QueryBundle

from src.retrieval.bm25 import LlamaIndexBM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.llamaindex import DenseLlamaIndexRetriever, RankedResultsRetriever, as_hits, snapshot_nodes


class LlamaIndexRetrievalSuite:
    """Compare dense, BM25 and equal-contribution reciprocal rank fusion.

    The BM25 snapshot is built once per process and per filter selection. Rebuild
    this suite after ingestion changes. No document vectors or LLM calls are made.
    """

    DEFAULT_METHODS = ("dense", "bm25", "hybrid")
    METHODS = (*DEFAULT_METHODS, "hybrid_reranked")

    def __init__(self, dense: DenseRetriever, *, snapshot=None, candidate_k: int = 20,
                 reranker=None, rerank_candidates: int = 20):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.dense = dense
        self.snapshot = snapshot if snapshot is not None else dense.vector_store.snapshot()
        self.nodes = snapshot_nodes(self.snapshot)
        self.bm25 = LlamaIndexBM25Retriever(self.nodes)
        self.candidate_k = candidate_k
        if rerank_candidates <= 0:
            raise ValueError("rerank_candidates must be positive")
        self.reranker, self.rerank_candidates = reranker, rerank_candidates

    def retrieve_all(self, question: str, top_k: int = 5, filters=None,
                     methods: Sequence[str] = DEFAULT_METHODS):
        return self.retrieve_all_with_diagnostics(question, top_k, filters, methods)[0]

    def retrieve_all_with_diagnostics(self, question: str, top_k: int = 5, filters=None,
                                      methods: Sequence[str] = DEFAULT_METHODS):
        if not question.strip() or top_k <= 0:
            raise ValueError("question must be nonempty and top_k must be positive")
        if not methods or any(method not in self.METHODS for method in methods):
            raise ValueError("methods must be dense, bm25, hybrid or hybrid_reranked")
        rerank = "hybrid_reranked" in methods
        if rerank and (self.reranker is None or not top_k <= self.rerank_candidates <= self.candidate_k):
            raise ValueError("reranking requires a configured reranker and top_k <= rerank_candidates <= candidate_k")
        fusion_needed = "hybrid" in methods or rerank
        depth = max(top_k, self.candidate_k)
        bundle = QueryBundle(query_str=question)
        started = time.perf_counter()
        dense = DenseLlamaIndexRetriever(self.dense, self.nodes, depth, filters).retrieve(bundle) if (
            "dense" in methods or fusion_needed) else []
        dense_seconds = time.perf_counter() - started
        started = time.perf_counter()
        sparse = self.bm25.retrieve_nodes(question, depth, filters) if (
            "bm25" in methods or fusion_needed) else []
        sparse_seconds = time.perf_counter() - started
        times = {"dense": dense_seconds, "bm25": sparse_seconds}
        candidates = {"dense": dense, "bm25": sparse}
        diagnostics = {"method_seconds": times}
        if fusion_needed:
            # MockLLM fills the framework's LLM slot. num_queries=1 prevents its use.
            fusion = QueryFusionRetriever(
                [RankedResultsRetriever(dense), RankedResultsRetriever(sparse)],
                llm=MockLLM(), num_queries=1, mode="reciprocal_rerank",
                similarity_top_k=self.rerank_candidates if rerank else top_k, use_async=False, verbose=False,
            )
            started = time.perf_counter()
            candidates["hybrid"] = fusion.retrieve(bundle)
            times["hybrid"] = dense_seconds + sparse_seconds + time.perf_counter() - started
        if rerank:
            selected = candidates["hybrid"][:self.rerank_candidates]
            started = time.perf_counter()
            candidates["hybrid_reranked"] = self.reranker.rerank_nodes(question, selected, top_k=max(1, len(selected)))
            seconds = time.perf_counter() - started
            times["hybrid_reranked"] = times["hybrid"] + seconds
            diagnostics.update(rerank_seconds=seconds,
                               rerank_candidate_ids=[item.node.node_id for item in selected],
                               rerank_window_count=sum(item.node.metadata["payload"]["metadata"]["reranking"]["window_count"]
                                                       for item in candidates["hybrid_reranked"]))
        return {method: as_hits(candidates[method][:top_k]) for method in dict.fromkeys(methods)}, diagnostics

    def retrieve(self, question: str, top_k: int = 5, filters=None, method: str = "hybrid"):
        return self.retrieve_all(question, top_k, filters, (method,))[method]
