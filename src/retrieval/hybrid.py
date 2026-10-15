from __future__ import annotations

from collections.abc import Sequence

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

    METHODS = ("dense", "bm25", "hybrid")

    def __init__(self, dense: DenseRetriever, *, snapshot=None, candidate_k: int = 20):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.dense = dense
        self.snapshot = snapshot if snapshot is not None else dense.vector_store.snapshot()
        self.nodes = snapshot_nodes(self.snapshot)
        self.bm25 = LlamaIndexBM25Retriever(self.nodes)
        self.candidate_k = candidate_k

    def retrieve_all(self, question: str, top_k: int = 5, filters=None,
                     methods: Sequence[str] = METHODS):
        if not question.strip() or top_k <= 0:
            raise ValueError("question must be nonempty and top_k must be positive")
        if not methods or any(method not in self.METHODS for method in methods):
            raise ValueError("methods must be dense, bm25 or hybrid")
        depth = max(top_k, self.candidate_k)
        bundle = QueryBundle(query_str=question)
        dense = DenseLlamaIndexRetriever(self.dense, self.nodes, depth, filters).retrieve(bundle) if (
            "dense" in methods or "hybrid" in methods) else []
        sparse = self.bm25.retrieve_nodes(question, depth, filters) if (
            "bm25" in methods or "hybrid" in methods) else []
        candidates = {"dense": dense, "bm25": sparse}
        if "hybrid" in methods:
            # MockLLM fills the framework's LLM slot. num_queries=1 prevents its use.
            fusion = QueryFusionRetriever(
                [RankedResultsRetriever(dense), RankedResultsRetriever(sparse)],
                llm=MockLLM(), num_queries=1, mode="reciprocal_rerank",
                similarity_top_k=top_k, use_async=False, verbose=False,
            )
            candidates["hybrid"] = fusion.retrieve(bundle)
        return {method: as_hits(candidates[method][:top_k]) for method in dict.fromkeys(methods)}

    def retrieve(self, question: str, top_k: int = 5, filters=None, method: str = "hybrid"):
        return self.retrieve_all(question, top_k, filters, (method,))[method]
