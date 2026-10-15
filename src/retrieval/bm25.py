from __future__ import annotations

from collections.abc import Mapping

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from src.indexing.models import RetrievalHit
from src.retrieval.llamaindex import as_hits


class LlamaIndexBM25Retriever:
    """English BM25 over chunk text, with policy filtering before top-K selection."""

    def __init__(self, nodes: Mapping[str, TextNode]):
        self.nodes = nodes
        self._indexes: dict[tuple, BM25Retriever | None] = {}

    def retrieve_nodes(self, question: str, top_k: int = 5, filters=None) -> list[NodeWithScore]:
        if not question.strip() or top_k <= 0:
            raise ValueError("question must be nonempty and top_k must be positive")
        key = tuple(sorted((filters or {}).items()))
        if key not in self._indexes:
            selected = [node for node in self.nodes.values()
                        if all(node.metadata["payload"].get(field) == value for field, value in key)]
            self._indexes[key] = BM25Retriever.from_defaults(
                nodes=selected, similarity_top_k=1, language="english", verbose=False,
            ) if selected else None
        index = self._indexes[key]
        if index is None:
            return []
        # Avoid mutating the cached index's requested K across calls.
        from copy import copy
        request = copy(index)
        request.similarity_top_k = min(top_k, len(index.corpus))
        hits = request.retrieve(question)
        # No keyword match is not useful evidence. bm25s otherwise fills with zeros.
        return [hit for hit in hits if hit.score is not None and hit.score > 0]

    def retrieve(self, question: str, top_k: int = 5, filters=None) -> list[RetrievalHit]:
        return as_hits(self.retrieve_nodes(question, top_k, filters))
