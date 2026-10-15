"""Bridge existing Qdrant payloads to LlamaIndex without re-embedding documents."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.indexing.models import RetrievalHit
from src.retrieval.dense import DenseRetriever


def snapshot_nodes(snapshot: Mapping[str, dict[str, Any]]) -> dict[str, TextNode]:
    nodes = {}
    for point_id, item in sorted(snapshot.items()):
        payload = deepcopy(item["payload"])
        chunk_id = payload["chunk_id"]
        if chunk_id in nodes:
            raise ValueError("snapshot contains duplicate chunk IDs")
        nodes[chunk_id] = TextNode(
            id_=chunk_id, text=payload["text"],
            metadata={"point_id": point_id, "payload": payload},
            excluded_embed_metadata_keys=["point_id", "payload"],
            excluded_llm_metadata_keys=["point_id", "payload"],
        )
    if not nodes:
        raise ValueError("retrieval requires a nonempty index")
    return nodes


def as_hits(nodes: Sequence[NodeWithScore]) -> list[RetrievalHit]:
    result = []
    for rank, scored in enumerate(nodes, 1):
        payload = scored.node.metadata["payload"]
        result.append(RetrievalHit(
            rank=rank, score=float(scored.score or 0), point_id=scored.node.metadata["point_id"],
            chunk_id=payload["chunk_id"], document_id=payload["document_id"], text=payload["text"],
            section_path=payload.get("section_path", []), page_numbers=payload.get("page_numbers", []),
            source_refs=payload.get("source_refs", []), source_file=payload.get("source_file"),
            validation_status=payload.get("validation_status"), token_count=payload.get("token_count"),
            metadata=payload.get("metadata", {}),
        ))
    return result


class DenseLlamaIndexRetriever(BaseRetriever):
    """A native LlamaIndex retriever over the existing BGE/Qdrant collection."""

    def __init__(self, dense: DenseRetriever, nodes: Mapping[str, TextNode], top_k: int, filters=None):
        super().__init__()
        self.dense, self.nodes, self.top_k, self.filters = dense, nodes, top_k, filters

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        import numpy as np

        vector = (np.asarray(query_bundle.embedding, dtype=np.float32)
                  if query_bundle.embedding is not None
                  else self.dense.embedder.embed_query(query_bundle.query_str))
        hits = self.dense.vector_store.search(vector, limit=self.top_k, filters=self.filters)
        if any(hit.chunk_id not in self.nodes for hit in hits):
            raise RuntimeError("index changed; rebuild the retrieval snapshot")
        return [NodeWithScore(node=self.nodes[hit.chunk_id], score=hit.score) for hit in hits]


class RankedResultsRetriever(BaseRetriever):
    """Feed already-computed candidates into native query fusion exactly once."""

    def __init__(self, results: Sequence[NodeWithScore]):
        super().__init__()
        self.results = results

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # QueryFusionRetriever updates result scores; never mutate another run.
        return [NodeWithScore(node=item.node, score=item.score) for item in self.results]
