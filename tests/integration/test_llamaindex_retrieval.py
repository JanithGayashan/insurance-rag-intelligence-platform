import pytest
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore

from src.retrieval.hybrid import LlamaIndexRetrievalSuite
from src.retrieval.llamaindex import RankedResultsRetriever, as_hits, snapshot_nodes
from tests.integration.evaluation.test_dense_evaluation import fixture


def test_native_bm25_preserves_provenance_and_filters_before_ranking(fixture):
    _, dense = fixture
    suite = LlamaIndexRetrievalSuite(dense)
    hits = suite.retrieve("flood", method="bm25", top_k=20, filters={"document_id": "doc-b"})
    assert len(hits) == 1
    assert hits[0].document_id == "doc-b"
    assert hits[0].source_file == "doc-b.pdf"
    assert hits[0].page_numbers == [1]
    assert hits[0].source_refs
    assert suite.retrieve("flood", method="bm25", filters={"document_id": "absent"}) == []
    assert suite.retrieve("zzzzzzzzzzzz", method="bm25") == []


def test_bm25_does_not_embed_queries(fixture, monkeypatch):
    _, dense = fixture
    def fail(*args, **kwargs):
        raise AssertionError("BM25 must not embed a query")
    monkeypatch.setattr(dense.embedder, "embed_query", fail)
    assert LlamaIndexRetrievalSuite(dense).retrieve("flood", method="bm25")


def test_hybrid_embeds_once_and_never_calls_llm(fixture, monkeypatch):
    _, dense = fixture
    original = dense.embedder.embed_query
    calls = []
    def embed(text):
        calls.append(text)
        return original(text)
    def fail(*args, **kwargs):
        raise AssertionError("Query fusion must not call an LLM")
    monkeypatch.setattr(dense.embedder, "embed_query", embed)
    monkeypatch.setattr(MockLLM, "complete", fail)
    results = LlamaIndexRetrievalSuite(dense).retrieve_all("flood")
    assert len(calls) == 1
    assert set(results) == {"dense", "bm25", "hybrid"}
    assert len({hit.chunk_id for hit in results["hybrid"]}) == len(results["hybrid"])
    # Identical text in different policies must remain separate nodes.
    assert {hit.document_id for hit in results["hybrid"]} >= {"doc-a", "doc-b"}


def test_native_rrf_rewards_agreement_and_does_not_mutate_original_scores(fixture):
    _, dense = fixture
    nodes = list(snapshot_nodes(dense.vector_store.snapshot()).values())
    a, b, c = nodes
    left = [NodeWithScore(node=a, score=0.9), NodeWithScore(node=b, score=0.8)]
    right = [NodeWithScore(node=c, score=10), NodeWithScore(node=b, score=9)]
    fusion = QueryFusionRetriever([RankedResultsRetriever(left), RankedResultsRetriever(right)],
                                 llm=MockLLM(), num_queries=1, mode="reciprocal_rerank", use_async=False)
    hits = fusion.retrieve("question")
    assert hits[0].node.node_id == b.node_id
    assert hits[0].score == pytest.approx(2 / 61)
    assert left[1].score == 0.8
    assert as_hits(hits)[0].chunk_id == b.node_id


def test_dense_adapter_keeps_existing_rankings(fixture):
    _, dense = fixture
    suite = LlamaIndexRetrievalSuite(dense)
    expected = dense.retrieve("flood", top_k=3)
    actual = suite.retrieve("flood", top_k=3, method="dense")
    assert [hit.chunk_id for hit in actual] == [hit.chunk_id for hit in expected]
    assert [hit.score for hit in actual] == pytest.approx([hit.score for hit in expected])


@pytest.mark.parametrize("kwargs", [{"question": ""}, {"question": "flood", "top_k": 0}, {"question": "flood", "method": "bad"}])
def test_invalid_retrieval_requests_are_rejected(fixture, kwargs):
    _, dense = fixture
    with pytest.raises(ValueError):
        LlamaIndexRetrievalSuite(dense).retrieve(**kwargs)
