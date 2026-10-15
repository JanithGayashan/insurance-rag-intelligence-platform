from copy import deepcopy

import numpy as np
import pytest
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval.reranking import LlamaIndexCrossEncoderReranker


class CharacterTokenizer:
    """Deterministic tokenizer fixture; no downloads or model weights in CI."""

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [ord(char) for char in text] + ([0, 0] if add_special_tokens else [])

    def decode(self, ids, **kwargs):
        return "".join(chr(value) for value in ids)

    def __call__(self, text, pair=None, **kwargs):
        return {"input_ids": self.encode(text) + self.encode(pair or "") + [0] * (3 if pair is not None else 2)}

    def get_vocab(self):
        return {"character-fixture": 0}


class PairPredictor:
    def __init__(self, score=None):
        self.pairs = []
        self.score = score or (lambda question, passage: 5.0 if "Flood damage is covered." in passage else -2.0)

    def predict(self, pairs):
        self.pairs.extend(pairs)
        return np.asarray([self.score(question, passage) for question, passage in pairs], dtype=np.float32)


def make_reranker(score=None, max_tokens=512):
    predictor = PairPredictor(score)
    native = SentenceTransformerRerank.model_construct(
        model="test-native-postprocessor", top_n=1, device="cpu", keep_retrieval_score=True,
        trust_remote_code=False,
    )
    native._model = predictor
    reranker = LlamaIndexCrossEncoderReranker(
        postprocessor=native, tokenizer=CharacterTokenizer(), max_sequence_length=max_tokens,
        window_overlap_tokens=min(16, max_tokens // 4),
    )
    return reranker, predictor


def make_node(chunk_id, text, score=0.03):
    metadata = {"point_id": f"point-{chunk_id}", "payload": {
        "chunk_id": chunk_id, "document_id": "doc-a", "source_file": "doc-a.pdf", "text": text,
        "page_numbers": [3], "source_refs": ["#/texts/8"], "metadata": {"existing": True},
    }}
    return NodeWithScore(node=TextNode(id_=chunk_id, text=text, metadata=metadata,
                                     excluded_embed_metadata_keys=list(metadata)), score=score)


def test_native_postprocessor_reorders_candidates_and_preserves_full_provenance():
    reranker, predictor = make_reranker()
    nodes = [make_node("a", "Policy introduction"), make_node("b", "Flood damage is covered.")]
    before = deepcopy(nodes)
    result = reranker.rerank_nodes("Is flood damage covered?", nodes, top_k=1)
    assert result[0].node.node_id == "b"
    assert result[0].score == 5
    payload = result[0].node.metadata["payload"]
    assert payload["text"] == nodes[1].node.text
    assert payload["page_numbers"] == [3]
    assert payload["metadata"]["existing"] is True
    assert payload["metadata"]["reranking"]["input_rank"] == 2
    assert payload["metadata"]["reranking"]["input_score"] == 0.03
    assert nodes == before
    assert reranker.postprocessor.top_n == 1
    assert all("Policy source: doc-a.pdf" in passage for _, passage in predictor.pairs)


def test_long_passages_score_tail_windows_without_silent_truncation():
    reranker, predictor = make_reranker(score=lambda q, p: 10 if "EVIDENCE" in p else 0, max_tokens=96)
    text = "x" * 150 + " EVIDENCE " + "y" * 25
    result = reranker.rerank_nodes("find", [make_node("long", text)])
    assert result[0].score == 10
    assert result[0].node.metadata["payload"]["text"] == text
    info = result[0].node.metadata["payload"]["metadata"]["reranking"]
    assert info["window_count"] > 1
    assert info["best_window_index"] > 0
    assert info["max_pair_tokens"] <= 96
    assert info["truncated"] is False
    assert all(len(q) + len(p) + 3 <= 96 for q, p in predictor.pairs)
    # Every original character token is covered by temporary windows.
    windows = reranker._windows("find", make_node("long", text))
    covered = set().union(*(set(range(start, end)) for _, start, end in windows))
    assert covered == set(range(len(text)))


def test_ties_keep_hybrid_order_and_negative_scores_are_valid():
    reranker, _ = make_reranker(score=lambda q, p: -3)
    nodes = [make_node("first", "first text"), make_node("second", "second text")]
    assert [item.node.node_id for item in reranker.rerank_nodes("question", nodes)] == ["first", "second"]


def test_empty_input_does_not_call_the_model():
    reranker, predictor = make_reranker()
    assert reranker.rerank_nodes("question", []) == []
    assert predictor.pairs == []


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_are_rejected(score):
    reranker, _ = make_reranker(score=lambda q, p: score)
    with pytest.raises(ValueError, match="non-finite"):
        reranker.rerank_nodes("question", [make_node("a", "text")])


def test_duplicate_candidates_and_oversized_questions_are_rejected():
    reranker, predictor = make_reranker(max_tokens=96)
    node = make_node("a", "text")
    with pytest.raises(ValueError, match="unique"):
        reranker.rerank_nodes("question", [node, node])
    with pytest.raises(ValueError, match="input budget"):
        reranker.rerank_nodes("q" * 120, [node])
    assert predictor.pairs == []


@pytest.mark.parametrize("kwargs", [{"revision": "main"}, {"model_name": "custom/model"},
                                   {"max_sequence_length": 513}, {"window_overlap_tokens": -1},
                                   {"postprocessor": object()}])
def test_invalid_configuration_fails_before_model_download(kwargs):
    with pytest.raises(ValueError):
        LlamaIndexCrossEncoderReranker(**kwargs)
