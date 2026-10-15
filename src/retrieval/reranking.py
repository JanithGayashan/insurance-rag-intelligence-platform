"""Pinned LlamaIndex cross-encoder reranking with explicit pair-length handling."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from copy import deepcopy
from importlib.metadata import version

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode

from src.indexing.configuration import json_sha256


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANKER_REVISION = "fbf9045f293a58fa68636213c5e0cb8a2de5d45e"


class LlamaIndexCrossEncoderReranker:
    """Score every candidate using native SentenceTransformerRerank.

    Only temporary scoring passages are windowed. Original indexed chunks,
    provenance and vectors are preserved. Candidate scores use the maximum
    window score; ties retain the original hybrid rank.
    """

    def __init__(self, model_name=DEFAULT_RERANKER_MODEL, revision=None,
                 device="cpu", max_sequence_length=512, window_overlap_tokens=32,
                 *, postprocessor=None, tokenizer=None):
        revision = revision or (DEFAULT_RERANKER_REVISION if model_name == DEFAULT_RERANKER_MODEL else "")
        if not model_name.strip() or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("reranker model and immutable 40-character revision are required")
        if not 8 <= max_sequence_length <= 512 or not 0 <= window_overlap_tokens < max_sequence_length:
            raise ValueError("pair limit must be 8..512 and overlap must be smaller than that limit")
        if (postprocessor is None) != (tokenizer is None):
            raise ValueError("provide both an injected native postprocessor and tokenizer, or neither")
        started = time.perf_counter()
        if postprocessor is None:
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer

            # LlamaIndex's pinned constructor has no revision parameter. A pinned
            # local snapshot supplies immutable weights without modifying library internals.
            snapshot = snapshot_download(
                model_name, revision=revision,
                allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json",
                                "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
                                "config_sentence_transformers.json", "modules.json", "sentence_bert_config.json"],
            )
            postprocessor = SentenceTransformerRerank(
                model=snapshot, top_n=1, device=device,
                keep_retrieval_score=True, trust_remote_code=False,
            )
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        if not isinstance(postprocessor, SentenceTransformerRerank):
            raise ValueError("a native LlamaIndex SentenceTransformerRerank postprocessor is required")
        self.postprocessor, self.tokenizer = postprocessor, tokenizer
        self.max_sequence_length, self.window_overlap_tokens = max_sequence_length, window_overlap_tokens
        tokenizer_json = (tokenizer.backend_tokenizer.to_str() if hasattr(tokenizer, "backend_tokenizer")
                          else json.dumps(tokenizer.get_vocab(), sort_keys=True))
        self.configuration = {
            "model_name": model_name, "model_revision": revision, "device": device,
            "max_sequence_length": max_sequence_length, "window_overlap_tokens": window_overlap_tokens,
            "window_aggregation": "max", "source_context": "source_file", "trust_remote_code": False,
            "tokenizer_sha256": hashlib.sha256(tokenizer_json.encode()).hexdigest(),
            "backend": {name: version(name) for name in ("llama-index-core", "sentence-transformers", "transformers", "torch")},
        }
        self.fingerprint = json_sha256(self.configuration)
        self.load_seconds = time.perf_counter() - started

    def _pair_length(self, question, passage):
        return len(self.tokenizer(question, passage, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])

    def _windows(self, question, node):
        payload = node.node.metadata["payload"]
        source = (payload.get("source_file") or "").replace("\\", "/")
        prefix = f"Policy source: {source}\n\n" if source else ""
        text = node.node.get_content(metadata_mode=MetadataMode.EMBED)
        ids = self.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        if not ids:
            raise ValueError("cannot rerank an empty passage")
        overhead = self._pair_length(question, prefix)
        budget = self.max_sequence_length - overhead - 4
        if budget < 1:
            raise ValueError("question and policy context exceed the reranker input budget")
        if self._pair_length(question, prefix + text) <= self.max_sequence_length:
            return [(prefix + text, 0, len(ids))]
        windows, start = [], 0
        while start < len(ids):
            end = min(start + budget, len(ids))
            while end > start:
                passage = prefix + self.tokenizer.decode(
                    ids[start:end], skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )
                excess = self._pair_length(question, passage) - self.max_sequence_length
                if excess <= 0:
                    break
                end -= max(1, excess)
            if end <= start:
                raise ValueError("cannot form a reranker window within the pair input budget")
            windows.append((passage, start, end))
            if end == len(ids):
                break
            overlap = min(self.window_overlap_tokens, (end - start) // 4)
            start = end - overlap
        return windows

    def rerank_nodes(self, question, nodes, top_k=5):
        if not question.strip() or top_k <= 0:
            raise ValueError("question must be nonempty and top_k must be positive")
        ids = [node.node.node_id for node in nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("reranking candidates must have unique chunk IDs")
        if not nodes:
            return []
        windows, per_chunk = [], {}
        for node in nodes:
            passages = self._windows(question, node)
            per_chunk[node.node.node_id] = passages
            for index, (text, start, end) in enumerate(passages):
                metadata = {"parent_chunk_id": node.node.node_id, "window_index": index,
                            "token_start": start, "token_end": end}
                scoring_node = TextNode(
                    id_=f"{node.node.node_id}:rerank-window:{index}", text=text, metadata=metadata,
                    excluded_embed_metadata_keys=list(metadata), excluded_llm_metadata_keys=list(metadata),
                )
                windows.append(NodeWithScore(node=scoring_node, score=node.score))
        # Native postprocessing mutates scores. Use disposable windows and a shallow
        # request copy so calls never change the baseline nodes or shared top_n.
        request = self.postprocessor.model_copy()
        request.top_n = len(windows)
        scored = request.postprocess_nodes(windows, query_str=question)
        if {node.node.node_id for node in scored} != {node.node.node_id for node in windows} or len(scored) != len(windows):
            raise RuntimeError("reranker did not score every candidate window")
        best = {}
        for item in scored:
            score = float(item.score) if item.score is not None else float("nan")
            if not math.isfinite(score):
                raise ValueError("reranker produced a non-finite score")
            parent = item.node.metadata["parent_chunk_id"]
            if parent not in best or score > best[parent][0]:
                best[parent] = (score, item)
        result = []
        for input_rank, node in enumerate(nodes, 1):
            score, best_window = best[node.node.node_id]
            original = deepcopy(node.node)
            original.metadata["payload"].setdefault("metadata", {})["reranking"] = {
                "fingerprint": self.fingerprint, "input_rank": input_rank, "input_score": node.score,
                "window_count": len(per_chunk[node.node.node_id]),
                "best_window_index": best_window.node.metadata["window_index"],
                "best_window_token_range": [best_window.node.metadata["token_start"], best_window.node.metadata["token_end"]],
                "max_pair_tokens": max(self._pair_length(question, text) for text, _, _ in per_chunk[node.node.node_id]),
                "truncated": False,
            }
            result.append(NodeWithScore(node=original, score=score))
        return sorted(result, key=lambda item: -item.score)[:top_k]
