import hashlib
import json

import numpy as np

from src.chunking.semantic.custom import CustomSemanticChunker
from src.chunking.semantic.llamaindex import LlamaIndexSemanticChunker


class DeterministicEmbedder:
    model_name = "deterministic-test-embedder"

    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(24, dtype=np.float32)
            lowered = text.casefold()
            for term, index in {
                "cover": 0,
                "repair": 1,
                "damage": 2,
                "claim": 8,
                "report": 9,
                "document": 10,
                "exclude": 16,
                "war": 17,
            }.items():
                vector[index] = lowered.count(term)
            if not vector.any():
                digest = hashlib.sha256(text.encode()).digest()
                vector = np.asarray([byte / 255 for byte in digest[:24]], dtype=np.float32)
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.asarray(vectors)


def sample_document():
    texts = [
        ("section_header", "Coverage"),
        (
            "text",
            "Accidental damage cover pays vehicle repair costs. "
            "The repair cover includes approved replacement parts. "
            "Claims must be reported within seven days. "
            "Claim documents include the police report and repair estimate.",
        ),
        ("section_header", "Exclusions"),
        (
            "text",
            "War damage is excluded by this policy. "
            "Nuclear risks and deliberate damage are also excluded.",
        ),
    ]
    nodes = [
        {
            "self_ref": f"#/texts/{index}",
            "children": [],
            "label": label,
            "text": text,
            "prov": [{"page_no": index // 2 + 1}],
        }
        for index, (label, text) in enumerate(texts)
    ]
    return {
        "name": "test-policy",
        "body": {
            "self_ref": "#/body",
            "children": [{"$ref": node["self_ref"]} for node in nodes],
        },
        "texts": nodes,
        "groups": [],
        "tables": [],
        "pictures": [],
        "key_value_items": [],
        "form_items": [],
    }


def test_custom_semantic_chunker_detects_topics_without_crossing_sections():
    chunks = CustomSemanticChunker(
        DeterministicEmbedder(),
        min_chunk_size=8,
        target_chunk_size=20,
        max_chunk_size=45,
        breakpoint_percentile=75,
        buffer_size=0,
    ).chunk_document(sample_document(), "policy")

    assert len(chunks) >= 3
    assert all(chunk.token_count <= 45 for chunk in chunks)
    assert all(chunk.metadata["section_boundary_enforced"] for chunk in chunks)
    assert {chunk.section_path[-1] for chunk in chunks} == {"Coverage", "Exclusions"}
    assert any(
        chunk.metadata["boundary_reason"] == "semantic_distance" for chunk in chunks
    )


def test_llamaindex_semantic_chunker_uses_same_hard_section_boundaries():
    chunks = LlamaIndexSemanticChunker(
        DeterministicEmbedder(),
        min_chunk_size=8,
        target_chunk_size=20,
        max_chunk_size=45,
        breakpoint_percentile=75,
        buffer_size=1,
    ).chunk_document(sample_document(), "policy")

    assert chunks
    assert all(chunk.token_count <= 45 for chunk in chunks)
    assert all(chunk.metadata["section_boundary_enforced"] for chunk in chunks)
    assert {chunk.section_path[-1] for chunk in chunks} == {"Coverage", "Exclusions"}


def test_chunk_file_accepts_utf8_docling_json(tmp_path):
    path = tmp_path / "document.json"
    path.write_text(json.dumps(sample_document()), encoding="utf-8")
    chunks = CustomSemanticChunker(
        DeterministicEmbedder(), min_chunk_size=8, target_chunk_size=20, max_chunk_size=45
    ).chunk_file(path, "policy")
    assert chunks[0].document_id == "policy"
