import hashlib

import pytest
import tiktoken

from src.generation.configuration import GenerationConfiguration
from src.generation.context import context_text, prepare_context
from src.indexing.models import RetrievalHit


def hit(text="Flood damage is covered, subject to policy conditions.", *, chunk="a-1", document="a", **kwargs):
    values = dict(rank=1, score=0.8, point_id=chunk, chunk_id=chunk, document_id=document,
                  text=text, source_file=f"{document}.pdf", page_numbers=[2], section_path=["Coverage"], source_refs=["#/texts/1"])
    values.update(kwargs)
    return RetrievalHit(**values)


def test_complete_passages_and_provenance_survive_without_mutating_hits():
    original = hit("Coverage:\n- Flood damage.\n- Subject to a Rs. 500 excess.")
    context = prepare_context(" Is flood covered? ", [original], GenerationConfiguration())
    assert context.question == "Is flood covered?"
    source = context.sources[0]
    assert source.text == original.text
    assert source.page_numbers == [2] and source.section_path == ["Coverage"]
    assert source.source_refs == original.source_refs
    assert source.text_sha256 == hashlib.sha256(original.text.encode()).hexdigest()
    assert "source_id" not in original.metadata


def test_duplicates_are_removed_within_policy_but_never_across_policies():
    context = prepare_context("Flood?", [hit(), hit(chunk="a-2"), hit(document="b", chunk="b-1")], GenerationConfiguration())
    assert [item.document_id for item in context.sources] == ["a", "b"]
    assert context.skipped == [{"chunk_id": "a-2", "reason": "duplicate"}]


def test_oversized_chunk_is_skipped_whole_and_later_smaller_passage_can_fit():
    text = "This policy excludes war damage. " * 500
    context = prepare_context("Flood?", [hit(text), hit(chunk="a-2")], GenerationConfiguration(context_token_budget=256))
    assert [item.chunk_id for item in context.sources] == ["a-2"]
    assert context.sources[0].source_id == "S1"
    assert context.skipped[0]["reason"] == "context_budget"
    assert context.token_count == len(tiktoken.get_encoding("o200k_base").encode(context_text(context.sources)))
    assert context.token_count <= 256


@pytest.mark.parametrize("kwargs, reason", [({"text": "   "}, "empty_text"),
    ({"page_numbers": []}, "missing_provenance"), ({"page_numbers": [0]}, "missing_provenance"),
    ({"source_file": None}, "missing_provenance"), ({"score": float("nan")}, "nonfinite_score")])
def test_unusable_passages_do_not_enter_context(kwargs, reason):
    context = prepare_context("Flood?", [hit(**kwargs)], GenerationConfiguration())
    assert not context.sources and context.skipped[0]["reason"] == reason


def test_special_token_strings_in_document_are_treated_as_text():
    context = prepare_context("Flood?", [hit("Policy text <|endoftext|> containing special token-like text.")], GenerationConfiguration())
    assert len(context.sources) == 1


@pytest.mark.parametrize("question", ["", "   ", "flood " * 600])
def test_invalid_questions_are_rejected(question):
    with pytest.raises(ValueError):
        prepare_context(question, [], GenerationConfiguration())
