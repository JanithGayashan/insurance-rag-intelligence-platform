import json

import pytest

from src.generation.configuration import GenerationConfiguration, require_api_key
from src.generation.llamaindex import GenerationError, ProviderOutput, usage_from_response
from src.generation.models import TokenUsage
from src.generation.pipeline import RAGAnswerService, save_answer_artifacts
from tests.unit.generation.test_context import hit


class Suite:
    METHODS = ("hybrid_reranked",)

    def __init__(self, hits=None):
        self.hits = [hit()] if hits is None else hits
        self.calls = []

    def retrieve_all_with_diagnostics(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return {"hybrid_reranked": self.hits}, {"method_seconds": {"hybrid_reranked": 0.1}}


def draft(source_id="S1", quote="Flood damage is covered, subject to policy conditions."):
    return {"status": "answered", "claims": [{"text": "Flood damage is covered, subject to the policy conditions.",
            "evidence": [{"source_id": source_id, "quote": quote}]}]}


class Generator:
    def __init__(self, value=None, *, status="completed", refused=False):
        self.value = draft() if value is None else value
        self.status, self.refused, self.calls = status, refused, []

    def generate(self, context):
        self.calls.append(context)
        return ProviderOutput(json.dumps(self.value), self.status, "resp_test", self.refused, TokenUsage(input_tokens=400, output_tokens=100))


def test_answer_citations_are_built_from_verified_context_not_model_file_names():
    suite, generator = Suite(), Generator()
    result = RAGAnswerService(suite, generator=generator).answer("Flood?", filters={"source_file": "a.pdf"})
    assert result.status == "answered"
    assert result.answer.endswith("[S1]")
    assert result.citations[0].source_file == "a.pdf" and result.citations[0].page_numbers == [2]
    assert suite.calls[0][1]["filters"] == {"source_file": "a.pdf"}
    assert len(generator.calls) == 1 and result.diagnostics["api_calls"] == 1
    assert result.usage.output_tokens == 100


@pytest.mark.parametrize("value", [draft(source_id="S99"), draft(quote="Flood claims are always fully reimbursed."),
    draft(quote=" " * 20),
    {"status": "answered", "claims": []}, {"status": "insufficient_evidence", "claims": draft()["claims"]},
    {"status": "answered", "claims": [{"text": "Covered.", "evidence": []}]},
    {"answer": "A fabricated answer without any evidence."}])
def test_invalid_or_fabricated_citations_block_entire_answer(value):
    result = RAGAnswerService(Suite(), generator=Generator(value)).answer("Flood?")
    assert result.status == "invalid_output"
    assert not result.claims and not result.citations
    assert result.validation_errors


def test_quote_from_another_source_cannot_be_attributed_to_first_source():
    result = RAGAnswerService(Suite([hit(), hit("War damage is explicitly excluded from this policy.", chunk="b-1", document="b")]),
        generator=Generator(draft(quote="War damage is explicitly excluded"))).answer("Flood?")
    assert result.status == "invalid_output"


def test_insufficient_evidence_is_not_published_as_coverage_exclusion():
    result = RAGAnswerService(Suite(), generator=Generator({"status": "insufficient_evidence", "claims": []})).answer("Space travel?")
    assert result.status == "insufficient_evidence"
    assert not result.citations and not result.claims


@pytest.mark.parametrize("dry_run, hits", [(True, [hit()]), (False, [])])
def test_dry_run_and_empty_context_never_call_generator(dry_run, hits):
    generator = Generator()
    result = RAGAnswerService(Suite(hits), generator=generator).answer("Flood?", dry_run=dry_run)
    assert not generator.calls and result.diagnostics["api_calls"] == 0
    assert result.status == ("prepared" if dry_run else "insufficient_evidence")


@pytest.mark.parametrize("status, refused, expected", [("incomplete", False, "incomplete"),
    ("failed", False, "incomplete"), ("completed", True, "refused")])
def test_partial_and_refused_responses_never_publish_partial_claims(status, refused, expected):
    result = RAGAnswerService(Suite(), generator=Generator(status=status, refused=refused)).answer("Flood?")
    assert result.status == expected and not result.claims and not result.citations


def test_long_question_is_rejected_before_retrieval_or_generation():
    suite, generator = Suite(), Generator()
    with pytest.raises(ValueError):
        RAGAnswerService(suite, generator=generator).answer("word " * 700)
    assert not suite.calls and not generator.calls


def test_safe_provider_failure_can_be_saved_without_publishing_claims():
    class FailingGenerator:
        def generate(self, context):
            raise GenerationError("OpenAI request timed out. No automatic retry was made.")
    result = RAGAnswerService(Suite(), generator=FailingGenerator()).answer("Flood?")
    assert result.status == "provider_error"
    assert not result.citations and not result.claims
    assert result.diagnostics["api_calls"] == 1
    assert result.usage.estimated_cost_usd is None


def test_artifacts_include_full_context_prompt_and_answer_without_environment_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-sensitive-do-not-save")
    result = RAGAnswerService(Suite(), generator=Generator()).answer("Flood?")
    root = save_answer_artifacts(result, tmp_path)
    assert {path.name for path in root.iterdir()} == {"answer.json", "answer.txt", "context.txt", "prompt.json"}
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.iterdir())
    assert "sk-test-sensitive" not in text and hit().text in text
    assert json.loads((root / "prompt.json").read_text(encoding="utf-8"))["text"]["format"]["strict"]


def test_usage_does_not_double_count_reasoning_tokens():
    usage = usage_from_response({"usage": {"input_tokens": 1000, "output_tokens": 100,
        "input_tokens_details": {"cached_tokens": 200}, "output_tokens_details": {"reasoning_tokens": 80}}}, "gpt-5-nano")
    assert usage.estimated_cost_usd == pytest.approx((800 * .05 + 200 * .005 + 100 * .4) / 1e6)
    assert usage.reasoning_tokens == 80
    assert usage_from_response({}, "gpt-5-nano").estimated_cost_usd is None


def test_model_override_gets_compatible_default_reasoning(tmp_path, monkeypatch):
    for name in ("OPENAI_MODEL", "OPENAI_REASONING_EFFORT", "OPENAI_MAX_OUTPUT_TOKENS", "RAG_CONTEXT_TOKEN_BUDGET"):
        monkeypatch.delenv(name, raising=False)
    configuration = GenerationConfiguration.from_environment(tmp_path / "absent.env", model="gpt-6-luna")
    assert configuration.reasoning_effort == "none"
    with pytest.raises(ValueError):
        GenerationConfiguration(model="gpt-5-nano", reasoning_effort="none")


def test_api_key_error_does_not_echo_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-secret")
    with pytest.raises(ValueError, match="OPENAI_API_KEY") as error:
        require_api_key()
    assert "placeholder-secret" not in str(error.value)
