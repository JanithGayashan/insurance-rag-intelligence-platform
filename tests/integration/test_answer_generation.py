import json

import httpx
import pytest
from openai import OpenAI
from llama_index.llms.openai import OpenAIResponses

from src.generation.configuration import GenerationConfiguration
from src.generation.context import prepare_context
from src.generation.llamaindex import GenerationError, LlamaIndexAnswerGenerator, response_format
from src.generation.pipeline import RAGAnswerService
from src.retrieval.hybrid import LlamaIndexRetrievalSuite
from tests.integration.evaluation.test_dense_evaluation import fixture
from tests.unit.generation.test_answers import draft
from tests.unit.generation.test_context import hit


def native_generator(handler, configuration=None):
    configuration = configuration or GenerationConfiguration()
    client = OpenAI(api_key="sk-test-not-a-real-key", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    llm = OpenAIResponses(
        model=configuration.model, api_key="sk-test-not-a-real-key", openai_client=client,
        max_output_tokens=configuration.max_output_tokens, max_retries=0,
        reasoning_options={"effort": configuration.reasoning_effort},
        additional_kwargs={"reasoning": {"effort": configuration.reasoning_effort}, "text": response_format()},
        store=False, track_previous_responses=False, context_window=400000,
    )
    return LlamaIndexAnswerGenerator(configuration, llm=llm), client


def response_body(text, *, status="completed", refusal=False):
    content = [{"type": "refusal", "refusal": "Declined"}] if refusal else [
        {"type": "output_text", "text": text, "annotations": []}]
    return {"id": "resp_mocked", "object": "response", "created_at": 1,
            "model": "gpt-5-nano", "status": status, "error": None,
            "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
            "output": [{"type": "message", "id": "msg_test", "role": "assistant", "status": "completed", "content": content}],
            "usage": {"input_tokens": 800, "output_tokens": 200, "total_tokens": 1000,
                      "input_tokens_details": {"cached_tokens": 100}, "output_tokens_details": {"reasoning_tokens": 50}}}


def test_actual_llamaindex_openai_sdk_serializes_bounded_responses_request():
    requests = []
    def handle(request):
        requests.append(request)
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        assert body["model"] == "gpt-5-nano"
        assert body["reasoning"] == {"effort": "minimal"}
        assert body["max_output_tokens"] == 1800
        assert body["store"] is False
        assert "temperature" not in body and "top_p" not in body
        assert body["text"]["format"]["strict"] is True
        assert body["text"]["format"]["schema"]["additionalProperties"] is False
        assert hit().text in str(body["input"])
        assert "S1" in str(body["input"])
        assert not body.get("tools")
        return httpx.Response(200, json=response_body(json.dumps(draft())))
    generator, client = native_generator(handle)
    try:
        output = generator.generate(prepare_context("Flood?", [hit()], GenerationConfiguration()))
    finally:
        client.close()
    assert len(requests) == 1
    assert output.status == "completed" and output.response_id == "resp_mocked"
    assert output.usage.input_tokens == 800 and output.usage.reasoning_tokens == 50
    assert output.usage.estimated_cost_usd is not None


@pytest.mark.parametrize("model, effort", [("gpt-5-nano", "minimal"), ("gpt-6-luna", "none")])
def test_production_constructor_configures_native_client_without_hidden_retries(monkeypatch, model, effort):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response_body(json.dumps(draft())))
    transport = httpx.Client(transport=httpx.MockTransport(handle))
    sdk_client = OpenAI(api_key="sk-test-not-a-real-key", max_retries=0, http_client=transport)
    original = OpenAIResponses
    created = []
    def construct(**kwargs):
        kwargs["openai_client"] = sdk_client
        value = original(**kwargs)
        created.append(value)
        return value
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr("llama_index.llms.openai.OpenAIResponses", construct)
    configuration = GenerationConfiguration(model=model, reasoning_effort=effort)
    generator = LlamaIndexAnswerGenerator(configuration)
    try:
        output = generator.generate(prepare_context("Flood?", [hit()], configuration))
        assert output.status == "completed"
        assert created[0].max_retries == 0
        assert len(requests) == 1 and requests[0]["reasoning"] == {"effort": effort}
        assert requests[0]["model"] == model
        assert requests[0]["store"] is False
        assert "sk-test-not-a-real-key" not in json.dumps(requests[0])
    finally:
        generator.close()
        sdk_client.close()


def test_real_bm25_qdrant_and_llamaindex_generation_keep_selected_policy(fixture):
    _, dense = fixture
    suite = LlamaIndexRetrievalSuite(dense)
    captured = []
    def handle(request):
        body = json.loads(request.content)
        captured.append(body)
        assert "doc-a.pdf" in str(body["input"]) and "doc-b.pdf" not in str(body["input"])
        value = {"status": "answered", "claims": [{"text": "Flood damage is covered.",
                 "evidence": [{"source_id": "S1", "quote": "Flood damage is covered."}]}]}
        return httpx.Response(200, json=response_body(json.dumps(value)))
    generator, client = native_generator(handle)
    try:
        result = RAGAnswerService(suite, generator=generator, method="hybrid").answer(
            "Is flood covered?", filters={"document_id": "doc-a"})
    finally:
        client.close()
    assert result.status == "answered" and len(captured) == 1
    assert all(source.document_id == "doc-a" for source in result.citations)
    assert result.diagnostics["index_snapshot_sha256"]
    assert result.diagnostics["collection"] == "evaluation"


@pytest.mark.parametrize("status, refusal", [("incomplete", False), ("completed", True)])
def test_native_provider_incomplete_and_refusal_metadata(status, refusal):
    generator, client = native_generator(lambda request: httpx.Response(200, json=response_body("", status=status, refusal=refusal)))
    try:
        output = generator.generate(prepare_context("Flood?", [hit()], GenerationConfiguration()))
    finally:
        client.close()
    assert output.status == status and output.refused == refusal


def test_real_sdk_authentication_error_is_redacted_without_retry():
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "secret-provider-body-sk-sensitive", "type": "invalid_api_key", "code": "invalid_api_key"}})
    generator, client = native_generator(handle)
    try:
        with pytest.raises(GenerationError) as error:
            generator.generate(prepare_context("Flood?", [hit()], GenerationConfiguration()))
    finally:
        client.close()
    assert len(requests) == 1
    assert "secret-provider-body" not in str(error.value)
    assert "OPENAI_API_KEY" in str(error.value)
