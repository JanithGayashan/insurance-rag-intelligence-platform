import asyncio
from contextlib import contextmanager
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.app import RequestLimits, create_app
from src.api.configuration import APIConfiguration
from src.api.models import ChatRequest, ChatResponse, Citation, PolicyDocument
from src.api.runtime import BackendGenerationError, RAGRuntime
from tests.unit.generation.test_answers import Generator, Suite
from src.generation.pipeline import RAGAnswerService


TOKEN = "test-only-backend-token-0123456789abcdef"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


class FakeRuntime:
    def __init__(self, settings):
        self.started, self.closed, self.calls = [], [], []
        self.documents = [PolicyDocument(document_id="policy-a", source_file="motor/a.pdf", display_name="a.pdf", chunk_count=2)]
        self.error = None

    def start(self):
        self.started.append(threading.get_ident())

    def close(self):
        self.closed.append(threading.get_ident())

    def answer(self, payload):
        self.calls.append((threading.get_ident(), payload))
        if self.error:
            raise self.error
        return ChatResponse(request_id="run-test", status="answered", answer="Flood is covered. [S1]",
                            citations=[Citation(source_id="S1", document_id="policy-a", source_file="motor/a.pdf",
                                                page_numbers=[1], quotes=["Flood damage is covered."])])


@contextmanager
def client_fixture(**overrides):
    settings = APIConfiguration(backend_api_key=TOKEN, **overrides)
    runtimes = []
    def factory(configuration):
        value = FakeRuntime(configuration)
        runtimes.append(value)
        return value
    app = create_app(settings, runtime_factory=factory)
    with TestClient(app) as client:
        yield client, runtimes[0], app


def test_startup_reuses_service_and_shutdown_runs_on_same_worker_thread():
    with client_fixture() as (client, runtime, app):
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").status_code == 200
        for _ in range(2):
            response = client.post("/v1/chat", json={"message": "Flood?", "document_id": "policy-a"}, headers=HEADERS)
            assert response.status_code == 200
        assert len(runtime.started) == 1
        assert {call[0] for call in runtime.calls} == set(runtime.started)
    assert runtime.closed == runtime.started


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer incorrect"}, {"Authorization": "Basic test"}])
def test_authentication_prevents_retrieval_and_generation(headers):
    with client_fixture() as (client, runtime, _):
        for path, method in [("/v1/chat", "POST"), ("/v1/documents", "GET"), ("/v1/dataset", "GET")]:
            response = client.request(method, path, headers=headers, json={"message": "Flood?"} if method == "POST" else None)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "unauthorized"
        assert not runtime.calls


def test_dataset_metadata_and_readiness_identify_the_active_corpus():
    with client_fixture(dataset_id="slic") as (client, runtime, _):
        assert client.get("/health/ready").json()["dataset_id"] == "slic"
        metadata = client.get("/v1/dataset", headers=HEADERS).json()
        assert metadata["id"] == "slic" and metadata["short_name"] == "SLIC"
        assert "qdrant_path" not in metadata and "project_root" not in metadata
        assert not runtime.calls


def test_explicit_dataset_selects_its_own_api_storage_defaults():
    slic = APIConfiguration(backend_api_key=TOKEN, dataset_id="slic")
    allianz = APIConfiguration(backend_api_key=TOKEN, dataset_id="allianz")
    assert slic.qdrant_path != allianz.qdrant_path
    assert slic.collection != allianz.collection
    assert slic.output_directory != allianz.output_directory


def test_public_response_contains_citations_without_private_context_or_usage():
    with client_fixture() as (client, _, _):
        response = client.post("/v1/chat", headers=HEADERS, json={"message": "Flood?"})
        assert set(response.json()) == {"request_id", "status", "answer", "citations"}
        assert response.json()["citations"][0]["quotes"] == ["Flood damage is covered."]
        assert response.headers["X-Request-ID"] == response.json()["request_id"]
        assert response.headers["Cache-Control"] == "no-store"
        assert TOKEN not in response.text
        assert client.get("/v1/documents", headers=HEADERS).json()[0]["document_id"] == "policy-a"


@pytest.mark.parametrize("payload", [{"message": " "}, {"message": "x" * 4001}, {"message": "Flood?", "model": "expensive-model"},
                                      {"message": "Flood?", "history": []}, {}])
def test_invalid_requests_do_not_call_rag(payload):
    with client_fixture() as (client, runtime, _):
        assert client.post("/v1/chat", json=payload, headers=HEADERS).status_code == 422
        assert not runtime.calls


def test_unknown_document_cannot_select_arbitrary_files():
    with client_fixture() as (client, runtime, _):
        response = client.post("/v1/chat", json={"message": "Flood?", "document_id": "../../.env"}, headers=HEADERS)
        assert response.status_code == 404 and not runtime.calls


def test_large_body_is_rejected_even_without_content_length():
    with client_fixture() as (client, runtime, _):
        response = client.post("/v1/chat", content=iter([b'{"message":"', b'x' * 17000, b'"}']),
                               headers={**HEADERS, "Content-Type": "application/json"})
        assert response.status_code == 413 and not runtime.calls


def test_rate_limit_blocks_paid_work_and_exposes_retry_after():
    with client_fixture(requests_per_minute=1) as (client, runtime, _):
        assert client.post("/v1/chat", json={"message": "Flood?"}, headers=HEADERS).status_code == 200
        response = client.post("/v1/chat", json={"message": "Flood?"}, headers=HEADERS)
        assert response.status_code == 429 and response.headers["Retry-After"]
        assert len(runtime.calls) == 1


def test_limit_windows_expire_but_day_budget_still_applies():
    now = [0]
    limits = RequestLimits(1, 2, clock=lambda: now[0])
    limits.admit()
    now[0] = 61
    limits.admit()
    now[0] = 122
    with pytest.raises(Exception) as error:
        limits.admit()
    assert error.value.status_code == 429
    now[0] = 86500
    limits.admit()


def test_busy_backend_returns_without_waiting_or_doing_work():
    with client_fixture() as (client, runtime, app):
        app.state.gate.acquire()
        try:
            response = client.post("/v1/chat", json={"message": "Flood?"}, headers=HEADERS)
            assert response.status_code == 503 and not runtime.calls
            assert client.get("/health/live").status_code == 200
        finally:
            app.state.gate.release()


def test_cancelled_http_request_keeps_inference_slot_until_worker_finishes():
    started, release = threading.Event(), threading.Event()
    class SlowRuntime(FakeRuntime):
        def answer(self, payload):
            started.set()
            assert release.wait(timeout=10)
            return super().answer(payload)
    async def exercise():
        app = create_app(APIConfiguration(backend_api_key=TOKEN), runtime_factory=SlowRuntime)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                first = asyncio.create_task(client.post("/v1/chat", json={"message": "Flood?"}, headers=HEADERS))
                try:
                    assert await asyncio.to_thread(started.wait, 3)
                    assert (await client.get("/health/live")).status_code == 200
                    assert (await client.post("/v1/chat", json={"message": "Flood?"}, headers=HEADERS)).status_code == 503
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await first
                    assert app.state.gate.locked()
                finally:
                    release.set()
                for _ in range(100):
                    if not app.state.gate.locked():
                        break
                    await asyncio.sleep(0.01)
                assert not app.state.gate.locked()
    asyncio.run(exercise())


@pytest.mark.parametrize("error, status, code", [(BackendGenerationError("failed-run"), 502, "generation_failed"),
    (RuntimeError("secret-provider-body-sk-sensitive"), 500, "internal_error"), (ValueError("private input"), 422, "invalid_request")])
def test_failures_do_not_leak_private_errors(error, status, code, caplog):
    with client_fixture() as (client, runtime, _):
        runtime.error = error
        response = client.post("/v1/chat", json={"message": "Flood?"}, headers=HEADERS)
        assert response.status_code == status and response.json()["error"]["code"] == code
        assert "secret-provider-body" not in response.text + caplog.text
        assert "private input" not in response.text
        if status == 502:
            assert response.json()["error"]["request_id"] == "failed-run"


def test_cors_allows_only_configured_origin():
    with client_fixture(allowed_origins=("https://portfolio.example",)) as (client, _, _):
        for origin, expected in [("https://portfolio.example", 200), ("https://untrusted.example", 400)]:
            response = client.options("/v1/chat", headers={"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"})
            assert response.status_code == expected
        assert client.post("/v1/chat", json={"message": "Flood?"}, headers={"Origin": "https://portfolio.example"}).status_code == 401


def test_weak_backend_configuration_does_not_echo_token_in_error():
    with pytest.raises(ValueError) as error:
        APIConfiguration(backend_api_key="private-secret-too-short")
    assert "private-secret-too-short" not in str(error.value)
    with pytest.raises(ValueError):
        APIConfiguration(backend_api_key=TOKEN, allowed_origins=("*",))


def test_runtime_projects_validated_answer_and_keeps_full_artifacts_private(tmp_path):
    runtime = RAGRuntime(APIConfiguration(backend_api_key=TOKEN, output_directory=tmp_path))
    runtime.service = RAGAnswerService(Suite(), generator=Generator())
    response = runtime.answer(ChatRequest(message="Flood?"))
    assert response.status == "answered"
    assert response.citations[0].page_numbers == [2]
    assert (tmp_path / response.request_id / "prompt.json").is_file()
    assert "context" not in response.model_dump()


def test_runtime_blocks_bad_citations_and_keeps_model_draft_in_private_report(tmp_path):
    import json
    from tests.unit.generation.test_answers import draft
    runtime = RAGRuntime(APIConfiguration(backend_api_key=TOKEN, output_directory=tmp_path))
    runtime.service = RAGAnswerService(Suite(), generator=Generator(draft(source_id="S99")))
    with pytest.raises(BackendGenerationError) as error:
        runtime.answer(ChatRequest(message="Flood?"))
    report = json.loads((tmp_path / error.value.request_id / "answer.json").read_text(encoding="utf-8"))
    assert report["status"] == "invalid_output"
    assert "S99" in report["diagnostics"]["model_output"]
    assert not report["citations"]


def test_startup_failure_still_closes_runtime_on_its_worker():
    instances = []
    class FailingRuntime(FakeRuntime):
        def start(self):
            super().start()
            raise RuntimeError("Synthetic startup failure")
    def factory(configuration):
        value = FailingRuntime(configuration)
        instances.append(value)
        return value
    with pytest.raises(RuntimeError, match="Synthetic startup failure"):
        with TestClient(create_app(APIConfiguration(backend_api_key=TOKEN), runtime_factory=factory)):
            pass
    assert instances[0].closed == instances[0].started
