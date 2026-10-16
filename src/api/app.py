from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
import math
import secrets
import threading
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.configuration import APIConfiguration
from src.api.models import ChatRequest, ChatResponse, PolicyDocument
from src.api.runtime import BackendGenerationError, RAGRuntime
from src.configuration.dataset import get_dataset_configuration


logger = logging.getLogger(__name__)
bearer = HTTPBearer(auto_error=False)
MAX_BODY_BYTES = 16384


class RequestLimits:
    """Process-local rolling limits for a small, single-worker portfolio demo."""

    def __init__(self, minute: int, day: int, clock=time.monotonic):
        self.minute_limit, self.day_limit, self.clock = minute, day, clock
        self.minute, self.day = deque(), deque()

    def admit(self):
        now = self.clock()
        for queue, window, limit in [(self.minute, 60, self.minute_limit), (self.day, 86400, self.day_limit)]:
            while queue and queue[0] <= now - window:
                queue.popleft()
            if len(queue) >= limit:
                raise HTTPException(429, "Request budget reached. Please try again later.",
                                    headers={"Retry-After": str(max(1, math.ceil(queue[0] + window - now)))})
        self.minute.append(now)
        self.day.append(now)


def error_response(request_id: str, status: int, code: str, message: str, headers=None):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "request_id": request_id}},
                        headers={"X-Request-ID": request_id, "Cache-Control": "no-store", **(headers or {})})


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_BODY_BYTES:
                response = error_response(uuid.uuid4().hex, 413, "request_too_large", "Request body exceeds 16 KB.")
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        return await self.app(scope, replay, send)


def create_app(configuration: APIConfiguration | None = None, *, runtime_factory=RAGRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        settings = configuration or APIConfiguration.from_environment()
        app.state.configuration = settings
        app.state.ready = False
        app.state.gate = threading.Lock()
        app.state.limits = RequestLimits(settings.requests_per_minute, settings.requests_per_day)
        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="insurance-rag")
        app.state.worker = worker
        runtime = runtime_factory(settings)
        app.state.runtime = runtime
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(worker, runtime.start)
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            try:
                await loop.run_in_executor(worker, runtime.close)
            finally:
                worker.shutdown(wait=True)

    app = FastAPI(title="Insurance RAG API", version="1.0.0", lifespan=lifespan)
    app.state.ready = False
    app.add_middleware(BodyLimitMiddleware)
    # Environment is read without requiring credentials during module import.
    # Startup is responsible for validating credentials and loading models.
    if configuration is not None:
        origins = configuration.allowed_origins
    else:
        from dotenv import load_dotenv
        import os
        from src.generation.configuration import PROJECT_ROOT
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        origins = tuple(value.strip() for value in os.getenv("RAG_ALLOWED_ORIGINS", "").split(",") if value.strip())
    if origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=list(origins), allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type"], expose_headers=["X-Request-ID"], allow_credentials=False)

    @app.middleware("http")
    async def request_metadata(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        try:
            response = await call_next(request)
        except Exception as error:
            # Handle before ServerErrorMiddleware can re-raise and log a raw
            # provider/storage exception body containing private configuration.
            logger.error("RAG request failed: request_id=%s type=%s", request.state.request_id, type(error).__name__)
            response = error_response(request.state.request_id, 500, "internal_error", "Unable to process the question. Please try again later.")
        response.headers.setdefault("X-Request-ID", request.state.request_id)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException):
        code = {401: "unauthorized", 404: "not_found", 422: "invalid_request", 429: "rate_limited", 503: "busy"}.get(error.status_code, "request_failed")
        return error_response(request.state.request_id, error.status_code, code, str(error.detail), error.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        return error_response(request.state.request_id, 422, "invalid_request", "Supply a nonempty message (up to 4,000 characters) and an optional document_id. No other fields are accepted.")

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception):
        identifier = getattr(request.state, "request_id", uuid.uuid4().hex)
        logger.error("RAG request failed: request_id=%s type=%s", identifier, type(error).__name__)
        return error_response(identifier, 500, "internal_error", "Unable to process the question. Please try again later.")

    async def authenticate(request: Request, credential: HTTPAuthorizationCredentials | None = Depends(bearer)):
        expected = request.app.state.configuration.backend_api_key.get_secret_value()
        if credential is None or not secrets.compare_digest(credential.credentials.encode(), expected.encode()):
            raise HTTPException(401, "A valid backend bearer token is required.", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health/live")
    async def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready():
        settings = getattr(app.state, "configuration", None)
        return JSONResponse({"status": "ready" if app.state.ready else "starting",
                             "dataset_id": settings.dataset_id if settings else None}, status_code=200 if app.state.ready else 503)

    @app.get("/v1/dataset", dependencies=[Depends(authenticate)])
    async def dataset():
        return get_dataset_configuration(app.state.configuration.dataset_id).public_metadata()

    @app.get("/v1/documents", response_model=list[PolicyDocument], dependencies=[Depends(authenticate)])
    async def documents():
        return app.state.runtime.documents

    @app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(authenticate)])
    async def chat(payload: ChatRequest, request: Request):
        if payload.document_id and payload.document_id not in {item.document_id for item in app.state.runtime.documents}:
            raise HTTPException(404, "The selected policy was not found.")
        gate = app.state.gate
        if not gate.acquire(blocking=False):
            raise HTTPException(503, "The assistant is answering another question. Please try again shortly.", headers={"Retry-After": "3"})
        try:
            app.state.limits.admit()
            future = app.state.worker.submit(app.state.runtime.answer, payload)
        except BaseException:
            gate.release()
            raise
        # Release only after real work finishes, including when the HTTP client
        # disconnects. A cancelled HTTP request must not start parallel inference.
        future.add_done_callback(lambda _: gate.release())
        try:
            result = await asyncio.shield(asyncio.wrap_future(future))
        except BackendGenerationError as error:
            request.state.request_id = error.request_id
            return error_response(error.request_id, 502, "generation_failed", "The assistant could not produce a validated answer. Please try again later.")
        except ValueError:
            raise HTTPException(422, "The question exceeds the supported input limit.") from None
        request.state.request_id = result.request_id
        return result

    return app


app = create_app()
