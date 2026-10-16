# Connect your Vercel portfolio to the insurance RAG backend

Keep your existing portfolio repository. This Python repository now exposes the
RAG pipeline through FastAPI. Your Next.js page calls a small server route in
the portfolio, and that route calls FastAPI with a private bearer token.

```text
Browser: portfolio chat page
  -> Next.js server route on Vercel: /api/insurance-chat
    -> Python FastAPI service: /v1/chat
      -> BGE query embedding + Qdrant + BM25 + hybrid fusion + reranker
        -> prepared context + LlamaIndex/OpenAI answer
    <- validated answer + source citations
  <- JSON for your chat UI
```

The frontend does not need Python, Qdrant credentials, the OpenAI SDK or your
OpenAI key. `RAG_BACKEND_API_KEY` is a separate secret used between the Next.js
server and Python server. Keep both keys out of browser code and variables
starting with `NEXT_PUBLIC_`.

## Start the backend locally

The existing OpenAI key remains in `.env`. A private backend token has also
been generated there without printing it. On a fresh checkout, generate it with:

```powershell
.\venv\Scripts\python.exe -m src.api.setup
```

This command preserves an existing token. Run the API from the repository root:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Open [interactive API documentation](http://127.0.0.1:8000/docs). Click **Authorize**
and enter the local `RAG_BACKEND_API_KEY`, then try the protected endpoints. Do
not enter your OpenAI key in the Swagger authorization box.

Models, the BM25 snapshot and the index connection load once at startup and are
reused. The first startup can take longer while loading models. Indexing is a
separate offline step; serving a question never rebuilds document embeddings.

Use one worker with the local Qdrant path. Do not run a CLI query or indexing
process against that path while this server owns it. Stop the server first, or
use a Qdrant server/Cloud with `QDRANT_URL` for separate processes. After corpus
updates, restart the backend to refresh its BM25/retrieval snapshot.

## Endpoints

| Method | Path | Purpose | Authentication |
| --- | --- | --- | --- |
| GET | `/health/live` | Process liveness | Public |
| GET | `/health/ready` | Models/index initialized | Public |
| GET | `/v1/documents` | Policy choices for your frontend dropdown | Backend bearer token |
| POST | `/v1/chat` | Ask a question | Backend bearer token |

Authentication header:

```http
Authorization: Bearer <RAG_BACKEND_API_KEY>
```

Get policy IDs from `/v1/documents`; each item contains `document_id`,
`source_file`, `display_name` and `chunk_count`. Do not use filesystem paths as
document IDs. A missing ID means search across the indexed policies.

Example request:

```json
{
  "message": "Is flood damage covered?",
  "document_id": "<document_id returned by /v1/documents>"
}
```

Example response shape (illustrative IDs):

```json
{
  "request_id": "generated-run-id",
  "status": "answered",
  "answer": "Flood is listed as covered under Section 1. [S1]",
  "citations": [
    {
      "source_id": "S1",
      "document_id": "selected-document-id",
      "source_file": "motor/Allianz_Motor_Insurance_Policy_(English).pdf",
      "page_numbers": [1],
      "quotes": ["<verbatim supporting quotation>"]
    }
  ]
}
```

Successful HTTP responses can have `answered`, `insufficient_evidence` or
`refused` status. Render the answer and, when present, expandable citation cards
containing filename, PDF pages and supporting quotes. This version returns a
complete JSON response after validation; it does not stream unvalidated text.

The API is currently **single-turn**. Your UI can display a conversation, but
each request must contain a self-contained question. Previous messages are not
automatically available to retrieval or generation. Do not send a `history`
field until conversation support is implemented.

Errors use this shape:

```json
{
  "error": {
    "code": "rate_limited",
    "message": "Request budget reached. Please try again later.",
    "request_id": "generated-request-id"
  }
}
```

| HTTP status | Meaning |
| --- | --- |
| 401 | Missing/incorrect backend token |
| 404 | Unknown selected policy |
| 413 | Request body exceeds 16 KB |
| 422 | Invalid/oversized question or unsupported fields |
| 429 | Request budget reached; respect `Retry-After` |
| 503 | Worker busy; respect `Retry-After` |
| 502 | Provider failure, incomplete answer or failed citation validation |
| 500 | Unexpected internal failure; details stay private |

Do not automatically retry generation requests: a provider request can have
completed and been billed even if the browser timed out. Citation validation
intentionally rejects changed/invented quotations; cheap models can still
produce these. Display a useful error and let the user decide whether to retry.

## Next.js server route for your portfolio

Put these **server-only** settings in your portfolio's `.env.local`, and later
in the Vercel project's environment settings:

```dotenv
RAG_BACKEND_URL=http://127.0.0.1:8000
RAG_BACKEND_API_KEY=<copy the backend token from this Python project's .env>
```

The localhost URL is only for local Next.js development on this computer. A
deployed Vercel function cannot reach your laptop at `127.0.0.1`. Production
needs an HTTPS URL for a hosted Python service and the same private token on
both servers. Configure Preview/Production environments as needed and redeploy
the portfolio after environment changes.

Copy the following into `app/api/insurance-chat/route.ts` in the portfolio
(or `src/app/...` if that project uses `src`). GET returns policy choices; POST
asks the assistant. It uses the default Node.js runtime and no additional SDK.

```typescript
export const maxDuration = 120;

function failure(message: string, status: number) {
  return Response.json({ error: { message } }, {
    status, headers: { "Cache-Control": "no-store" },
  });
}

async function forward(path: string, body?: unknown) {
  const base = process.env.RAG_BACKEND_URL;
  const token = process.env.RAG_BACKEND_API_KEY;
  if (!base || !token) return failure("Assistant is not configured.", 503);
  try {
    const url = new URL(path, base);
    if (process.env.NODE_ENV === "production" && url.protocol !== "https:") {
      return failure("Assistant is not configured.", 503);
    }
    const upstream = await fetch(url, {
      method: body === undefined ? "GET" : "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: "no-store",
      signal: AbortSignal.timeout(90_000),
    });
    const headers: Record<string, string> = { "Cache-Control": "no-store" };
    for (const name of ["Retry-After", "X-Request-ID"]) {
      const value = upstream.headers.get(name);
      if (value) headers[name] = value;
    }
    return Response.json(await upstream.json(), {
      status: upstream.status, headers,
    });
  } catch {
    console.error("Insurance backend request failed");
    return failure("Assistant is temporarily unavailable.", 502);
  }
}

export async function GET() {
  return forward("/v1/documents");
}

export async function POST(request: Request) {
  const origin = request.headers.get("origin");
  if (origin && origin !== new URL(request.url).origin) {
    return failure("Origin is not allowed.", 403);
  }
  if (!request.headers.get("content-type")?.includes("application/json")) {
    return failure("Send application/json.", 415);
  }
  try {
    const body = await request.json();
    if (typeof body.message !== "string" || !body.message.trim() ||
        body.message.length > 4000 ||
        (body.document_id !== undefined && body.document_id !== null &&
          typeof body.document_id !== "string")) {
      return failure("Enter a valid question and policy.", 422);
    }
    return forward("/v1/chat", {
      message: body.message,
      ...(body.document_id ? { document_id: body.document_id } : {}),
    });
  } catch {
    return failure("Invalid JSON request.", 422);
  }
}
```

Your chat component then calls **its own Next.js route**, without any secret:

```typescript
const response = await fetch("/api/insurance-chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: userQuestion, document_id: selectedPolicyId }),
});
const data = await response.json();
if (!response.ok) throw new Error(data.error?.message ?? "Unable to answer.");
// Display data.answer and data.citations.
```

This is a normal JSON API, rather than the Vercel AI SDK streaming protocol.
Use ordinary fetch/state handling for this first frontend. No frontend page or
portfolio deployment was changed by the backend work.

## Hosting the Python side

For the existing persistent local Qdrant index, use an always-running Python
service with a persistent disk. Install the pinned requirements, restore the
index and model caches (or run the existing indexing pipeline once on that
host), configure the two private keys, and start Uvicorn with one worker:

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1
```

Place HTTPS ingress in front of the service and set readiness checks to
`/health/ready`. Keep the index/artifact directories on persistent storage.
Restart after indexing updates. Do not copy a live, open local Qdrant directory;
stop the owning process first before backing up/moving it.

Vercel also supports FastAPI deployments; this is not a Python compatibility
restriction. Deploying this backend as stateless functions requires a hosted
Qdrant server/Cloud, external artifact storage (or disabled local artifacts),
appropriate model packaging and shared rate limiting. The current local-index
server setup is not a Vercel deployment configuration.

The default demo budget is 20 accepted requests per rolling minute and 200 per
rolling 24 hours, **per running process**. It resets on restart and is not shared
across replicas. One inference runs at a time; other chat requests get 503 while
health/policy routes remain responsive. Before public release, add per-visitor
protection on the portfolio route and durable/shared budget controls for scaling.
CORS is optional and is not authentication; the server-to-server proxy needs no
CORS setting. If needed, `RAG_ALLOWED_ORIGINS` accepts explicit comma-separated
origins, with no wildcard or trailing slash.

Full local traces are saved under `data/outputs/api/<request_id>/` when
`RAG_API_SAVE_ARTIFACTS=true`. They include question, prepared context, prompt,
usage and validation errors. The public API never exposes these files or full
internal diagnostics. Define retention/access controls when deploying them.

Official references:

- [FastAPI lifespan for loading shared models](https://fastapi.tiangolo.com/advanced/events/)
- [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/)
- [Next.js server-only environment variables](https://nextjs.org/docs/app/guides/environment-variables)
- [Vercel function duration](https://vercel.com/docs/functions/configuring-functions/duration)
- [FastAPI on Vercel](https://vercel.com/docs/frameworks/backend/fastapi)
- [Qdrant local/server modes](https://github.com/qdrant/qdrant-client)

## Verification on 2026-10-15

All **241 tests passed**, with one upstream Starlette/httpx deprecation warning;
`pip check` reported no broken requirements. Tests cover authentication, schema
validation, request/body limits, concurrency and cancellation, resource lifecycle,
error redaction, CORS, private artifact storage and public response projection.

The running HTTP API returned readiness 200, rejected unauthenticated chat with
401 and listed all 11 indexed policies. A live authenticated insurance question
returned HTTP 200 with a validated flood-coverage answer citing PDF page 1.
Its private run ID is `e1fdfb3fccf7442fbab6e63234f43f8d`; estimated OpenAI cost
was $0.00013695. Two earlier live attempts returned 502 because their quoted
text failed validation. Quote-copy instructions were clarified without weakening
validation, and raw model output is now retained in private diagnostics. This is
a functional smoke test, not a production answer-quality benchmark.
