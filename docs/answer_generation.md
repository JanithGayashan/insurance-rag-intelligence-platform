# Context preparation and LlamaIndex answer generation

The answer pipeline reuses the existing **structure-aware chunks**, local BGE
embeddings, Qdrant collection, LlamaIndex hybrid retrieval and optional reranker.
It does not rebuild the index or upload the document corpus to OpenAI. Only the
question, selected evidence, instructions and answer schema are sent for generation.

## Setup and first answer

Add your API key to the existing, Git-ignored project `.env` file:

```dotenv
OPENAI_API_KEY=your_actual_key
```

Do not paste the key into source code or commit it. `.env.example` documents all
optional settings; the API key is the only required addition. Existing local
Qdrant and model caches are reused. A fresh installation needs the existing
ingestion/indexing steps and `python -m pip install -r requirements.txt` first.

From the repository root with its virtual environment activated:

```powershell
python -m src.generation.cli "Under the Allianz Motor Insurance Policy (English), is flood damage included in the listed coverage?"
```

The default is `hybrid_reranked`, retrieving five passages after reranking a
20-candidate hybrid shortlist. To select a specific policy explicitly:

```powershell
python -m src.generation.cli "Is flood damage covered?" --source-file 'motor\Allianz_Motor_Insurance_Policy_(English).pdf'
```

The filter must match the indexed `source_file` value, including path separators.
`--document-id` selects a document fingerprint instead. Filters come from the
caller; evaluation labels are never used to choose a policy for a user question.
Alternative retrieval methods are `dense`, `bm25` and `hybrid`.

To inspect evidence without a key or API charge:

```powershell
python -m src.generation.cli "Is flood damage covered?" --dry-run
```

Dry runs still execute local retrieval/reranking. They return `prepared`, never a
simulated model answer. If no passage has usable text, filename and PDF pages,
live generation returns `insufficient_evidence` without a provider call.

## Model and cost

The default is **`gpt-5-nano`**, selected for its lowest listed standard
text-generation token rates among the models checked on **2026-10-15**:
$0.05 per million input tokens, $0.005 per million cached input tokens and $0.40
per million output tokens. Actual cost also depends on reasoning/output length.
Minimal reasoning, a 6,000-token evidence budget and a 1,800-token output cap keep
requests small. The output cap includes reasoning and JSON, not just visible prose.
These settings require quality evaluation for insurance answers.

**Model lifecycle:** OpenAI marks GPT-5 nano deprecated and schedules removal of
its original snapshot on **2026-12-11**. Before removal, change these optional
settings together:

```dotenv
OPENAI_MODEL=gpt-6-luna
OPENAI_REASONING_EFFORT=none
```

GPT-6 Luna's standard short-context rates checked on the same date are $0.10
per million input tokens and $0.50 per million output tokens. It supports
`none` reasoning; GPT-5 nano requires `minimal`, `low`, `medium` or `high`.
The application supports these two model families and snapshot IDs within them.
Model access and billing depend on the user's OpenAI API account. There is no
automatic switch to a more expensive model.

One generation operation makes **one Responses API request** through native
LlamaIndex `OpenAIResponses.chat`. Automatic retries are disabled by default;
there are no refinement, query rewriting, repair or LLM evaluator calls. API
timeouts can still be billed by the provider; retrying manually is a new request.
Usage and estimated cost come from provider response metadata. Reasoning tokens
are already included in output tokens and are not charged twice in the estimate.
No usage/cost is fabricated for dry runs. Estimates are not billing guarantees.

Official references:

- [GPT-5 nano pricing and capabilities](https://developers.openai.com/api/docs/models/gpt-5-nano)
- [OpenAI model deprecations](https://developers.openai.com/api/docs/deprecations)
- [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)
- [LlamaIndex OpenAI integration](https://developers.llamaindex.ai/python/framework-api-reference/llms/openai/)

## Context construction

1. Validate the question before retrieval or embedding work.
2. Retrieve ranked chunks using the requested method and caller-supplied filters.
3. Remove repeated chunk IDs and identical text within the same document. Identical
   passages in different policies remain separate, with their own provenance.
4. Pack **whole chunks** in retrieval order, counting the serialized evidence
   headers and text with `o200k_base`. Skip oversized passages instead of cutting
   off a condition, table or exclusion; smaller later passages may still fit.
5. Assign source IDs `S1`, `S2`, etc. and include filenames, PDF page numbers,
   document fingerprints and section paths. Record skipped chunks and reasons.

The 6,000-token budget covers serialized evidence, not the entire API input.
Instructions, question, message overhead and JSON schema also consume tokens;
the report includes a full-input estimate, and provider usage is authoritative.
The tokenizer is for generation budgeting; BGE's existing tokenizer and stored
document embeddings remain unchanged. There is no semantic compression or
sentence rewriting of the retrieved text.

## Generation and citation checks

LlamaIndex sends explicit system/user messages and a strict JSON answer schema to
OpenAI. The model returns concise claims, each with source IDs and supporting
verbatim quotes. It is instructed to preserve policy conditions and distinguish
different products, and to abstain when evidence cannot answer the question.
Retrieved text and the question are marked as untrusted data; this prompt does
not provide a formal guarantee against prompt injection.

Application checks reject unknown sources, quotations absent from the cited
passage, inconsistent answer status, malformed JSON and claims without evidence.
Citation markers, filenames and pages are rendered from stored context, never
from model-invented filenames. A failing claim blocks the entire answer; no
automatic paid repair is attempted. Incomplete/refused outputs also return a
clear status without publishing partial claims.
Provider failures produce `provider_error` artifacts with a safe error message;
usage remains unknown if no usage metadata was returned.

**Quotation integrity is not proof of answer accuracy.** A real quote can still
be misinterpreted or fail to support the claim. This implementation does not claim
automated semantic faithfulness, legal correctness or production answer quality.
Representative live answers need review against the original PDFs; existing
retrieval labels also remain provisional until reviewed. The current interface is
single-question generation, without conversation memory or a web chat UI.

## Inspectable output

Each invocation saves a unique run directory under `data/outputs/answers/`:

| File | Contents |
| --- | --- |
| `answer.txt` | Readable question, status, answer and source references. |
| `context.txt` | Complete passages supplied to generation, with source IDs/pages. |
| `answer.json` | Claims, quotes, citations, prepared context, skipped chunks, retrieval results, raw model output, configuration, timing, usage and validation errors. |
| `prompt.json` | System/user messages, strict output schema, configuration and context fingerprint. |

These local artifacts are ignored by Git and can later feed the admin dashboard.
Provider errors are presented without raw request/credential bodies. Credentials
never enter public configuration or artifacts. OpenAI response storage is disabled
(`store=False`); provider data-handling policies still apply.

Tests use deterministic generators and HTTP-mocked **real LlamaIndex/OpenAI SDK**
requests. They do not require keys or make paid API calls. Without a configured
key, local dry runs verify retrieval and context preparation; actual model answer
quality cannot be measured until live requests are made.

Local verification on 2026-10-15: **217 tests passed** (40 generation/context
tests and 177 existing tests), and `pip check` reported no broken requirements.
A fresh policy-filtered Allianz query exercised BGE, Qdrant, hybrid fusion,
reranking and context export with five source passages (1,136 evidence tokens).
Additionally, context preparation over 80 saved reranking results produced no
empty contexts and stayed within the 6,000-token budget (1,859–5,999 tokens).
These were key-free checks; they are not live OpenAI answer-quality results.
