# Insurance RAG Intelligence Platform

A production-oriented **Retrieval-Augmented Generation (RAG)** platform for insurance knowledge retrieval, built to explore, implement, and evaluate modern RAG techniques through an end-to-end system.

The project focuses on reliable ingestion of insurance policies and regulatory documents, high-quality retrieval, grounded answer generation, source citation, and systematic evaluation of RAG components.

The active dataset is **Sri Lanka Insurance (SLIC)**. Shared dataset configuration
automatically discovers motor, travel, health, and additional category folders
across parsing, chunking, indexing, retrieval, generation, and the local frontend.
See [dataset configuration and processing commands](docs/datasets.md).
Existing Allianz benchmark results below describe the earlier corpus.

Compare Docling, pdfplumber, PyMuPDF, Unstructured and LlamaParse on the same SLIC
PDF with the [five-parser comparison pipeline](docs/parser_comparison.md), including
saved outputs, per-page diagnostics and reference checks in the local frontend.

## 🎯 Project Goals

- Build an end-to-end RAG pipeline for insurance knowledge retrieval.
- Develop reliable document ingestion and parsing for heterogeneous insurance PDFs.
- Compare chunking, embedding, retrieval, reranking, and query transformation strategies.
- Implement vector, BM25, and hybrid retrieval.
- Generate grounded answers with source citations.
- Evaluate retrieval and generation quality using standard RAG metrics.
- Analyze component-level failures through controlled experiments.
- Build the system using production-oriented engineering practices.

## 🏗️ Project Roadmap

### Phase 1 — Document Ingestion & Parsing ✅

Implemented a document ingestion pipeline for insurance policy and regulatory PDFs.

Current capabilities include:

- Recursive PDF discovery and batch ingestion.
- PDF parsing using **Docling**.
- Preservation of native structured Docling JSON.
- Markdown and plain-text export.
- Document metadata and SHA-256 fingerprint generation.
- Processing-time and parsing-status tracking.
- Automated parsing-quality validation.
- Cross-parser validation using **pdfplumber**.
- Page-level native text-layer detection.
- Digital, mixed, OCR-text-over-image, and scanned-page classification.
- Image-area coverage analysis for scanned-page detection.
- Text completeness comparison using token coverage and character-length metrics.
- Table-detection comparison between Docling and pdfplumber.
- Page- and document-level `PASS`, `REVIEW`, and `FAIL` quality decisions.
- JSON validation reports for individual documents and the complete corpus.

### Phase 2 — Chunking ✅

Implemented:

- Baseline token-based chunking.
- Custom and LlamaIndex recursive chunking.
- Structure-aware chunking using Docling reading order and document hierarchy.
- Preservation of section paths, page numbers, and native source references.
- List-aware formatting and table-preserving chunk boundaries.
- Token-budget fallback splitting for oversized structural elements.
- Parent-child chunking with Docling section parents and paragraph/sentence children.
- Structure-constrained semantic chunking using local BGE embeddings.
- Custom and LlamaIndex semantic breakpoint implementations using the same model.
- Semantic cohesion, boundary separation, text coverage, size, and section retrieval metrics.
- Readable text and JSON comparison outputs for every validated document.
- Batch chunking for validated `PASS` and `REVIEW` documents.
- Per-document chunk artifacts and corpus-level chunking summaries.

Planned work:

- Reviewed insurance retrieval question set.
- Controlled retrieval evaluation across all chunking strategies.

### Phase 3 — Dense Embeddings & Vector Retrieval 🔄

Implemented:

- Loading and validation of structure-aware chunk artifacts.
- Normalized document and query embeddings with `BAAI/bge-small-en-v1.5`.
- Deterministic Qdrant point identifiers derived from chunk IDs.
- Local persistent Qdrant indexing with optional remote Qdrant configuration.
- Cosine-similarity dense retrieval with metadata filters.
- Index completeness, vector-dimension, duplicate-ID, and normalization checks.
- Example insurance-query retrieval reports with complete source metadata.
- **Embedding and indexing pipeline v2** using our custom Python orchestration.
- Persistent SQLite embedding cache keyed by text and full embedding configuration.
- Incremental document/chunk updates, durable checkpoints, and automatic restart recovery.
- Pinned BGE model revision and query/document configuration compatibility checks.
- BGE-tokenizer input validation and lossless, marked fallback splits for oversized chunks.
- Removal of outdated vectors after replacements are stored and the source snapshot is rechecked.

Run the complete local indexing pipeline:

```bash
python -m src.indexing.pipeline
```

Query the persisted Qdrant index:

```bash
python -m src.retrieval.cli "What accidental damage is covered?" --top-k 5
```

Set `QDRANT_URL` and optionally `QDRANT_API_KEY` to use a server or Qdrant
Cloud instead of the local persistent index.

Repeated runs skip unchanged points and embed only uncached text. The default
v2 collection is `insurance_structure_bge_small_en_v1_5_v2`; the previous
collection is preserved. See [the pipeline v2 guide](docs/embedding_pipeline_v2.md)
for cache/recovery behavior, update examples, and operating limits.

Retrieval evaluation now includes 40 source-checked questions across the 11
policies, original-PDF evidence checks, and Hit Rate@K, Recall@K, and MRR@K.
Unfiltered search and search with a known policy selected are reported separately.
Labels remain provisional until a human reviewer approves them.

```bash
python -m src.evaluation.retrieval.cli --include-source-checked
```

See [the retrieval evaluation guide](docs/retrieval_evaluation.md) for the
question dataset, review process, metric definitions, and generated reports.

### LlamaIndex BM25, hybrid retrieval and pooled evaluation

Native LlamaIndex `BM25Retriever` and `QueryFusionRetriever` now reuse the same
structure-aware index records (812 in the earlier Allianz benchmark). Hybrid retrieval combines BGE/Qdrant dense
results with BM25 using reciprocal rank fusion, without an LLM call.

```bash
python -m src.retrieval.cli "Is flood damage covered?" --method bm25
python -m src.retrieval.cli "Is flood damage covered?" --method hybrid
python -m src.evaluation.retrieval.cli --include-source-checked --retriever all
```

The evaluator builds deduplicated pools from all methods and exports a review
file with positive, negative and unjudged states. Precision@5, Hit Rate@5 and
MRR@5 expose unknown judgments through N/A values and bounds. Pooled Recall@5
and evidence coverage are explicitly relative to annotated evidence. Original
positive labels remain provisional until human review; unjudged candidates
are never automatically marked irrelevant.

See [the LlamaIndex retrieval and pooling guide](docs/llamaindex_retrieval.md) for
review/import commands, metric definitions, reproducibility and operating limits.

### LlamaIndex reranking

The optional `hybrid_reranked` method uses native `SentenceTransformerRerank`
with a pinned local MS MARCO MiniLM cross-encoder. It scores the same fixed
hybrid shortlist, preserves full chunk text/provenance and covers oversized
question/passage pairs with temporary scoring windows. Existing retrieval
defaults and stored document embeddings are preserved.

```bash
python -m src.retrieval.cli "Is flood damage covered?" --method hybrid_reranked --top-k 5
python -m src.evaluation.retrieval.cli --retriever hybrid --rerank --include-source-checked
```

The comparison reports annotated evidence coverage, judgment-aware ranking
metrics, paired improvements/regressions and CPU timing. High reranker scores
do not automatically become relevance labels. See [the reranking guide](docs/reranking.md)
for model configuration, input limits, review and actual ranking artifacts.

### Context preparation and grounded answers

Native LlamaIndex `OpenAIResponses` now generates answers from token-budgeted
structure-aware evidence after hybrid retrieval and optional reranking. Claims
require source IDs and supporting verbatim quotes; the application validates
quotations and renders citations with original filenames and PDF pages.

Add only `OPENAI_API_KEY` to your local `.env`, then run:

```bash
python -m src.generation.cli "Under the Allianz Motor Insurance Policy (English), is flood damage included in the listed coverage?"
python -m src.generation.cli "Is flood damage covered?" --dry-run
```

The configurable default is `gpt-5-nano` with minimal reasoning and bounded
context/output. OpenAI schedules its removal for December 11, 2026; the guide
documents the supported `gpt-6-luna` alternative. Each run saves complete context,
prompt, readable answer and JSON diagnostics under `data/outputs/answers/`.
Dry runs require no key and make no API call. Citation checks do not establish
semantic answer correctness; live generation quality still needs evaluation.
See [the answer generation guide](docs/answer_generation.md) and `.env.example`.

Planned work:

- Human approval and expansion of the source-checked insurance question set.
- nDCG evaluation and a held-out test set for subsequent retrieval tuning.
- Embedding-model comparison using the reviewed question set.

### Portfolio backend API

FastAPI now exposes authenticated `POST /v1/chat` and `GET /v1/documents`, plus
public liveness/readiness checks. Models and retrieval resources load once at
startup. A dedicated worker serializes inference for the local Qdrant index;
requests have body/question limits, process-local request budgets, safe errors
and a public response containing only the answer and citations.

```powershell
.\venv\Scripts\python.exe -m src.api.setup
.\venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000/docs`. Your Next.js portfolio should call this API
through its own server route using the separate `RAG_BACKEND_API_KEY`; neither
that token nor the OpenAI key belongs in browser code. See
[the portfolio backend guide](docs/portfolio_backend.md) for the API contract,
a Vercel/Next.js proxy example and the deployment/storage requirements.

### Upcoming Phases

- Query rewriting and multi-query retrieval.
- HyDE and advanced query transformation.
- Advanced context compression and conversation handling.
- Retrieval and generation evaluation.
- RAG observability and failure analysis.
- Hosted deployment, shared request budgets and container packaging.

## 🔍 Parsing Validation Strategy

Document parsing quality is evaluated using three levels:

1. **Automatic validation**
   - Page counts
   - Text availability
   - Character quality
   - Image coverage
   - Table detection
   - Suspicious or missing content

2. **Cross-parser validation**
   - Docling extraction is compared page-by-page against pdfplumber when a usable native PDF text layer exists.

3. **Human validation**
   - Representative `PASS`, `REVIEW`, and `FAIL` pages are manually inspected against the original PDFs to calibrate validation thresholds and identify failure modes not captured by automatic metrics.

pdfplumber is used as an independent validation signal rather than absolute ground truth. Scanned or image-based pages that do not contain a usable native text layer are handled separately because pdfplumber does not perform OCR.

## 🛠️ Tech Stack

### Current

- **Python**
- **Docling**
- **pdfplumber**
- **LlamaIndex** — chunking experiments, retrieval, reranking and OpenAI answer generation
- **Qdrant** — dense vector indexing and metadata filters
- **BGE / Sentence Transformers** — document and query embeddings
- **BM25** — keyword retrieval through LlamaIndex
- **OpenAI** — grounded answer generation through LlamaIndex
- **FastAPI / Uvicorn** — portfolio backend API

### Planned / Upcoming

- **RAGAS**
- **PostgreSQL**
- **Docker**

## 📂 Current Pipeline

```text
Raw Insurance PDFs
        ↓
Document Discovery
        ↓
Docling Parsing
        ↓
Structured JSON / Markdown / Text
        ↓
Parsing Quality Validation
        ↓
Docling ↔ pdfplumber Cross-Validation
        ↓
Page & Document Quality Classification
        ↓
PASS / REVIEW / FAIL
        ↓
Validated Documents
        ↓
Chunking
        ↓
BGE Embeddings + Qdrant Index
        ↓
Dense / BM25 / Hybrid Retrieval (LlamaIndex)
        ↓
Optional Cross-Encoder Reranking
        ├── Pooled Retrieval Evaluation + Evidence Review
        ↓
Token-Budgeted Context + Grounded OpenAI Answer (LlamaIndex)
