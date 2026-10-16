# Shared insurance dataset configuration

`config/datasets.json` is the dataset registry. `src/configuration/dataset.py`
resolves it for every pipeline, the API, and the local frontend. Set the active
dataset in the repository's ignored `.env`:

```dotenv
RAG_DATASET=slic
```

Paths are resolved from the repository root, independent of the terminal's current
directory. Restart running Python processes after changing dataset or storage
settings. Pipeline modules bind their defaults when imported.

## Folder discovery

Put PDFs anywhere underneath category directories:

```text
data/raw/slic/
  motor/policy.pdf
  travel/policy.pdf
  health/policy.pdf
```

An explicit business-line layer is also supported:

```text
data/raw/slic/general/motor/policy.pdf
data/raw/slic/general/travel/nested/policy.PDF
data/raw/slic/general/health/policy.pdf
data/raw/slic/life/protection/policy.pdf
```

Both layouts can coexist. Without a business-line directory, SLIC uses `general`.
New category names are discovered from folders; there is no motor/travel/health
allowlist to edit. Empty category folders appear with zero documents. Files at the
dataset root have `uncategorized` source metadata; use category folders for the
dashboard. Folder names express the classification; the system does not infer
insurance products from PDF contents.

The parser mirrors the relative folder layout under `data/processed/slic/`, with
a directory per PDF containing `document.json`, `document.md`, `document.txt`, and
`metadata.json`. Source metadata includes insurer, business line, category, category
key, and SHA-256 fingerprint. Existing parsing, validation and chunking algorithms
are unchanged. All chunking entry points recursively discover the active dataset's
processed files, including nested categories.

## Run processing

From the repository root, with the backend virtual environment activated:

```powershell
python -m src.configuration.dataset --json
python -m src.ingestion.pipeline
python -m src.ingestion.validator
python -m src.chunking.structure_aware.pipeline
python -m src.indexing.pipeline
```

No category-specific path arguments are needed. Parsing discovers PDFs, validation
checks the mirrored outputs, structure-aware chunking accepts `PASS`/`REVIEW`
documents, and indexing synchronizes all eligible structure-aware chunks.
Re-running indexing uses its existing cache and incremental synchronization.
Stop the API before indexing the same local Qdrant storage directory.

Other experiment entry points use the same processed root:

```powershell
python -m src.chunking.fixed.inspect
python -m src.chunking.fixed.evaluate
python -m src.chunking.structure_aware.comparison
python -m src.chunking.sentence_paragraph.comparison
python -m src.chunking.parent_child.pipeline
python -m src.chunking.semantic.pipeline
```

These experiments are optional; the active retrieval index still uses
`structure_chunks.json`. Explicit directory arguments remain available where the
entry point already supported them.

## Dataset isolation

SLIC defaults:

| Artifact | Location |
| --- | --- |
| Raw PDFs | `data/raw/slic/` |
| Parsed documents and chunk artifacts | `data/processed/slic/` |
| Embedding cache and checkpoints | `data/processed/slic/.indexing/state.sqlite3` |
| Local Qdrant | `data/indexes/slic/qdrant/` |
| Qdrant collection | `insurance_slic_structure_bge_small_en_v1_5_v2` |
| Evaluation questions | `evaluation/datasets/slic_questions_v1.json` |
| Evaluation results | `data/evaluation/slic/reranking_comparison/` |
| API question traces | `data/outputs/slic/api/` |
| CLI answers | `data/outputs/slic/answers/` |

The registry retains Allianz with its original Qdrant path, collection, answer
folders, and question dataset. Switching to SLIC does not delete or rebuild them.
To work on that corpus again, set `RAG_DATASET=allianz` and restart the processes.

Optional `RAG_QDRANT_PATH`, `RAG_QDRANT_COLLECTION`, and
`RAG_API_OUTPUT_DIRECTORY` override the shared defaults for indexing, retrieval,
generation, and the API. Remove old overrides when switching datasets unless the
shared storage is intentional. Remote Qdrant continues to use `QDRANT_URL` and
`QDRANT_API_KEY`, with the active dataset's collection name.

## Evaluation and local frontend

Create and review SLIC-specific questions before running
`python -m src.evaluation.retrieval.cli --rerank`. The CLI refuses a missing question
dataset rather than using Allianz labels. Saved scores are scoped to the active
dataset; an absent report appears as `NOT_RUN` in the dashboard. Historical Allianz
benchmarks in the repository remain historical results.

The ignored `local-frontend/` server invokes the Python configuration resolver;
it does not maintain a second dataset registry. The category selector updates
policies, document inspection, vectors, and saved question lists. Index totals and
evaluation aggregates explicitly describe the entire active dataset. Chat requests
are scoped to the chosen policy. Categories without parsed policies cannot submit
a question. Refresh saved outputs after processing new documents.

The backend exposes authenticated `GET /v1/dataset` metadata and includes its
dataset ID in readiness responses. The frontend blocks chat if the backend was
started with a different dataset. Answer artifacts carry a dataset ID; untagged
legacy runs are not presented as current SLIC traces.

Start FastAPI on port 8000 and run `npm run dev` inside `local-frontend/`, then open
`http://127.0.0.1:3001`. The root `/local-frontend/` ignore rule remains in place.
API credentials stay server-side in the backend `.env`.
