# Embedding and indexing pipeline v2

V2 extends the custom Python indexing pipeline with repeatable incremental
updates and recovery. It consumes the existing `structure_chunks.json` artifacts
and continues to use BAAI/bge-small-en-v1.5, normalized 384-dimensional vectors,
and Qdrant cosine search.

## Run and query

```bash
python -m src.indexing.pipeline
python -m src.retrieval.cli "What accidental damage is covered?" --top-k 5
```

The default collection is `insurance_structure_bge_small_en_v1_5_v2`. V1's
collection is retained. Build v2 before using the updated query command.
`QDRANT_URL` and `QDRANT_API_KEY` optionally select a remote server.

```bash
python -m src.indexing.pipeline --batch-size 32 --state-path data/indexes/embedding_state.sqlite3
python -m src.indexing.pipeline --collection insurance_experiment_v2
```

Use the same state path on subsequent runs. By default it is
`data/processed/allianz/.indexing/state.sqlite3`. Keep this database on persistent
storage. It contains cached vectors, run statuses, batch checkpoints, and the
last committed document registry. It is generated data and is not committed.

## Update behavior

1. Load and validate a complete snapshot of structure-aware artifacts.
2. Count input tokens using the actual embedding tokenizer, including special
   tokens and configured instructions.
3. Prepare deterministic fallback parts for oversized inputs.
4. Check the collection's complete embedding fingerprint and corpus ownership.
5. Compare desired point payloads and vectors with the live index. Skip unchanged
   points. For changed points, reuse cached embeddings or compute missing vectors.
6. Persist each valid embedding batch before writing Qdrant points. Save a
   checkpoint only after Qdrant confirms the upsert.
7. Confirm desired points exist and reload the source snapshot. If sources changed
   or a replacement failed, cancel cleanup.
8. Remove obsolete points, validate the final index, and commit the document registry.

The source path is the logical policy identity; a separate fingerprint tracks
its version. Changes to metadata require a payload update but reuse the text's
embedding. Identical text across policies shares a cached vector while keeping
separate points and provenance. Changing a policy's fingerprint may change all
its chunk IDs; text-level caching still reuses its unchanged passages.

### Authoritative corpus and deletions

Each collection belongs to one processed corpus. All `structure_chunks.json`
files under that root form its authoritative snapshot. Removing a processed
artifact removes its points on the next successful sync. Deleting only a raw
PDF does not remove a still-existing processed artifact. Do not run this sync on
a partial subset of a collection's corpus.

Missing roots and malformed artifacts abort processing. A completely empty
corpus requires an explicit flag:

```bash
python -m src.indexing.pipeline --allow-empty-corpus
```

### Cache and model identity

The cache key includes the text hash, document task, and embedding configuration
fingerprint. The configuration records the resolved model revision, tokenizer
hash, sequence limit, pooling, instructions, dimension, normalization, precision,
and Sentence Transformers/Transformers versions. Invalid cached vectors are
discarded and recomputed.

The default model revision is pinned to
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. BGE's recommended retrieval instruction
is added explicitly to queries; document inputs have no instruction. A query
uses the same pinned configuration as its collection. Overlong questions raise
an error instead of silently losing query text.

Model or configuration changes require a new collection. For custom models,
specify the revision consistently during indexing and querying:

```bash
python -m src.indexing.pipeline --embedding-model MODEL --revision REVISION --collection NEW_COLLECTION
python -m src.retrieval.cli "Question" --embedding-model MODEL --revision REVISION --collection NEW_COLLECTION
```

### Oversized inputs

BGE's limit is 512 model tokens including special tokens. `cl100k_base` chunk
counts cannot enforce this limit. Source chunk artifacts are preserved; fallback
preparation creates smaller index records and marks them with:

- `parent_chunk_id`, section path, source references, and pages;
- actual `embedding_token_count`;
- `embedding_fallback=true` and `embedding_fallback_reason=model_token_limit`;
- part index/count and contiguous character offsets into the original chunk.

Fallback splitting prefers paragraph, sentence, newline, and word boundaries.
Character splitting is the final fallback. Later parts repeat their section
heading for context; offsets exclude that added context. Original source slices
cover the complete original chunk without gaps. An oversized heading raises an
error rather than silently dropping heading text.

## Recovery and operating limits

Rerun the same command after an interruption. Vectors already saved in SQLite
are reused, and current Qdrant contents determine which writes remain necessary.
Checkpoints never cause missing points to be skipped. Failed attempts are stored
as `FAILED`; an unfinished run with the same source/model signature resumes.

One writer per collection is required. A file lock prevents concurrent writers
using the same state database. For a remote deployment, all writers must share
that coordination point or be serialized by an external job system.

The incremental update is not an atomic collection-wide switch: old and new
points may temporarily coexist while replacements are written. Products needing
strict snapshot consistency should build a separate collection and switch a
collection alias after validation. This pipeline still loads source records and
a live index snapshot into memory; streaming and distributed processing are
future scale options.

An explicit `--rebuild` recreates the selected collection using cached vectors.
Use it for repair or controlled experiments, not during live serving:

```bash
python -m src.indexing.pipeline --rebuild
```

## Verification and reports

`dense_indexing_summary.json` and `.md` report cache reuse, unique texts embedded,
point/document changes, resumed checkpoints, token fallbacks, normalization,
index completeness, and example retrieval.

Tests cover unchanged reruns, policy changes/deletion, metadata-only updates,
shared-text caching, interrupted writes and recovery, lost index points,
configuration/corpus mismatches, changed sources during a run, empty-corpus
protection, invalid vectors, and lossless token fallback preparation. Tests use
controlled embedders and local Qdrant; no model download is needed for CI.

Index validation does not establish answer accuracy. Reviewed insurance
questions, Recall@K, MRR, and answer/source evaluation remain separate work.

### Real-corpus acceptance run

Verified against the 11 existing Allianz documents:

| Measurement | Result |
| --- | --- |
| Source structure-aware chunks | 796 |
| Oversized source inputs | 6 |
| Prepared/indexed records | 812 |
| Unique texts embedded on the initial run | 714 |
| Duplicate inputs reused | 98 |
| Missing or unexpected index points | 0 |
| Document embeddings on unchanged rerun | 0 |
| Index writes on unchanged rerun | 0 |

The generated `data/processed/allianz/dense_indexing_v2_verification.json`
contains the initial and unchanged-run measurements. The corpus and generated
reports remain local, following the project's data ignore rules.
