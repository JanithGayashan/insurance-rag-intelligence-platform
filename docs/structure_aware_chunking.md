# Structure-aware chunking

The active implementation is the custom `StructureAwareChunker`. Docling's
`HybridChunker` remains a comparison implementation; selecting it in the local
frontend does not change the indexing strategy.

Run from the repository root with the project virtual environment:

```powershell
python -m src.chunking.structure_aware.pipeline
python -m src.chunking.structure_aware.comparison
```

Both commands discover documents under the active dataset's processed root,
including motor, travel and health categories. Native Docling files are inputs;
no parser, OCR service, embedding model or LLM is called by these commands.

## Preservation and limits

- Consecutive headings are retained as a compound context when Docling assigns
  equal levels. Other explicit levels are respected. Inferred context is marked
  and is not a claim that the document hierarchy has been verified by a human.
- Original numbered/lettered list markers are rendered in chunk text. Their
  original source references remain available.
- The HybridChunker adapter normalizes heading levels on a deep copy and uses
  a Docling serializer that keeps original markers and non-header/footer text
  in all content layers, including extracted text nested inside pictures. Its
  native Markdown table serializer retains cells and supports repeated headers.
  The parser's saved document is unchanged.
- The configured `cl100k_base` limit applies to the final text including heading
  context. A final adapter guard splits oversized native chunks and records the
  reason. This is a project adapter around the genuine native chunker.
- Oversized custom tables first split by rows. Other oversized blocks prefer
  paragraph/sentence boundaries; hard cuts use original character positions,
  never decoding a partial UTF-8 byte sequence. Impossible character budgets
  raise an error instead of corrupting text. Fallbacks and context truncation
  are recorded. Full heading metadata remains even when a prefix is shortened.
- Custom overlap applies only within oversized source blocks. Normal adjacent
  structural chunks are not forced to overlap.

## Saved artifacts

Each processed document receives `structure_chunks.json` for active indexing.
Its `structure_aware_comparison/` folder contains every chunk from both methods:

- `custom_chunks.json`, `custom_chunks.txt`
- `docling_hybrid_chunks.json`, `docling_hybrid_chunks.txt`
- `comparison.json`, `comparison.md`

The comparison records input/output fingerprints, configuration, timing and
automatic quality diagnostics. Previous chunk artifacts are recoverable from
ignored `.chunk_history/` folders. Discovery excludes those archives.

Rejected validation or a failed custom run invalidates active artifacts.
The index loader also checks current validation, source identity and the parsed
document fingerprint so an old `PASS` label cannot override current rejection.
Standalone legacy chunk fixtures without ingestion sidecars remain supported.
An entirely empty authoritative corpus still requires the index pipeline's
explicit `allow_empty_corpus` option before existing vectors can be removed.

## Evaluation and frontend

Open `http://127.0.0.1:3001`, choose **Chunking**, then use the comparison table,
findings and methodology tabs. Select either implementation to browse its actual
chunks, inspect source elements/JSON and download the complete text or JSON.
Viewing saved results runs no parsing, chunking or provider requests.

Diagnostics include source-element coverage, heading/list-marker retention,
missing-word rate, sentence fragmentation, section crossings, token-limit and
Unicode errors, tiny/duplicate chunks, fallback flags and runtime. `PASS` means
the checked preservation/size/provenance invariants passed; sentence/tiny/duplicate
counts still need interpretation. Missing-word rate is a per-element word-multiset
check, not proof of exact full-text preservation. Sentence checks use punctuation
and normalized phrase containment. They can be affected by abbreviations and
formatting. Boundaries are evaluated against the same inferred contexts.

These are automatic diagnostics against parsed Docling content, not a human
annotated chunking benchmark, retrieval metrics or answer quality. Parsing errors
remain outside their scope. Regenerating chunks does not refresh embeddings or
the vector index; run that phase before testing answers against the new version.
The frontend remains excluded by the repository's `/local-frontend/` ignore rule.
