# LlamaIndex BM25, hybrid retrieval and pooled evaluation

The existing Docling structure-aware chunks, BGE vectors and Qdrant collection
are reused. This phase adds native LlamaIndex retrieval orchestration; it does
not regenerate chunks, re-embed documents or generate answers.

## Retrieval implementations

| File | Responsibility |
| --- | --- |
| `src/retrieval/llamaindex.py` | Convert Qdrant payloads into `TextNode` objects and expose dense retrieval through `BaseRetriever` |
| `src/retrieval/bm25.py` | Native `llama_index.retrievers.bm25.BM25Retriever` |
| `src/retrieval/hybrid.py` | Native `QueryFusionRetriever` in `reciprocal_rerank` mode |

Every node carries its original chunk ID, Qdrant point ID and complete payload.
Provenance is excluded from the BM25 tokenized text. It stays in node metadata,
so equal text from different policies remains separate during fusion.

BM25 uses English stop words, the library's English stemmer, and its default
token pattern. Zero-score keyword matches are omitted. Filters select the
candidate corpus before BM25 ranking; statistics are recomputed for that
filtered corpus and the index is cached in memory per filter selection. A
selection with no documents returns no results. The snapshot is loaded from
Qdrant once per suite; create a new suite after ingestion changes. This small
local implementation is not a distributed sparse-index service.

Hybrid search collects `candidate_k` results from each branch (20 by default),
or more if the requested output is larger. LlamaIndex's native RRF gives both
branches equal contribution and uses rank constant 60. Its pinned implementation
uses zero-based ranks: `1 / (60 + rank_index)`. Scores are fusion scores, not
probabilities, cosine similarity or calibrated relevance confidence.

`num_queries=1` disables query generation. An explicit `MockLLM` fills the
framework's LLM slot without looking up a hosted provider. Tests fail if fusion
attempts to call it. Dense and BM25 candidates are computed once, then passed
to the native fusion retriever. Each `retrieve_all` call embeds the query once,
not once per method. Standalone BM25 scoring does not embed the question; the
shared CLI still loads BGE to check compatibility with the collection.

```bash
python -m src.retrieval.cli "Is flood damage covered?" --method bm25 --top-k 5
python -m src.retrieval.cli "Is flood damage covered?" --method hybrid --top-k 5 --candidate-k 20
python -m src.retrieval.cli "Is flood damage covered?" --method hybrid --source-file "motor\Allianz_Motor_Insurance_Policy_(English).pdf"
```

Dense remains the query CLI default. Retrieval returns ranked text, source
files, physical PDF pages, headings and source references for every method.

## Why the evaluation changed

The previous baseline bound positive source quotations to chunks but had no
explicit negatives. A passage absent from those labels may still be relevant.
The new default evaluator therefore separates **positive, negative and unknown**
question/chunk judgments. It never converts pool membership or a low score
into a relevance label.

```bash
python -m src.evaluation.retrieval.cli --include-source-checked --retriever all --output data/evaluation/allianz/llamaindex_comparison
```

The same questions and corpus snapshot are used for all methods. Both unrestricted
search and explicitly selected-policy search are reported separately. Known
policy filters come from query context, not invented from golden evidence.

For each question the pool is the deduplicated union of the top 20 results from
each method in each evaluated context, plus the original positive evidence.
A question/chunk pair is judged once and shared across methods and contexts.
Two deterministic samples per question from outside that pool are also included
for auditing shared misses. They do not modify rankings. This small audit is a
diagnostic, not a statistical estimate of corpus-wide recall.

Source-quotation matches seed positive judgments with their original review
status. All other candidates remain `unjudged`. A named human is required for
`approved` status. `source_checked` is explicitly provisional and only admitted
when `--include-source-checked` is supplied. No external judge is silently called
and no human review is fabricated.

## Artifacts and review

| Output | Purpose |
| --- | --- |
| `retrieval_comparison.json` | Complete rankings, source metadata, scores, judgment status and per-question metrics |
| `retrieval_comparison.md` | Method/context comparison with precision and MRR bounds |
| `pool.json` | Candidate text, provenance, contributing runs/ranks, questions and evidence definitions |
| `judgments.json` | Editable, validated relevance assessments bound to dataset and corpus hashes |
| `pool_review.md` | Readable candidates, prioritizing top-five results; ranks/method identities are omitted from the review display |

Generated artifacts stay local under ignored `data/evaluation`. Review the
candidate and original PDF, then edit that pair in `judgments.json`:

```json
{
  "question_id": "INS-001",
  "chunk_id": "COPY_THE_ACTUAL_CHUNK_ID",
  "text_sha256": "KEEP_THE_GENERATED_HASH",
  "relevance": 1,
  "review_status": "approved",
  "reviewer": "Actual human reviewer name",
  "rationale": "This clause supports the requested fact in the correct policy version.",
  "supporting_quotes": ["An exact quotation from this candidate"],
  "evidence_ids": ["INS-001-E1"]
}
```

For an irrelevant passage use `relevance: 0`, a rationale and empty evidence IDs.
Positive judgments require quotations; the loader checks their normalized text
against the candidate and checks the expected policy/version. Evidence IDs must
belong to the question. An approved judgment needs a human reviewer; keep
assistant assessments `source_checked`. If a judgment conflicts with the
source-established positive evidence, revise and review the question dataset
first instead of silently overwriting it. That changes its fingerprint.

Existing judgments in the output directory are automatically reloaded before
writing new artifacts, so rerunning does not discard completed review. A changed
snapshot requires a fresh output directory or explicitly reviewed migrated
labels. You can also keep a separate reviewed file and rerun:

```bash
python -m src.evaluation.retrieval.cli --include-source-checked --retriever all --judgments data/evaluation/allianz/llamaindex_comparison/judgments.json --output data/evaluation/allianz/llamaindex_comparison
```

Add `--require-judged-top5` to reject publication until every returned top-five
pair is assessed. It still requires human-approved questions/labels for a
reviewed benchmark; the provisional flag does not confer approval. A changed
dataset, corpus or chunk text rejects stale assessments. Duplicate pairs,
missing positive quotations, unknown evidence IDs and contradictory judgments
are also rejected. Extend the same labels when adding a new method and rescore
all methods against the same snapshot.

## Metrics

| Metric | Definition and limits |
| --- | --- |
| Precision@K | Relevant results / K; missing result slots count as zero. Exact only if all returned positions are judged |
| Precision bounds@K | Lower: known relevant / K; upper: (known relevant + unknown returned) / K |
| Judged fraction@K | Assessed / actually returned results, or 1 for an empty result list |
| Hit Rate@K | At least one relevant result; unresolved only when unknowns could change the outcome |
| MRR@K | Reciprocal first relevant rank, zero outside K; unresolved when an earlier unknown could change it |
| Pooled Recall@K | Known relevant retrieved / all accepted relevant judgments for this question; not exhaustive corpus recall |
| Evidence coverage@K | Fraction of annotated required evidence units supported by accepted positives in the results; alternatives count once |
| Complete evidence coverage@K | Whether every annotated required evidence unit is covered |

When unknown judgments can change a metric, the exact value is `null` (N/A in
Markdown), with lower/upper bounds alongside it. Bounds describe missing
judgments, not annotation errors or statistical confidence. Macro averages
remain null if any question is unresolved; the report includes determined
question counts instead of silently averaging an easier subset. By contrast,
evidence coverage and pooled recall are always relative to the current labels.

Example: five relevant results among eight relevant annotated chunks gives
Precision@5 = 1 and Pooled Recall@5 = 5/8. If those passages repeat one fact,
one retrieved passage can give Evidence coverage@5 = 1. Evidence units must
represent the required facts: the existing development dataset often has one
unit per question, including a few compound source excerpts. Further fact-level
annotation and expert review are needed before claiming complete answer coverage.

All pooled runs include cutoff five. Other cutoffs use `--k`; `--pool-depth`
must cover them and `--candidate-k` must cover the pool depth. Review quality,
question diversity, unseen test questions and corpus versioning remain important.
No retrieval metric here establishes generated-answer correctness.

The previous positive-only report remains reproducible through:

```bash
python -m src.evaluation.retrieval.cli --retriever dense --legacy-dense --include-source-checked
```

## References

- [LlamaIndex BM25 retriever](https://developers.llamaindex.ai/python/framework/integrations/retrievers/bm25_retriever/)
- [LlamaIndex reciprocal rank fusion](https://developers.llamaindex.ai/python/framework/integrations/retrievers/reciprocal_rerank_fusion/)
- [NIST pooling terminology](https://trec.nist.gov/howto.html)
- [NIST pooling limitations](https://www.nist.gov/publications/bias-and-limits-pooling-large-collections)
