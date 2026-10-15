# Insurance retrieval evaluation

The default CLI now compares dense, BM25 and hybrid retrieval with judgment-aware
pooling. See [the LlamaIndex retrieval guide](llamaindex_retrieval.md) for current
commands, Precision@5, missing-judgment bounds and review artifacts. The original
positive-only baseline described below can be reproduced with `--retriever dense
--legacy-dense`; its recall is relative to the annotated chunks, not exhaustive
corpus relevance. Existing numeric snapshots remain historical development data.

This phase evaluates the existing dense retriever against a fixed question set.
It does not regenerate the corpus, change chunking, add BM25, or generate answers.

## Question set

`evaluation/datasets/allianz_questions_v1.json` contains 40 questions covering all
11 current Allianz policies. Categories include coverage, exclusions, claims,
limits, eligibility, waiting periods, cancellation, geography, complaints, and
policy conditions.

Each question records:

- a stable question ID, category, and reference answer;
- the source PDF path and SHA-256 document version;
- one-based physical PDF page numbers and supporting quotations;
- optional alternative passages supporting the same evidence unit;
- explicit policy context for the separately evaluated policy-scoped scenario;
- review status, reviewer, and review notes.

These are assistant-authored development questions. All cited excerpts were
checked against original PDF text and source chunk text. Tables and other
layout-sensitive examples were also inspected as rendered PDF pages. This is
source verification; it is not human domain-expert approval. Questions start as
`source_checked`, with no human reviewer assigned.

The labels were prepared from the source documents, independently of retrieval
rankings. Review of initial misses identified equivalent company-cancellation
refund wording in Fire Dwelling House Condition 10 on PDF page 6. It is included
as an alternative to the primary passage on page 3. The question is about
company cancellation; policyholder cancellation has different conditions.

## Run the provisional baseline

First build the current v2 index, then run:

```bash
python -m src.indexing.pipeline
python -m src.evaluation.retrieval.cli --include-source-checked
```

The evaluator checks original PDFs under `data/raw/allianz`, opens the existing
v2 Qdrant collection, and runs both scenarios:

1. `unfiltered`: search all indexed policies using the question as written.
2. `policy_scoped`: search with the explicit policy filter supplied in the
   question's query context, as if a user had already selected their policy.

The second scenario assumes knowledge of the policy and is not a fair substitute
for unrestricted search. Its metrics are always reported separately. Gold
evidence is never used to invent a filter for the unrestricted run.

```bash
python -m src.evaluation.retrieval.cli --include-source-checked --mode unfiltered
python -m src.evaluation.retrieval.cli --include-source-checked --k 1 3 5 10
```

`QDRANT_URL` and `QDRANT_API_KEY` optionally select the remote collection, just
as in indexing and retrieval. Evaluation never writes or deletes index points.

## Metrics

| Metric | Definition |
| --- | --- |
| Hit Rate@K | Fraction of questions having at least one labeled relevant chunk in the first K results. |
| Recall@K | Per-question fraction of annotated relevant chunk IDs retrieved, averaged equally across questions. |
| MRR@K | Mean reciprocal rank of the first relevant chunk, with zero when none appears within K. |
| Evidence coverage@K | Fraction of required evidence units covered; multiple chunks or alternative passages supporting the same unit count once. |

Results are computed at K=1,3,5,10 by default. Empty search results score zero.
Duplicate retrieved chunk IDs are rejected, rather than inflating recall.

Recall is defined over annotated chunks, not every potentially useful passage
in the corpus. It can differ from evidence coverage when several chunks support
the same fact. A reference answer is provided for review; this phase does not
score generated answers or infer correctness from cosine similarity.

Metric definitions follow the standard query/relevance-label evaluation approach
described by [Qdrant](https://qdrant.tech/documentation/search-evaluation/retrieval-relevance/)
and [Sentence Transformers](https://www.sbert.net/docs/package_reference/sentence_transformer/evaluation.html).

## Generated artifacts

Outputs are under `data/evaluation/allianz` and remain local generated data:

| File | Purpose |
| --- | --- |
| `dense_retrieval_evaluation.json` | Complete rankings, relevance flags, references, per-question metrics, configuration, and corpus/dataset hashes. |
| `dense_retrieval_evaluation.md` | Overall/category metrics and failures at five with the retrieved source sections. |
| `question_review.md` | Readable question/answer/evidence checklist with links to original PDFs. |
| `relevance_labels.json` | Exact evidence-to-chunk bindings for that dataset and corpus snapshot. |

The evaluator verifies source fingerprints, cited pages, and quotes before
searching. Evidence must be present in an indexed chunk with the correct policy
version and page provenance. Rechunking can require label review if a complete
passage no longer fits a single chunk. An index change during evaluation aborts
publication of results. The embedding configuration must match the collection.

## Human review and later comparisons

Open `question_review.md` alongside the cited PDFs. Check the interpretation of
the question, reference answer, policy conditions, quoted evidence, and any other
valid supporting passages. Edit the versioned dataset only after review:

```json
{
  "review_status": "approved",
  "reviewer": "Actual human reviewer name",
  "review_notes": "Verified the answer and evidence against the cited policy."
}
```

Mark ambiguous questions `draft` and rejected questions `excluded`. The normal
command evaluates only approved questions:

```bash
python -m src.evaluation.retrieval.cli
```

It refuses to run when there are no approved questions. The explicit
`--include-source-checked` flag permits a `PROVISIONAL_BASELINE` without silently
turning assistant annotations into human-approved labels. The report identifies
how many questions were included/excluded and their review status.

For BM25, hybrid retrieval, reranking, or model comparisons, keep the same
approved questions, corpus version, label scope, and query-context scenario.
Freeze a separate unseen test set before tuning. Forty hand-authored questions
do not establish production accuracy or represent every real user query.

## Tests

Automated tests cover metric cutoffs and macro averaging, empty/duplicate result
handling, review gates, wrong-policy and stale-source labels, missing evidence,
overlapping/alternative supporting passages, original-PDF path/fingerprint
validation, separate search modes, and concurrent index changes. Synthetic test
fixtures do not require the real policies or a downloaded embedding model.

## Recorded development baseline

The compact measured snapshot is committed at
`evaluation/baselines/dense_bge_small_v2.json`. Complete rankings and source
excerpts remain in the local generated report.

| Scenario | Hit Rate@5 | Recall@5 | MRR@5 | Evidence coverage@5 |
| --- | --- | --- | --- | --- |
| Unfiltered dense search | 0.725 | 0.725 | 0.521667 | 0.725 |
| Known-policy dense search | 1.0 | 0.9875 | 0.741667 | 1.0 |

This provisional run evaluated all 40 source-checked questions against 812 index
records. All 41 primary/alternative evidence locations were verified on original
PDF pages. Eleven unfiltered questions had no labeled evidence in their top five.
Several misses ranked policy introductions above the requested clauses.

Known-policy Recall@5 is below its Hit Rate@5 because one question has two valid
supporting passages: finding either answers that evidence unit, while chunk
recall counts both annotated passages. The policy filter is supplied context;
its higher score does not establish unrestricted retrieval accuracy.

The question set is small and assistant-prepared. Human relevance review and an
unseen test set are still needed before interpreting these as production quality.
