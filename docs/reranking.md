# LlamaIndex cross-encoder reranking

The new `hybrid_reranked` method scores a fixed set of hybrid candidates with
LlamaIndex's native `SentenceTransformerRerank`. It reuses the current chunks,
vectors, provenance and policy filters. It does not generate answers.

## Retrieval flow

1. Embed the question once and retrieve BGE/Qdrant candidates.
2. Retrieve BM25 candidates once.
3. Fuse the two rankings using the existing native LlamaIndex RRF.
4. Select the first 20 hybrid candidates by default.
5. Cross-encode the question with each candidate passage, reorder by score,
   and return the requested number of original chunks.

Plain hybrid remains available. The retrieval CLI default remains dense;
adding reranking does not silently change callers' previous configurations.

```bash
python -m src.retrieval.cli "Is flood damage covered?" --method hybrid_reranked --top-k 5
```

`--candidate-k` controls candidates per dense/BM25 branch. `--rerank-candidates`
controls the hybrid shortlist scored by the cross-encoder. Reranking requires
`top_k <= rerank_candidates <= candidate_k`. The defaults retrieve 20 per branch,
rerank 20 fused candidates and return five. Fewer candidates are allowed when the filtered corpus
is small. An empty selection returns no results.

## Model and configuration

- Model: `cross-encoder/ms-marco-MiniLM-L6-v2`.
- Revision: `fbf9045f293a58fa68636213c5e0cb8a2de5d45e`.
- Device: CPU by default; configurable with `--reranker-device`.
- Maximum question/passage pair: 512 tokens, including special tokens.
- Temporary passage-window overlap: up to 32 tokens, reduced for small budgets.
- Window aggregation: maximum score; equal scores retain hybrid order.
- Source context: the actual policy source filename, included in each scoring passage.
- Remote model code: disabled.

The pinned LlamaIndex constructor has no revision argument. Our wrapper downloads
an immutable Hugging Face snapshot, then passes its local directory to the native
postprocessor. A configuration fingerprint records the requested model revision,
tokenizer hash, window settings, source context, device and library versions.
The model is loaded once per suite/evaluation run. First use downloads its files.

Custom models require an explicit immutable revision:

```bash
python -m src.retrieval.cli "What is excluded?" --method hybrid_reranked --reranker-model ORGANIZATION/MODEL --reranker-revision FULL_40_CHARACTER_COMMIT_SHA
```

This adapter targets compatible single-score cross-encoders supported by the
pinned `SentenceTransformerRerank` integration. Changing the model name is not
a guarantee that an arbitrary architecture or larger context length is supported.

## Handling long passages

The embedding model's document limit does not guarantee a reranker pair fits:
the question, source context and separator tokens also consume space.

The wrapper counts actual tokenizer inputs before inference. If a pair is too
large, it creates overlapping temporary token windows covering the passage,
repeats the policy context, and scores every window through the native LlamaIndex
postprocessor. It uses the highest window score for that original chunk. The
output still contains the full original chunk text, source file, PDF pages,
headings and source references. Qdrant and the chunk artifacts are never rewritten.

Returned metadata includes original hybrid rank/score, window count, winning
window token range, maximum actual pair length, configuration fingerprint and
`truncated: false`. A question/context leaving no passage budget is rejected.
Non-finite scores and duplicate candidate IDs are rejected. These checks prevent
silent tokenizer truncation; they do not establish semantic correctness of scores.
Maximum-window aggregation can favor long passages with more windows and should
be reconsidered if later evaluation identifies that bias.

## Fair before/after evaluation

```bash
python -m src.evaluation.retrieval.cli --retriever hybrid --rerank --include-source-checked --judgments data/evaluation/allianz/llamaindex_comparison/judgments.json --output data/evaluation/allianz/reranking_comparison
```

Use `--retriever all --rerank` to retain dense and BM25 baselines too. By default,
the plain `all` evaluation still runs the previous three methods without loading
a reranker. The explicit reranking option adds the new method.

For a reranking comparison, `pool_depth == rerank_candidates <= candidate_k`.
This exports and scores the same complete shortlist before and after reranking.
Both rankings receive exactly the same ordered candidate IDs; an invariant check
rejects a comparison that adds or drops candidates. All labels, questions, corpus
versions and query context are shared. A source passage absent from the shortlist
cannot be recovered by reranking.

The original source-checked positives are reused. Candidates promoted by a high
model score remain unknown unless separately assessed. Scores never create
relevance labels. Existing judgments are validated and preserved, and new
top-five pairs enter the review queue. Precision, Hit Rate and MRR keep their
missing-judgment bounds; pooled recall and evidence coverage stay annotation-relative.

The report adds per-method mean/median/p95 seconds and paired evidence-coverage
improvements, regressions and ties. It identifies questions whose annotated
evidence is absent from the shortlist, distinguishing that retrieval limitation
from evidence available for reranking. Per-question records retain the original
candidate IDs, first labeled ranks, original hybrid scores, new model scores,
temporary-window counts and reranking seconds. With the same complete shortlist,
coverage at the full candidate cutoff must remain unchanged; the purpose is to
improve ordering and smaller cutoffs.

Timings sum the actual shared sequential retrieval/fusion components and add
cross-encoder work only to the reranked method. They include first-use BM25 index
work but exclude reranker startup/download time, which is recorded separately.
They are development CPU measurements, not production API latency. No generative
LLM or paid inference API is called. Larger shortlists and more windows cost more.

## Output artifacts

Outputs remain local under `data/evaluation/allianz/reranking_comparison`:

| File | Contents |
| --- | --- |
| `retrieval_comparison.md` | Method/context metrics, timing and paired change summary |
| `retrieval_comparison.json` | Full original text, scores, provenance and per-question diagnostics |
| `ranking_changes.md` | Top-five before/after sources and excerpts for every question and context |
| `pool.json` | Candidate text, contributing rankings and evidence definitions |
| `judgments.json` | Preserved assessments plus unknown candidates |
| `pool_review.md` | Review queue prioritizing all top-five candidates |

A compact measured snapshot is committed in `evaluation/baselines`; source-text
artifacts stay ignored. Forty assistant-prepared development questions do not
replace human relevance review, a held-out test set or generated-answer evaluation.

## Recorded development comparison

The measured snapshot is `evaluation/baselines/llamaindex_reranking_v1.json`:

| Method | Unfiltered annotated evidence coverage@5 | Mean local CPU seconds |
| --- | --- | --- |
| Hybrid | 0.825 | 0.077 |
| Hybrid + cross-encoder | 0.925 | 3.148 |

Four questions improved, none regressed and 36 tied on this coverage metric.
For INS-007, INS-010 and INS-027 the labeled evidence is absent from the candidate
shortlist; reranking cannot recover it. Known-policy coverage@5 remained 1.0 for
both methods. These are fixed-label development observations, not production
accuracy or a controlled hardware performance benchmark.

Across both contexts, the real run scored 1,705 temporary pairs; 138 candidate
pairs needed multiple windows. Maximum observed pair length was 511 tokens.
The existing pool still has 2,176 pairs, with 637 now appearing in some method's
top five; 596 of those remain unjudged. Exact Precision@5 and overall MRR@5 remain
unresolved, with bounds in the report. Model scores did not create any labels.

## Implementation and verification

`src/retrieval/reranking.py` wraps the native postprocessor. `hybrid.py` computes
candidates once and exposes per-call diagnostics without storing mutable timing
state on the suite. Both retrieval and evaluation CLIs expose the method. The
evaluation comparison retains the pool/review workflow introduced previously.

CI tests use the real LlamaIndex postprocessor with a deterministic injected
tokenizer/scorer; they do not download model weights. They check candidate-set
identity, input limits, complete window coverage, tail evidence, original-text
and metadata preservation, stable ties, empty selections, invalid scores,
configuration guards, unknown labels and paired reporting. Actual model inference
is verified separately on the local insurance corpus and recorded in the baseline.

References:
- [LlamaIndex SentenceTransformerRerank](https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/node_postprocessors/)
- [Sentence Transformers cross-encoder usage](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- [Pinned model checkpoint](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/fbf9045f293a58fa68636213c5e0cb8a2de5d45e)
