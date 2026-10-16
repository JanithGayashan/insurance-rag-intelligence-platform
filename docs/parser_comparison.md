# Five-parser comparison

The active ingestion pipeline continues to use Docling. The comparison runner
invokes the five existing adapters in isolated processes and saves results under
each document's `parsing_comparison/` directory. It does not create chunks,
embeddings, or alter the active native `document.json`.

## Run

Activate the backend environment and run from the repository root:

```powershell
python -m src.ingestion.comparison.pipeline --source motor/motor-english-policy-book-2023.pdf --llamaparse-version 2026-09-29
```

Paths are relative to the active dataset's raw root. Omitting `--source` discovers
all PDFs in that dataset. Select subsets with `--methods pdfplumber pymupdf`, for
example. The runner preserves previously saved methods when running a subset.

### Conditional OCR for PyMuPDF and pdfplumber

Both adapters now default to **extract -> assess -> OCR when needed**. A shared
preflight checks native-text encoding quality, visible image regions, hidden text
and small vector paths that may be outlined letters. It makes page/region
decisions; a heading or page number does not cause the entire page to be skipped.
These are heuristics for finding missing text, not a definitive scanned/digital
classifier or a guarantee that readable native text is correct.

Usable native words are masked **only in the temporary raster sent to OCR** so
image-based or outlined text can be added without repeating existing text.
Tesseract produces a text-only PDF layer. PyMuPDF adds that layer to an in-memory
copy; each adapter then performs its own normal text/word/table extraction.
pdfplumber reads the augmented PDF rather than receiving another parser's text.
The source PDF and its visible images/graphics are never written. Native pages
without OCR candidates use their original text extraction. Scanned-table layout
and semantic heading classification are not supplied by this text fallback.

Blank/fully masked regions avoid OCR. A region with no recognized words, low mean
word confidence, suspicious native text or an OCR error is marked for review.
Photos and unrecognized scans cannot always be distinguished automatically.
Errors/timeouts retain the available native text with explicit warnings and an
`OCR incomplete` badge; parser success alone does not imply complete extraction.
Clearly broken native text is replaced only after a usable OCR layer is obtained.

```powershell
python -m src.ingestion.comparison.pipeline --source motor/motor-english-policy-book-2023.pdf --methods pymupdf pdfplumber
```

Options: `--ocr-mode auto|off` (default auto), `--ocr-language eng` (or installed
packs such as `eng+tam`), `--ocr-dpi 300`, `--ocr-timeout 60` seconds per region.
The adapter functions accept matching `ocr_*` keyword arguments. `off` remains
available for controlled native-only measurements, not as the default pipeline.
`TESSERACT_CMD` and `TESSDATA_PREFIX` can override the local/system runtime.
Raster resolution is bounded to limit memory on large-format pages.

Before replacing results, the runner archives the previous artifacts under
`parsing_comparison/history/<parser>/<run_identity>/`. The earlier native-only
run for the same source fingerprint is also exposed as `baseline_result.json`,
`baseline_pages.txt` and `baseline_evaluation.json`. These files are not silently
replaced on each run. The frontend's **Before OCR** tab displays the actual old
text. Baselines from another PDF fingerprint are not exposed as a comparison.

Each page records decision reasons, candidate regions, masking/recognition
outcomes, OCR word confidence, errors and review flags. The summary records the
engine/version, languages, pages processed and failures. Cache identity includes
shared OCR code, helper libraries, Tesseract binary and discovered language-data
hashes, in addition to the normal adapter/source/settings signature.

Successful and failed attempts are cached against source SHA-256, adapter/worker
code, package version and parser settings. `--force` explicitly reruns selected
methods; the comparison page never launches a parser or paid request. Reference
checks are recalculated from saved results on subsequent comparison runs.

LlamaParse uses the existing `LLAMA_CLOUD_API_KEY` in the backend `.env`. The
initial run uploads the SLIC PDF to LlamaCloud. A raw response is saved before
normalization with a source/settings receipt, so normalization fixes can reuse the
paid response. `--llamaparse-tier` and `--llamaparse-version` are recorded. Pin a
dated version for reproducibility. Provider usage is displayed only when returned;
missing usage does not mean the request was free. Do not commit credentials.

Each parser folder contains the available artifacts:

- `result.json`: normalized pages, text, elements, tables, metadata and runtime.
- `text.txt` and `pages.txt`: full text and page-delimited text.
- `document.md`: Markdown when the parser supplies it.
- `tables.json`: structured table cells with source pages.
- `native.json`: original Docling, Unstructured or LlamaParse output when available.
- `evaluation.json`: automatic diagnostics and reference checks.
- `baseline_*.json` / `baseline_pages.txt`: previous native-only outputs when present.
- `manifest.json` and `run.log`: settings/cache identity and execution diagnostics.

The document-level `comparison.json` records all statuses and measurements. Source page PNG previews live under `source_pages/<source_sha256>/` so the dashboard can display the original page without relying on a browser PDF plugin.
A failed or timed-out parser does not stop the others and is never given a zero
accuracy score. Use its saved log to repair dependencies, then rerun that method.

## Dependencies

Python packages are already pinned in `requirements.txt`. Unstructured runs
locally with `strategy=hi_res`, English OCR and table inference. Poppler,
Tesseract and its English language data must be available. The comparison worker
also recognizes an isolated Windows runtime at `data/.tools/ocr/Library/bin`
and `data/.tools/ocr/Library/share/tessdata`. This entire tools directory is ignored
by Git and no global PATH modification is needed.

For this workstation, Tesseract 5.5.0 was provisioned from conda-forge in that
prefix using portable micromamba. English/OSD models came from the official
`tesseract-ocr/tessdata_fast` revision
`87416418657359cb625c412a48b6e1d6d41c29bd`. On Linux use the system packages for
Poppler and Tesseract plus the English language model. The first high-resolution
run may download Unstructured layout/table models and its spaCy language model.

## Evaluation boundaries

All 29 PDF pages are checked for page coverage, missing/empty pages, extracted
words and characters, replacement characters, repeated long lines, unlocated
elements and tables. Per-page native-text token overlap is an agreement diagnostic
against the PDF text layer, **not ground-truth accuracy**. The image-based back
cover has no usable native text, so its agreement score is unavailable.

Reference files live in `evaluation/parsing/<dataset>/<source_sha256>.json` and
must match the exact PDF fingerprint. The initial SLIC reference includes 26 checks
on PDF pages 4, 10, 16, 20, 27 and 29. These were transcribed from rendered PDF
pages and visually checked by the coding agent; they are explicitly pending human
review. They are not a fully reviewed benchmark.

- Passage checks require expected text within the same PDF page.
- Heading checks also require a heading/title element label. Plain text-block
  parsers do not provide semantic heading classification.
- Table-cell checks require text within an extracted cell on that PDF page.
  They do not validate the relationship between cells, row alignment, spans or
  complete table structure.
- Normalization applies Unicode NFKC, case folding and whitespace collapse.
  Punctuation differences can still cause mismatches; inspect each failed check.

Do not call these limited positive checks whole-document accuracy. Six complete agent-transcribed pages (PDF pages 4, 10, 16, 20, 27 and 29) now provide provisional WER and CER. These rates divide total edit distance by total reference words/characters on those six pages only. They remain pending human review and do not describe the whole document. Without complete transcriptions in `full_pages`, the rates remain null.
Text/table counts do not determine a winner, and single-run wall times include
initialization. OCR, layout and hosted AI capabilities differ between methods.

## Frontend

Open `http://127.0.0.1:3001`, choose SLIC → Motor → Parsing. The summary compares
all five parsers. Use the shared PDF page selector and one or two output panes to
inspect text, structure, cells, Markdown and JSON. Expand per-page diagnostics or
settings and download individual artifacts. Missing capabilities and failures stay
visible. The frontend remains fully ignored by the backend repository.

The scorecard separates saved **Page records** from nonempty **Text pages**, and
shows conditional OCR pages/review flags for PyMuPDF and pdfplumber. Their page
panes show the OCR decision and before/current character counts. Character counts
measure output volume, not accuracy. Other parsers retain their own OCR behavior.

Read-only routes accept only cataloged document fingerprints, supported parser
names and a fixed artifact filename allowlist. Downloads redact credentials and
local absolute paths. There is no arbitrary file browser or parsing trigger.

## Standard parsing evaluation

Reevaluate all saved parser outputs without rerunning a parser, OCR engine or cloud request:

```powershell
python -m src.ingestion.comparison.pipeline --source motor/motor-english-policy-book-2023.pdf --evaluate-only
```

`--evaluate-only` requires saved outputs for the same PDF fingerprint and cannot
be combined with `--force`. Existing parse outputs, timings, active Docling files,
chunks and vectors remain unchanged. Previous comparison reports are archived
under `evaluation_history/` when the reference/metric version changes. The report
records the evaluation version, reference SHA-256 and complete reference
annotations, so every measurement can be traced to its exact labels. Native-only
baselines are reevaluated against the **same** new references as their OCR runs.

The SLIC v2 reference contains six full page transcriptions, 13 headings, 17 list
items, three complete tables and 93 unique reading-order anchors. It covers
definition tables, numeric benefits, exclusions/conditions, nested lists and the
image/outlined-text back cover. Text and regions were annotated from original
Poppler page renders, not generated from any parser's prediction. Text boxes use
manual regions tightened to visible ink; table boxes cover complete tables.
All annotations are **agent checked and pending human review**. Review or correct
the tracked JSON in `evaluation/parsing/slic/` and rerun evaluation; do not change
the review status to human-approved without an actual human review.

Implemented metrics:

- **Text:** WER/CER use NFKC, case and whitespace normalization; WER tokenization
  uses Unicode word tokens. CER therefore measures normalized text, not raw PDF
  typography. Recognized Markdown table delimiters/separator rows are removed
  for text scoring because some adapters place Markdown in their text field;
  original saved outputs remain unchanged. Rates can exceed 100% for many insertions. Text normalized edit
  distance divides character edits by the larger reference/prediction length.
- **Heading/list Precision, Recall, F1:** categories are exhaustively annotated
  on the six pages, including pages with no positives. Same-page one-to-one
  matching uses token F1 >= 0.8 after removing list markers, maximizing the
  number of valid matches before similarity. Duplicate/spurious predictions are
  false positives; missed annotations are false negatives. Consecutive heading
  lines form one block; list groups are split at explicit item markers. This
  matching/granularity policy is project-specific. PyMuPDF/pdfplumber do not
  supply semantic labels: these metrics are unavailable, not fabricated zeros.
- **Reading order:** normalized Levenshtein distance between the expected anchor
  ID sequence and occurrence order in the page text. Exact token matching is
  followed by fuzzy partial ratio >= 85 with token-overlap >= 0.5. Missing and
  repeated anchors are penalized. Anchor coverage and per-page expected/predicted
  IDs are shown. This is a sampled block/cell ordering metric, not proof of every
  within-block word's position.
- **Tables:** TEDS follows the PubTabNet tree-edit formulation using APTED. HTML
  is canonicalized: row groups unified, th/td treated alike, inline formatting
  removed, cell text normalized, rowspan/colspan retained. Rows-only outputs
  assume unit spans. Structure-only TEDS ignores text. Matched TEDS averages
  one-to-one assigned table scores; E2E TEDS divides summed scores by the maximum
  reference/predicted table count on each annotated page before pooling, thereby
  penalizing missing and extra tables. Table assignment maximizes TEDS on the
  same page; per-table scores and representations remain inspectable.
- **Layout:** the official `pycocotools` COCO bbox evaluator computes AP50 and
  mAP across IoU .50:.05:.95, 101 recall points and maxDets=100. Mean best IoU uses
  same-page one-to-one matching with missed references contributing zero.
  Reference and parser boxes are converted to normalized top-left coordinates;
  Docling bottom-left coordinates and Unstructured's saved native pixel-space
  dimensions are handled explicitly. Missing coordinates stay unavailable.
  Predictions without confidence use score=1 and stable saved output order.
  Compare the **same category** across parsers, not aggregates covering different
  capabilities. These are scores on provisional SLIC annotations, not official
  OmniDocBench/DocLayNet benchmark results.
- **Operations:** original single-run wall time, seconds/page, warning counts,
  OCR failures and provider-reported usage. Memory and monetary cost were not
  measured in those saved runs and remain null, not zero. Single attempts do not
  establish production failure rates.

Open **Parsing -> Parsing evaluation**. Tabs show quality scores, category-specific
layout scores, reference transcriptions/region overlays, runtime and methodology.
Enable native-only baseline rows for a comparison under identical references.
Each parser pane has an **Evaluation** tab for page-level errors, reading-order
matches, missed/spurious labels and table scores. The reference boxes on the
original page can be used to review annotation placement. Parsing and frontend
viewing never launch an evaluation or paid parser request automatically.

Metric sources: [PubTabNet TEDS](https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src/metric.py),
[official COCO evaluator](https://github.com/cocodataset/cocoapi),
[OmniDocBench](https://github.com/opendatalab/OmniDocBench),
[OCR-D text evaluation](https://ocr-d.de/en/spec/ocrd_eval.html).

Documentation: [LlamaParse](https://developers.llamaindex.ai/llamaparse/parse/getting_started/),
[pricing](https://developers.llamaindex.ai/llamaparse/general/pricing/),
[Unstructured dependencies](https://docs.unstructured.io/open-source/installation/full-installation),
[Tesseract models](https://github.com/tesseract-ocr/tessdata_fast).
