# Insurance RAG Intelligence Platform

A production-oriented **Retrieval-Augmented Generation (RAG)** platform for insurance knowledge retrieval, built to explore, implement, and evaluate modern RAG techniques through an end-to-end system.

The project focuses on reliable ingestion of insurance policies and regulatory documents, high-quality retrieval, grounded answer generation, source citation, and systematic evaluation of RAG components.

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

### Phase 2 — Chunking 🔄

Planned work:

- Baseline token-based chunking.
- Recursive chunking.
- Structure-aware chunking using Docling document hierarchy.
- Semantic chunking.
- Parent-child / hierarchical chunking.
- Controlled comparison of chunking strategies.

### Upcoming Phases

- Embedding generation and analysis.
- Vector indexing with Qdrant.
- BM25 sparse retrieval.
- Hybrid retrieval.
- Metadata filtering.
- Query rewriting and multi-query retrieval.
- HyDE and advanced query transformation.
- Reranking.
- Context construction and compression.
- Grounded generation with citations.
- Retrieval and generation evaluation.
- RAG observability and failure analysis.
- FastAPI service and containerized deployment.

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

### Planned / Upcoming

- **LlamaIndex**
- **Qdrant**
- **BM25**
- **BGE / Sentence Transformers**
- **RAGAS**
- **FastAPI**
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
Future RAG Pipeline