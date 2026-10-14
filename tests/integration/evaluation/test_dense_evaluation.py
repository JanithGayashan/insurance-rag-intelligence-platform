import json

import pytest
from qdrant_client import QdrantClient

from src.evaluation.retrieval.dataset import bind_evidence, corpus_fingerprint, load_dataset, select_questions
from src.evaluation.retrieval.dataset import verify_pdf_sources
from src.evaluation.retrieval.models import QuestionDataset
from src.evaluation.retrieval.models import EvidenceLocation
from src.evaluation.retrieval.runner import evaluate_dense_retrieval
from src.indexing.qdrant_store import QdrantVectorStore
from src.retrieval.dense import DenseRetriever
from tests.unit.indexing.test_qdrant_index import KeywordEmbedder, _record


def make_dataset(root, status="approved"):
    question = {
        "question_id": "Q1", "question": "Is flood damage covered?", "category": "coverage",
        "expected_answer": "Flood damage is covered.", "review_status": status,
        "reviewer": "Test reviewer" if status == "approved" else None,
        "query_filters": {"source_file": "doc-a.pdf"},
        "evidence": [{"evidence_id": "E1", "source_file": "doc-a.pdf", "source_sha256": "doc-a", "pdf_pages": [1], "quotes": ["Flood damage is covered."]}],
    }
    path = root / "questions.json"
    path.write_text(json.dumps({"dataset_id": "test", "version": 1, "description": "synthetic test",
                                "label_scope": "annotated chunks", "questions": [question]}), encoding="utf-8")
    return path


@pytest.fixture
def fixture(tmp_path):
    embedder = KeywordEmbedder()
    store = QdrantVectorStore(QdrantClient(":memory:"), "evaluation", embedder.model_name, embedder.dimension, local_mode=True)
    store.ensure_collection(configuration=embedder.configuration)
    records = [_record(0, "Flood damage is covered."), _record(1, "War damage is excluded."),
               _record(0, "Flood damage is covered.", document="doc-b")]
    store.upsert(records, embedder.embed_documents([record.text for record in records]))
    yield tmp_path, DenseRetriever(embedder, store)
    store.close()


def test_full_evaluation_has_separate_modes_and_source_grounded_labels(fixture):
    root, retriever = fixture
    path = make_dataset(root)
    summary = evaluate_dense_retrieval(path, retriever, root / "reports")
    assert summary["status"] == "REVIEWED_BASELINE"
    assert summary["by_mode"]["policy_scoped"]["metrics"]["hit_rate@5"] == 1
    assert summary["by_mode"]["policy_scoped"]["metrics"]["recall@5"] == 1
    assert summary["questions_evaluated"] == 1
    assert summary["indexed_chunks"] == 3
    assert set(summary["by_mode"]) == {"unfiltered", "policy_scoped"}
    report = json.loads((root / "reports" / "dense_retrieval_evaluation.json").read_text(encoding="utf-8"))
    assert all("doc-b" not in chunk for row in report["queries"] for chunk in row["relevant_chunk_ids"])
    assert (root / "reports" / "question_review.md").is_file()
    assert (root / "reports" / "relevance_labels.json").is_file()


def test_source_checked_labels_require_opt_in_and_remain_provisional(fixture):
    root, retriever = fixture
    path = make_dataset(root, status="source_checked")
    with pytest.raises(ValueError, match="no approved questions"):
        evaluate_dense_retrieval(path, retriever, root / "reports")
    assert not (root / "reports").exists()
    result = evaluate_dense_retrieval(path, retriever, root / "reports", include_source_checked=True)
    assert result["status"] == "PROVISIONAL_BASELINE"


def test_draft_and_excluded_labels_are_not_included_in_provisional_run(tmp_path):
    path = make_dataset(tmp_path, status="draft")
    with pytest.raises(ValueError, match="no approved questions"):
        select_questions(load_dataset(path), include_source_checked=True)


def test_approved_questions_require_named_reviewer(tmp_path):
    value = json.loads(make_dataset(tmp_path).read_text(encoding="utf-8"))
    value["questions"][0]["reviewer"] = None
    with pytest.raises(ValueError, match="named human reviewer"):
        QuestionDataset.model_validate(value)


def test_changed_pdf_fingerprint_rejects_stale_relevance_labels(fixture):
    root, retriever = fixture
    question = load_dataset(make_dataset(root)).questions[0]
    question.evidence[0].source_sha256 = "different-version"
    with pytest.raises(ValueError, match="fingerprint changed"):
        bind_evidence([question], retriever.vector_store.snapshot())


def test_missing_quote_rejects_label_instead_of_selecting_nearest_chunk(fixture):
    root, retriever = fixture
    question = load_dataset(make_dataset(root)).questions[0]
    question.evidence[0].quotes = ["Flood damage is not covered."]
    with pytest.raises(ValueError, match="no chunk contains complete evidence"):
        bind_evidence([question], retriever.vector_store.snapshot())


def test_same_text_in_different_policy_is_not_a_relevant_chunk(fixture):
    root, retriever = fixture
    questions = load_dataset(make_dataset(root)).questions
    labels = bind_evidence(questions, retriever.vector_store.snapshot())
    assert len(labels["Q1"]["E1"]) == 1
    assert labels["Q1"]["E1"][0].startswith("doc-a")


def test_evaluation_does_not_publish_if_index_changes(fixture, monkeypatch):
    root, retriever = fixture
    path = make_dataset(root)
    original_search = retriever.vector_store.search

    def modified(*args, **kwargs):
        hits = original_search(*args, **kwargs)
        record = _record(4, "Hospital cover is available.")
        retriever.vector_store.upsert([record], retriever.embedder.embed_documents([record.text]))
        return hits

    monkeypatch.setattr(retriever.vector_store, "search", modified)
    with pytest.raises(RuntimeError, match="index changed"):
        evaluate_dense_retrieval(path, retriever, root / "reports")
    assert not (root / "reports").exists()


def test_corpus_fingerprint_changes_when_payload_text_changes(fixture):
    root, retriever = fixture
    points = retriever.vector_store.snapshot()
    before = corpus_fingerprint(points)
    next(iter(points.values()))["payload"]["text"] += " Altered policy."
    assert corpus_fingerprint(points) != before


def test_overlapping_chunks_do_not_double_count_evidence_coverage(fixture):
    root, retriever = fixture
    path = make_dataset(root)
    record = _record(2, "Flood damage is covered. Claims require receipts.")
    record.page_numbers = [1]
    retriever.vector_store.upsert([record], retriever.embedder.embed_documents([record.text]))
    summary = evaluate_dense_retrieval(path, retriever, root / "reports", k_values=[1, 5])
    assert summary["by_mode"]["policy_scoped"]["metrics"]["evidence_coverage@1"] == 1
    assert summary["by_mode"]["policy_scoped"]["metrics"]["recall@1"] == 0.5


def test_wrong_source_page_rejects_evidence(fixture):
    root, retriever = fixture
    question = load_dataset(make_dataset(root)).questions[0]
    question.evidence[0].pdf_pages = [99]
    with pytest.raises(ValueError, match="no chunk contains complete evidence"):
        bind_evidence([question], retriever.vector_store.snapshot())


def test_alternative_supporting_passages_are_relevant_to_one_evidence_unit(fixture):
    root, retriever = fixture
    path = make_dataset(root)
    dataset = load_dataset(path)
    record = _record(2, "Flood losses are indemnified.")
    retriever.vector_store.upsert([record], retriever.embedder.embed_documents([record.text]))
    dataset.questions[0].evidence[0].alternatives = [EvidenceLocation(pdf_pages=[3], quotes=["Flood losses are indemnified."])]
    path.write_text(dataset.model_dump_json(), encoding="utf-8")
    labels = bind_evidence(dataset.questions, retriever.vector_store.snapshot())
    assert len(labels["Q1"]["E1"]) == 2
    result = evaluate_dense_retrieval(path, retriever, root / "reports", k_values=[1, 5])
    assert result["by_mode"]["policy_scoped"]["metrics"]["evidence_coverage@1"] == 1
    assert result["by_mode"]["policy_scoped"]["metrics"]["recall@1"] == 0.5


def test_pdf_verification_rejects_path_traversal(tmp_path):
    question = load_dataset(make_dataset(tmp_path)).questions[0]
    question.evidence[0].source_file = "../outside.pdf"
    with pytest.raises(ValueError, match="missing or outside corpus"):
        verify_pdf_sources([question], tmp_path)


def test_pdf_verification_rejects_changed_file_before_parsing(tmp_path, monkeypatch):
    import pypdf
    question = load_dataset(make_dataset(tmp_path)).questions[0]
    (tmp_path / "doc-a.pdf").write_bytes(b"different PDF bytes")
    class Reader:
        pages = []
    monkeypatch.setattr(pypdf, "PdfReader", lambda path: Reader())
    with pytest.raises(ValueError, match="fingerprint changed"):
        verify_pdf_sources([question], tmp_path)
