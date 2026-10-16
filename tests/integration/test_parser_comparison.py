from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import pytest

from src.configuration.dataset import get_dataset_configuration
from src.ingestion.comparison import pipeline
from src.ingestion.parsers.llamaparse_parser import normalize_response, parse_pdf


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    dataset = replace(get_dataset_configuration(), raw_root=tmp_path / "raw", processed_root=tmp_path / "processed")
    source = dataset.raw_root / "motor/policy.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"test PDF")
    monkeypatch.setattr(pipeline, "get_dataset_configuration", lambda: dataset)
    monkeypatch.setattr(pipeline, "source_profile", lambda path, image_directory=None: [{"page_number": 1, "text": "Coverage", "has_images": False}])
    return source, dataset.processed_root / "motor/policy"


def runner(calls, fail=None):
    def execute(command, **kwargs):
        name, directory = command[3], Path(command[5])
        calls.append(name)
        if name == fail:
            raise subprocess.TimeoutExpired(command, 1)
        data = {"parser_name": name, "success": True, "text": "Coverage", "pages": [{"page_number": 1, "text": "Coverage"}],
                "elements": [], "tables": [], "metadata": {}, "processing_time_seconds": 1, "warnings": [], "error": None}
        (directory / "result.json").write_text(json.dumps(data), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    return execute


def test_comparison_is_cached_and_keeps_active_outputs_unchanged(corpus):
    source, directory = corpus
    directory.mkdir(parents=True)
    active = directory / "document.json"
    active.write_text("active Docling", encoding="utf-8")
    calls = []
    pipeline.compare_document(source, methods=["pdfplumber", "pymupdf"], runner=runner(calls))
    second = pipeline.compare_document(source, methods=["pdfplumber"], runner=runner(calls))
    assert calls == ["pdfplumber", "pymupdf"]
    assert set(second["parsers"]) == {"pdfplumber", "pymupdf"}
    assert second["parsers"]["pdfplumber"]["cached"]
    assert active.read_text(encoding="utf-8") == "active Docling"


def test_failure_does_not_stop_other_parsers_and_is_not_a_quality_score(corpus):
    source, _ = corpus
    report = pipeline.compare_document(source, methods=["pdfplumber", "pymupdf"], runner=runner([], "pdfplumber"))
    assert report["status"] == "COMPLETED_WITH_FAILURES"
    assert report["parsers"]["pdfplumber"]["evaluation"]["diagnostics"] is None
    assert report["parsers"]["pymupdf"]["status"] == "SUCCESS"


def test_source_change_invalidates_parser_cache(corpus):
    source, _ = corpus
    calls = []
    pipeline.compare_document(source, methods=["pdfplumber"], runner=runner(calls))
    source.write_bytes(b"new PDF")
    pipeline.compare_document(source, methods=["pdfplumber"], runner=runner(calls))
    assert len(calls) == 2


def test_native_baseline_is_preserved_before_ocr_and_not_overwritten(corpus):
    source, directory = corpus
    pipeline.compare_document(source, methods=["pymupdf"], runner=runner([]))
    original = (directory / "parsing_comparison/pymupdf/result.json").read_bytes()
    report = pipeline.compare_document(source, methods=["pymupdf"], force=True, runner=runner([]))
    baseline = directory / "parsing_comparison/pymupdf/baseline_result.json"
    assert baseline.read_bytes() == original
    assert report["parsers"]["pymupdf"]["baseline"]["evaluation"]["status"] == "EVALUATED"
    assert "baseline_result.json" in report["parsers"]["pymupdf"]["artifacts"]
    assert list((directory / "parsing_comparison/history/pymupdf").glob("*/manifest.json"))
    pipeline.compare_document(source, methods=["pymupdf"], force=True, runner=runner([]))
    assert baseline.read_bytes() == original


def test_ocr_runtime_changes_invalidate_cache_and_stale_baselines_are_hidden(corpus, monkeypatch):
    from src.ingestion.parsers import conditional_ocr
    source, _ = corpus
    calls = []
    monkeypatch.setattr(conditional_ocr, "cache_identity", lambda language: {"engine": "first"})
    pipeline.compare_document(source, methods=["pdfplumber"], runner=runner(calls))
    monkeypatch.setattr(conditional_ocr, "cache_identity", lambda language: {"engine": "second"})
    report = pipeline.compare_document(source, methods=["pdfplumber"], runner=runner(calls))
    assert calls == ["pdfplumber", "pdfplumber"]
    assert report["parsers"]["pdfplumber"].get("baseline")
    source.write_bytes(b"new PDF")
    report = pipeline.compare_document(source, methods=["pdfplumber"], runner=runner(calls))
    assert not report["parsers"]["pdfplumber"].get("baseline")
    assert "baseline_result.json" not in report["parsers"]["pdfplumber"]["artifacts"]


def test_process_timeout_stops_the_real_worker(tmp_path):
    marker = tmp_path / "must-not-exist"
    code = "import time,pathlib; time.sleep(2); pathlib.Path(" + repr(str(marker)) + ").write_text('still running')"
    with pytest.raises(subprocess.TimeoutExpired):
        pipeline.run_process([sys.executable, "-c", code], timeout=0.1,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert not marker.exists()


def test_wrong_reference_and_outside_source_are_rejected(corpus, tmp_path):
    source, _ = corpus
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"source_sha256": "different"}), encoding="utf-8")
    with pytest.raises(ValueError, match="different PDF version"):
        pipeline.compare_document(source, reference_path=reference)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"PDF")
    with pytest.raises(ValueError, match="active dataset"):
        pipeline.compare_document(outside)


def test_evaluate_only_recalculates_saved_outputs_without_invoking_parsers(corpus, tmp_path):
    source,directory=corpus
    pipeline.compare_document(source,methods=["pdfplumber"],runner=runner([]))
    result_file=directory/"parsing_comparison/pdfplumber/result.json"
    before=result_file.read_bytes()
    reference=tmp_path/"reference.json"
    reference.write_text(json.dumps({"source_sha256":pipeline.sha256(source),
        "review_status":"PENDING_HUMAN_REVIEW","full_pages":[{"page_number":1,"text":"Coverage"}]}),encoding="utf-8")
    report=pipeline.compare_document(source,evaluate_only=True,reference_path=reference,
        runner=lambda *args,**kwargs:pytest.fail("Evaluation must not invoke a parser/cloud call"))
    assert report["parsers"]["pdfplumber"]["evaluation"]["reference"]["word_error_rate"] == 0
    assert report["reference_sha256"] == pipeline.sha256(reference)
    assert result_file.read_bytes() == before
    assert report["evaluation_only"]
    assert list((directory/"parsing_comparison/evaluation_history").glob("*.json"))
    with pytest.raises(ValueError,match="force"):
        pipeline.compare_document(source,evaluate_only=True,force=True)


def test_cloud_raw_response_can_be_renormalized_without_an_api_call(tmp_path, monkeypatch):
    source = tmp_path / "policy.pdf"
    source.write_bytes(b"PDF")
    raw = {"job": {"status": "COMPLETED", "usage": {"credits": 10}},
           "text": {"pages": [{"page_number": 1, "text": "Flood is covered."}]},
           "items": {"pages": [{"page_number": 1, "items": [{"type": "heading", "value": "Coverage"}]}]}}
    native = tmp_path / "native.json"
    native.write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "cloud_request.json").write_text(json.dumps({"source_sha256": pipeline.sha256(source), "tier": "agentic", "version": "test"}), encoding="utf-8")
    import llama_cloud
    monkeypatch.setattr(llama_cloud, "LlamaCloud", lambda **kwargs: pytest.fail("Cloud API must not be called"))
    result = parse_pdf(source, version="test", native_output_path=native)
    assert result.success and result.pages[0].text == "Flood is covered."
    assert result.metadata["usage"] == {"credits": 10}
    with pytest.raises(ValueError, match="failed"):
        normalize_response({"job": {"status": "FAILED"}}, source, "agentic", 1)
