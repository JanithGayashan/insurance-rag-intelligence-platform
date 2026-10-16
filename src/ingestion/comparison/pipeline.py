from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from filelock import FileLock

from src.configuration.dataset import get_dataset_configuration
from src.ingestion.comparison.evaluation import evaluate_result
from src.ingestion.comparison.standard_metrics import VERSION, load_evaluation_input, validate_reference


PARSERS = ("pdfplumber", "pymupdf", "docling", "unstructured", "llamaparse")
PACKAGES = {"llamaparse": "llama-cloud", **{name: name for name in PARSERS if name != "llamaparse"}}
OCR_PARSERS = {"pdfplumber", "pymupdf"}
ARTIFACTS = {"result.json", "native.json", "text.txt", "pages.txt", "document.md", "tables.json", "evaluation.json"}
BASELINE_ARTIFACTS = {"baseline_result.json", "baseline_pages.txt", "baseline_evaluation.json"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def preserve_previous_output(directory: Path, fingerprint: str):
    """Archive before overwriting; retain the first native-only run as a baseline."""
    result_file, manifest_file = directory / "result.json", directory / "manifest.json"
    if not result_file.is_file() or not manifest_file.is_file():
        return
    result = json.loads(result_file.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    identity = hashlib.sha256((sha256(result_file) + sha256(manifest_file)).encode()).hexdigest()[:24]
    archive = directory.parent / "history" / directory.name / identity
    archive.mkdir(parents=True, exist_ok=True)
    for file in directory.iterdir():
        if file.is_file() and not (archive / file.name).exists():
            shutil.copy2(file, archive / file.name)
    baseline_manifest = directory / "baseline_manifest.json"
    existing_source = json.loads(baseline_manifest.read_text(encoding="utf-8")).get("source_sha256") if baseline_manifest.is_file() else None
    if (directory.name in OCR_PARSERS and result.get("success") and not result.get("metadata", {}).get("ocr")
        and manifest.get("source_sha256") == fingerprint and existing_source != fingerprint):
        for filename in ("result.json", "pages.txt", "evaluation.json", "manifest.json"):
            source = directory / filename
            if source.is_file():
                shutil.copy2(source, directory / f"baseline_{filename}")


def source_profile(source: Path, image_directory: Path | None = None) -> list[dict]:
    import pymupdf
    with pymupdf.open(source) as document:
        if image_directory is not None:
            image_directory.mkdir(parents=True, exist_ok=True)
            for index, page in enumerate(document):
                image_path = image_directory / f"page-{index + 1}.png"
                if not image_path.is_file():
                    image_path.write_bytes(page.get_pixmap(matrix=pymupdf.Matrix(1.7, 1.7)).tobytes("png"))
        return [{"page_number": index + 1, "text": page.get_text("text", sort=True),
                 "has_images": bool(page.get_images()), "width": page.rect.width,
                 "height": page.rect.height,"rotation":page.rotation} for index, page in enumerate(document)]


def run_process(command, *, timeout, check=False, **kwargs):
    """Terminate the whole parser process tree on timeout or cancellation."""
    process = subprocess.Popen(command, **kwargs, **(
        {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}))
    try:
        return subprocess.CompletedProcess(command, process.wait(timeout=timeout))
    except BaseException:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        raise


def compare_document(source: Path, *, methods=PARSERS, force=False, timeout=1200,
                     reference_path: Path | None = None, options: dict | None = None, runner=run_process,
                     evaluate_only=False) -> dict:
    if evaluate_only and force:
        raise ValueError("evaluate_only cannot be combined with force")
    dataset = get_dataset_configuration()
    source = source.resolve()
    if not source.is_file() or not source.is_relative_to(dataset.raw_root.resolve()) or source.suffix.lower() != ".pdf":
        raise ValueError("Select a PDF within the active dataset's raw directory")
    relative = source.relative_to(dataset.raw_root)
    destination = dataset.processed_root / relative.with_suffix("") / "parsing_comparison"
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / ".comparison.lock"), timeout=0):
        fingerprint = sha256(source)
        profile = source_profile(source, destination / "source_pages" / fingerprint)
        reference_path = reference_path or dataset.project_root / "evaluation/parsing" / dataset.dataset_id / f"{fingerprint}.json"
        reference_bytes=reference_path.read_bytes() if reference_path.is_file() else None
        reference = json.loads(reference_bytes.decode("utf-8-sig")) if reference_bytes else None
        if reference and reference.get("source_sha256") != fingerprint:
            raise ValueError("Reference annotations belong to a different PDF version")
        if reference:
            identifiers = set()
            for check in reference.get("checks", []):
                if (check.get("kind") not in {"passage", "heading", "table_cell"}
                    or check.get("page_number") not in range(1, len(profile) + 1)
                    or not isinstance(check.get("text"), str) or not check["text"].strip()
                    or not check.get("id") or check["id"] in identifiers):
                    raise ValueError("Reference checks require unique IDs, valid PDF pages, kinds and nonempty text")
                identifiers.add(check["id"])
            numbers = set()
            for page in reference.get("full_pages", []):
                if (page.get("page_number") not in range(1, len(profile) + 1)
                    or page["page_number"] in numbers or not isinstance(page.get("text"), str)
                    or not page["text"].strip()):
                    raise ValueError("Full reference pages require unique valid PDF pages and nonempty transcriptions")
                numbers.add(page["page_number"])
        validate_reference(reference,len(profile))
        report_path = destination / "comparison.json"
        report = {"schema_version": 1, "dataset_id": dataset.dataset_id, "active_parser": "docling",
            "source_file": relative.as_posix(), "source_sha256": fingerprint, "page_count": len(profile),
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "RUNNING", "parsers": {},
            "evaluation_only":evaluate_only,"evaluation_version":VERSION,
            "reference_sha256":hashlib.sha256(reference_bytes).hexdigest() if reference else None,
            "evaluation_signature":{"code":{file:sha256(Path(__file__).with_name(file)) for file in ("evaluation.py","standard_metrics.py")},
                                    "libraries":{name:version(name) for name in ("apted","pycocotools","rapidfuzz")}},
            "reference_annotations":reference,
            "reference_status": (reference or {}).get("review_status", "NOT_PROVIDED"),
            "limitations": ["Native-text agreement is a diagnostic, not extraction accuracy: the PDF text layer can be incomplete or out of reading order.",
                "Counts of text, headings and tables do not prove correctness. Repeated lines can be legitimate headers or footers.",
                "Reference checks cover selected annotated passages, headings and table cells only; they do not measure whole-document accuracy.",
                "WER and CER require complete reviewed page transcriptions; unavailable scores remain null.",
                "Timings are single-run wall times, including initialization. Local and hosted parsers use different OCR/layout capabilities.",
                "Conditional OCR uses heuristic text-quality, image-region and outlined-glyph checks; it cannot prove that all text was recovered. Inspect review flags and OCR failures.",
                "Adding an OCR text layer does not add semantic heading labels or general scanned-table structure recognition to PyMuPDF/pdfplumber.",
                "Alternative outputs are comparison artifacts. Active parsing, chunks and vectors remain unchanged."]}
        if report_path.is_file():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if previous.get("source_sha256") == fingerprint:
                report["parsers"] = previous.get("parsers", {})
            if (previous.get("reference_sha256"),previous.get("evaluation_signature")) != (report["reference_sha256"],report["evaluation_signature"]):
                history=destination/"evaluation_history"
                history.mkdir(exist_ok=True)
                snapshot=history/(sha256(report_path)+".json")
                if not snapshot.exists():
                    shutil.copy2(report_path,snapshot)
        if evaluate_only:
            if not report["parsers"]:
                raise ValueError("No saved parser outputs for this PDF version; run the comparison first")
            methods=()
        options = options or {"unstructured": {"strategy": "hi_res", "languages": ["eng"]},
                              "llamaparse": {"tier": "agentic", "version": "latest"}}
        for name in methods:
            if name not in PARSERS:
                raise ValueError(f"Unsupported parser {name}")
            directory = destination / name
            directory.mkdir(exist_ok=True)
            settings = options.get(name, {})
            if name in OCR_PARSERS:
                from src.ingestion.parsers.conditional_ocr import DEFAULTS
                settings = {**DEFAULTS, **settings}
            try:
                package_version = version(PACKAGES[name])
            except PackageNotFoundError:
                package_version = "NOT_INSTALLED"
            signature = {"source_sha256": fingerprint, "package_version": package_version,
                "adapter_sha256": sha256(dataset.project_root / f"src/ingestion/parsers/{name}_parser.py"),
                "worker_sha256": sha256(Path(__file__).with_name("worker.py")), "settings": settings}
            if name in OCR_PARSERS:
                from src.ingestion.parsers.conditional_ocr import cache_identity
                signature["ocr"] = cache_identity(settings["ocr_language"])
                signature["ocr"]["pymupdf_version"] = version("pymupdf")
                signature["ocr"]["pillow_version"] = version("pillow")
            saved_manifest = directory / "manifest.json"
            result_file = directory / "result.json"
            cached = not force and saved_manifest.is_file() and result_file.is_file() and json.loads(saved_manifest.read_text(encoding="utf-8")) == signature
            if not cached:
                preserve_previous_output(directory, fingerprint)
            entry = {"parser": name, "status": "RUNNING", "cached": cached, "configuration": signature}
            report["parsers"][name] = entry
            write_json(report_path, report)
            if not cached:
                print(f"Running {name} on {relative.as_posix()}", flush=True)
                started = time.perf_counter()
                try:
                    # Files and paths are arguments, never shell code. Logs stay in this ignored artifact folder.
                    with (directory / "run.log").open("w", encoding="utf-8") as log:
                        completed = runner([sys.executable, "-m", "src.ingestion.comparison.worker", name,
                            str(source), str(directory), "--options", json.dumps(settings)], cwd=dataset.project_root,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, stdout=log, stderr=subprocess.STDOUT,
                            timeout=timeout, check=False)
                    if completed.returncode or not result_file.is_file():
                        raise RuntimeError(f"Parser process failed (exit {completed.returncode}); see saved run.log")
                except (subprocess.TimeoutExpired, RuntimeError, OSError) as error:
                    write_json(result_file, {"parser_name": name, "source_file": relative.as_posix(), "success": False,
                        "text": "", "markdown": None, "pages": [], "elements": [], "tables": [], "metadata": {},
                        "warnings": [], "error": str(error), "processing_time_seconds": time.perf_counter() - started})
                write_json(saved_manifest, signature)
            result = load_evaluation_input(directory)
            evaluation = evaluate_result(result, profile, reference)
            write_json(directory / "evaluation.json", evaluation)
            entry.update(status="SUCCESS" if result["success"] else "FAILED", error=result.get("error"),
                processing_time_seconds=result["processing_time_seconds"], evaluation=evaluation,
                warnings=result.get("warnings", []), usage=result.get("metadata", {}).get("usage"),
                ocr=result.get("metadata", {}).get("ocr"),
                artifacts=[file.name for file in directory.iterdir() if file.name in (
                    ARTIFACTS
                    if result["success"] else {"result.json", "evaluation.json"})])
            write_json(report_path, report)
            print(f"{name}: {entry['status']}; {result['processing_time_seconds']:.2f}s; cached={cached}", flush=True)
        if sha256(source) != fingerprint:
            raise RuntimeError("Source PDF changed during comparison; report cannot be considered reproducible")
        # Reference annotations can be reviewed after a run, without calling a parser again.
        for name, entry in report["parsers"].items():
            result_file = destination / name / "result.json"
            if result_file.is_file() and entry.get("status") in {"SUCCESS", "FAILED"}:
                entry["evaluation"] = evaluate_result(load_evaluation_input(destination/name), profile, reference)
                write_json(destination / name / "evaluation.json", entry["evaluation"])
                baseline_manifest = destination / name / "baseline_manifest.json"
                baseline_file = destination / name / "baseline_result.json"
                if (baseline_file.is_file() and baseline_manifest.is_file()
                    and json.loads(baseline_manifest.read_text(encoding="utf-8")).get("source_sha256") == fingerprint):
                    baseline_result = load_evaluation_input(destination/name,"baseline_result.json")
                    baseline_evaluation = evaluate_result(baseline_result, profile, reference)
                    write_json(destination / name / "baseline_evaluation.json", baseline_evaluation)
                    entry["baseline"] = {"label": "Native text only, before conditional OCR",
                        "processing_time_seconds": baseline_result["processing_time_seconds"], "evaluation": baseline_evaluation}
                    entry["artifacts"] = sorted(set(entry.get("artifacts", [])) | {
                        file for file in BASELINE_ARTIFACTS if (destination / name / file).is_file()})
        report["status"] = "COMPLETED" if all(row["status"] == "SUCCESS" for row in report["parsers"].values()) else "COMPLETED_WITH_FAILURES"
        write_json(report_path, report)
        return report


def main():
    dataset = get_dataset_configuration()
    parser = argparse.ArgumentParser(description="Compare five PDF parsers without changing the active Docling pipeline.")
    parser.add_argument("--source", type=Path, help="Relative path within the active dataset's raw root; defaults to all its PDFs")
    parser.add_argument("--methods", nargs="+", choices=PARSERS, default=PARSERS)
    parser.add_argument("--force", action="store_true", help="Explicitly rerun selected methods, including cloud API work")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--evaluate-only",action="store_true",help="Reevaluate all saved outputs without invoking any parser or cloud API")
    parser.add_argument("--ocr-mode", choices=["auto", "off"], default="auto",
                        help="Conditional OCR for PyMuPDF/pdfplumber; off is for controlled native-only comparisons")
    parser.add_argument("--ocr-language", default="eng", help="Installed Tesseract language packs, e.g. eng+tam")
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--ocr-timeout", type=float, default=60, help="Seconds per OCR region")
    parser.add_argument("--llamaparse-tier", choices=["cost_effective", "agentic", "agentic_plus"], default="agentic")
    parser.add_argument("--llamaparse-version", default="latest")
    args = parser.parse_args()
    if args.source:
        sources = [dataset.raw_root / args.source]
    else:
        from src.ingestion.pipeline import discover_pdfs
        sources = discover_pdfs(dataset.raw_root)
    for source in sources:
        result = compare_document(source, methods=args.methods, force=args.force, timeout=args.timeout,
            reference_path=args.reference,evaluate_only=args.evaluate_only,options={"unstructured": {"strategy": "hi_res", "languages": ["eng"]},
                "llamaparse": {"tier": args.llamaparse_tier, "version": args.llamaparse_version},
                **{name: {"ocr_mode": args.ocr_mode, "ocr_language": args.ocr_language,
                          "ocr_dpi": args.ocr_dpi, "ocr_timeout": args.ocr_timeout} for name in OCR_PARSERS}})
        print(f"Comparison: {result['status']}; {result['source_file']}", flush=True)


if __name__ == "__main__":
    main()
