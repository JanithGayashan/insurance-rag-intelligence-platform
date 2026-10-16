"""Assess native PDF text, then add a temporary Tesseract text layer where needed.

The source file is never written. Images and outlined glyphs are candidates, not
proof that a page was scanned. OCR confidence and review flags are diagnostics.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unicodedata


POLICY_VERSION = "conditional_ocr_v1"
DEFAULTS = {"ocr_mode": "auto", "ocr_language": "eng", "ocr_dpi": 300, "ocr_timeout": 60}


def validate_options(mode, language, dpi, timeout):
    if mode not in {"auto", "off"}:
        raise ValueError("ocr_mode must be auto or off")
    if not re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", language):
        raise ValueError("ocr_language must name installed Tesseract language packs, e.g. eng or eng+tam")
    if not isinstance(dpi, int) or isinstance(dpi, bool) or not 72 <= dpi <= 600:
        raise ValueError("ocr_dpi must be an integer between 72 and 600")
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("ocr_timeout must be positive and finite")


def runtime() -> tuple[str | None, dict[str, str], Path | None]:
    """Use explicit settings, the isolated workstation runtime, or system PATH."""
    local = Path(__file__).resolve().parents[3] / "data/.tools/ocr/Library"
    configured = os.environ.get("TESSERACT_CMD")
    command = (shutil.which(configured) if configured else None) or (
        configured if configured else str(local / "bin/tesseract.exe")
        if (local / "bin/tesseract.exe").is_file() else shutil.which("tesseract"))
    env = dict(os.environ)
    if command:
        env["PATH"] = str(Path(command).parent) + os.pathsep + env.get("PATH", "")
    candidates = [Path(env["TESSDATA_PREFIX"])] if env.get("TESSDATA_PREFIX") else [
        local / "share/tessdata", Path(command).parent / "tessdata" if command else Path("/missing"),
        Path("/usr/share/tesseract-ocr/5/tessdata"), Path("/usr/share/tessdata")]
    data = next((path for path in candidates if path.is_dir()), None)
    if data is not None:
        env["TESSDATA_PREFIX"] = str(data)
    return command, env, data


def cache_identity(language: str = "eng") -> dict:
    command, _, data = runtime()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None
    return {"policy": POLICY_VERSION, "helper_sha256": digest(Path(__file__)),
        "tesseract_sha256": digest(Path(command)) if command else None,
        "language_data_sha256": {name: digest(data / f"{name}.traineddata") if data else None
                                 for name in language.split("+")}}


def text_quality(text: str) -> dict:
    visible = [char for char in text if not char.isspace()]
    bad = sum(char == "\ufffd" or unicodedata.category(char) in {"Co", "Cs", "Cc", "Cn"} for char in visible)
    cid = bool(re.search(r"\(cid:\d+\)", text))
    ratio = bad / len(visible) if visible else 0
    # Language-independent checks; short text and unfamiliar words are not errors.
    return {"characters": len(text.strip()), "alphanumeric_characters": sum(char.isalnum() for char in visible),
            "invalid_character_ratio": round(ratio, 4), "cid_placeholders": cid,
            "suspicious": cid or ratio > 0.05}


def assess_page(page, native_text: str) -> dict:
    import pymupdf
    quality = text_quality(native_text)
    native_words = page.get_text("words")
    # get_image_info uses unrotated coordinates; assess and render in display coordinates.
    images = [(pymupdf.Rect(item["bbox"]) * page.rotation_matrix) & page.rect
              for item in page.get_image_info()]
    images = [rect for rect in images if rect.width >= 12 and rect.height >= 12]
    drawings = page.get_drawings()
    small_paths = sum(0 < item["rect"].width < 25 and 0 < item["rect"].height < 25 for item in drawings)
    hidden = sum(len(item.get("chars", ())) for item in page.get_texttrace() if item.get("type") == 3)
    reasons = []
    if quality["suspicious"]:
        reasons.append("unreadable_native_text")
    if not native_text.strip() and (images or drawings):
        reasons.append("no_native_text_with_visible_content")
    if small_paths >= 80:
        reasons.append("possible_outlined_text")
    full = bool(reasons)
    if not full and images:
        reasons.append("check_image_regions_for_missing_text")
    regions = [page.rect] if full else images
    # Avoid recognizing the same overlapping image multiple times.
    merged = []
    for rect in regions:
        remaining = list(merged)
        merged = []
        while remaining:
            existing = remaining.pop()
            if (rect & existing).get_area() > 0:
                rect |= existing
                remaining.extend(merged)
                merged = []
            else:
                merged.append(existing)
        merged.append(rect)
    return {"policy": POLICY_VERSION, "native_quality": quality, "native_words": len(native_words),
        "possible_existing_ocr_characters": hidden, "image_regions": len(images), "small_vector_paths": small_paths,
        "reasons": reasons, "replace_native_text": quality["suspicious"],
        "action": "CANDIDATE" if merged else "NATIVE_TEXT" if native_text.strip() else "NO_TEXT_CONTENT",
        "regions": [{"bbox": list(rect), "status": "PENDING"} for rect in merged],
        "ocr_applied": False, "review_required": quality["suspicious"], "errors": []}


@dataclass
class PreparedPdf:
    document: object
    pages: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    changed: bool = False
    settings: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {**self.settings, "policy": POLICY_VERSION, "pages_assessed": len(self.pages),
            "pages_ocr_applied": [index + 1 for index, page in enumerate(self.pages) if page["ocr_applied"]],
            "pages_needing_review": [index + 1 for index, page in enumerate(self.pages) if page["review_required"]],
            "failed_pages": [index + 1 for index, page in enumerate(self.pages) if page["errors"]]}


def run_tesseract(image: Path, output: Path, *, language: str, dpi: int, timeout: float) -> dict:
    """A text-only searchable PDF preserves Unicode fonts and word positions."""
    command, env, _ = runtime()
    if not command:
        raise RuntimeError("Tesseract is unavailable; install it or set TESSERACT_CMD and TESSDATA_PREFIX")
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    completed = subprocess.run([command, str(image), str(output), "-l", language,
        "--dpi", str(dpi), "--psm", "3", "-c", "textonly_pdf=1", "pdf", "tsv"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False, **flags)
    if completed.returncode:
        raise RuntimeError(f"Tesseract exited {completed.returncode}: {completed.stderr.decode('utf-8', errors='replace')[-1200:]}")
    with output.with_suffix(".tsv").open(encoding="utf-8") as stream:
        words = [row for row in csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
                 if row.get("level") == "5" and any(char.isalnum() for char in row.get("text", ""))]
    confidences = [float(row["conf"]) for row in words if float(row["conf"]) >= 0]
    return {"recognized_words": len(words), "mean_word_confidence":
            round(sum(confidences) / len(confidences), 2) if confidences else None}


def apply_page_ocr(page, assessment: dict, *, language: str, dpi: int, timeout: float):
    import pymupdf
    from PIL import Image, ImageDraw, ImageStat
    native_words = page.get_text("words")
    for region in assessment["regions"]:
        rect = pymupdf.Rect(region["bbox"])
        # Limit raster memory for large-format pages.
        actual_dpi = max(72, min(dpi, int(72 * math.sqrt(20_000_000 / rect.get_area()))))
        pixmap = page.get_pixmap(dpi=actual_dpi, clip=rect, colorspace=pymupdf.csRGB, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        if not assessment["replace_native_text"]:
            draw = ImageDraw.Draw(image)
            for word in native_words:
                box = (pymupdf.Rect(word[:4]) * page.rotation_matrix) & rect
                if box.is_empty:
                    continue
                scale = actual_dpi / 72
                draw.rectangle(((box.x0 - rect.x0) * scale - 1, (box.y0 - rect.y0) * scale - 1,
                                (box.x1 - rect.x0) * scale + 1, (box.y1 - rect.y0) * scale + 1), fill="white")
        if max(ImageStat.Stat(image).stddev) < 1:
            region.update(status="NO_UNCOVERED_CONTENT", recognized_words=0)
            continue
        try:
            with TemporaryDirectory(prefix="insurance-ocr-") as temporary:
                root = Path(temporary)
                image.save(root / "page.png")
                metrics = run_tesseract(root / "page.png", root / "recognized", language=language,
                                         dpi=actual_dpi, timeout=timeout)
                region.update(metrics, dpi=actual_dpi)
                if not metrics["recognized_words"]:
                    region["status"] = "NO_TEXT_DETECTED"
                    # A photo and an unrecognized scan cannot be distinguished with certainty.
                    assessment["review_required"] = True
                    continue
                # Grafted PDF objects may outlive the source document. Read into
                # memory so Windows can remove the temporary file immediately.
                with pymupdf.open(stream=(root / "recognized.pdf").read_bytes(), filetype="pdf") as recognized:
                    if not recognized[0].get_text().strip():
                        raise RuntimeError("Tesseract returned words without a usable PDF text layer")
                    if assessment["replace_native_text"]:
                        # Stage replacement separately. An overlay failure must
                        # not discard the available native text in the working PDF.
                        parent, number = page.parent, page.number
                        with pymupdf.open() as staged:
                            staged.insert_pdf(parent, from_page=number, to_page=number)
                            replacement = staged[0]
                            for block in replacement.get_text("blocks"):
                                if len(block) > 6 and block[6] == 0:
                                    replacement.add_redact_annot(pymupdf.Rect(block[:4]), fill=None)
                            replacement.apply_redactions(images=0, graphics=0, text=0)
                            replacement.show_pdf_page(rect * replacement.derotation_matrix, recognized,
                                                      0, rotate=replacement.rotation, overlay=True)
                            parent.insert_pdf(staged, start_at=number + 1)
                            parent.delete_page(number)
                    else:
                        page.show_pdf_page(rect * page.derotation_matrix, recognized,
                                           0, rotate=page.rotation, overlay=True)
                region["status"] = "OCR_APPLIED"
                assessment["ocr_applied"] = True
                if metrics["mean_word_confidence"] is None or metrics["mean_word_confidence"] < 65:
                    assessment["review_required"] = True
        except (RuntimeError, OSError, subprocess.TimeoutExpired, ValueError) as error:
            region.update(status="FAILED", error=str(error))
            assessment["errors"].append(str(error))
            assessment["review_required"] = True
    assessment["action"] = ("OCR_PARTIAL" if assessment["ocr_applied"] else "OCR_FAILED") if assessment["errors"] else (
        "OCR_APPLIED" if assessment["ocr_applied"] else "NO_ADDITIONAL_TEXT")


@contextmanager
def prepare_pdf(source, *, ocr_mode="auto", ocr_language="eng", ocr_dpi=300, ocr_timeout=60):
    import pymupdf
    validate_options(ocr_mode, ocr_language, ocr_dpi, ocr_timeout)
    with pymupdf.open(source) as document:
        prepared = PreparedPdf(document, settings={"mode": ocr_mode, "language": ocr_language,
            "requested_dpi": ocr_dpi, "timeout_seconds": ocr_timeout, "engine": "Tesseract CLI"})
        for index, page in enumerate(document):
            assessment = assess_page(page, page.get_text("text", sort=True))
            if ocr_mode == "off":
                assessment["action"] = "DISABLED"
            elif assessment["regions"]:
                try:
                    apply_page_ocr(page, assessment, language=ocr_language, dpi=ocr_dpi, timeout=ocr_timeout)
                except (RuntimeError, OSError, ValueError) as error:
                    assessment["errors"].append(str(error))
                    assessment["review_required"] = True
                    assessment["action"] = "OCR_PARTIAL" if assessment["ocr_applied"] else "OCR_FAILED"
                    for region in assessment["regions"]:
                        if region["status"] == "PENDING":
                            region.update(status="FAILED", error=str(error))
            prepared.pages.append(assessment)
            prepared.changed |= assessment["ocr_applied"]
            for error in assessment["errors"]:
                prepared.warnings.append(f"Page {index + 1}: OCR incomplete: {error}")
        if prepared.changed:
            command, env, _ = runtime()
            prepared.settings["engine_version"] = "unavailable"
            if command:
                try:
                    completed = subprocess.run([command, "--version"], env=env, capture_output=True,
                        text=True, timeout=10, check=True,
                        **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
                    prepared.settings["engine_version"] = completed.stdout.splitlines()[0]
                except (OSError, subprocess.SubprocessError, IndexError):
                    pass  # Optional version metadata must not discard successful OCR.
        yield prepared
