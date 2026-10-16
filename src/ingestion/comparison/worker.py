"""Run one optional parser in its own process, so failures do not stop its peers."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from src.configuration.dataset import get_dataset_configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parser", choices=["docling", "pdfplumber", "pymupdf", "unstructured", "llamaparse"])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args()
    dataset = get_dataset_configuration()  # Load .env from the repository root, without printing it.
    # Optional local dependency prefix: no global installation or PATH changes required.
    ocr_bin = dataset.project_root / "data/.tools/ocr/Library/bin"
    if args.parser == "unstructured" and ocr_bin.is_dir():
        os.environ["PATH"] = str(ocr_bin) + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("TESSDATA_PREFIX", str(ocr_bin.parent / "share/tessdata"))
    options = json.loads(args.options)
    args.output.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module(f"src.ingestion.parsers.{args.parser}_parser")
    if args.parser in {"docling", "unstructured", "llamaparse"}:
        options["native_output_path"] = args.output / "native.json"
    result = module.parse_pdf(args.source, **options)
    value = result.to_dict()
    value["source_file"] = args.source.name
    destination = args.output / "result.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)
    (args.output / "text.txt").write_text(result.text, encoding="utf-8")
    if result.markdown is not None:
        (args.output / "document.md").write_text(result.markdown, encoding="utf-8")
    (args.output / "pages.txt").write_text("\n\n".join(
        f"{'=' * 25} PDF PAGE {page.page_number} {'=' * 25}\n{page.text}" for page in result.pages), encoding="utf-8")
    (args.output / "tables.json").write_text(json.dumps(value["tables"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.parser}: {'SUCCESS' if result.success else 'FAILED'}; {len(result.pages)} pages; {len(result.tables)} tables", flush=True)


if __name__ == "__main__":
    main()
