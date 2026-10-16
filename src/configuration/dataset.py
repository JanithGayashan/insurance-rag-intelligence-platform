from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Mapping

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def source_classification(source_file: str, *, default_business_line: str = "general") -> dict[str, str]:
    """Classify by folder convention, not by guessing document contents.

    Both motor/policy.pdf and general/motor/policy.pdf are supported.
    Other category names are discovered without a fixed category allowlist.
    """
    normalized = source_file.replace("\\", "/")
    source = PurePosixPath(normalized)
    if source.is_absolute() or ".." in source.parts or ":" in normalized:
        raise ValueError("source_file must be a relative path within the dataset")
    parts = source.parts
    folders = parts[:-1]
    business_line = default_business_line
    if len(folders) >= 2 and folders[0].lower() in {"general", "life"}:
        business_line, category = folders[0].lower(), folders[1].lower()
    else:
        category = folders[0].lower() if folders else "uncategorized"
    return {"business_line": business_line, "category": category,
            "category_key": f"{business_line}/{category}"}


@dataclass(frozen=True)
class DatasetConfiguration:
    dataset_id: str
    display_name: str
    short_name: str
    country: str
    default_business_line: str
    project_root: Path
    raw_root: Path
    processed_root: Path
    evaluation_root: Path
    evaluation_dataset: Path
    qdrant_path: Path
    collection: str
    api_output_directory: Path
    answer_output_directory: Path

    @property
    def state_path(self) -> Path:
        return self.processed_root / ".indexing" / "state.sqlite3"

    def source_metadata(self, source_file: str) -> dict[str, str]:
        return {"insurer": self.dataset_id,
                **source_classification(source_file, default_business_line=self.default_business_line)}

    def discover_categories(self) -> list[dict]:
        categories: dict[str, dict] = {}
        for base in (self.raw_root, self.processed_root):
            if not base.is_dir():
                continue
            for folder in sorted(base.iterdir()):
                if not folder.is_dir() or folder.is_symlink() or folder.name.startswith("."):
                    continue
                children = [folder]
                if folder.name.lower() in {"general", "life"}:
                    children = [child for child in folder.iterdir()
                                if child.is_dir() and not child.is_symlink() and not child.name.startswith(".")]
                for category in children:
                    if base == self.processed_root and not any(category.rglob("document.json")):
                        # Reports/previews beside categories are not insurance products.
                        continue
                    relative = category.relative_to(base).as_posix()
                    metadata = source_classification(f"{relative}/placeholder.pdf",
                                                     default_business_line=self.default_business_line)
                    key = metadata["category_key"]
                    entry = categories.setdefault(key, {"key": key, "name": metadata["category"].replace("_", " ").replace("-", " ").title(),
                        **metadata, "raw_documents": 0, "parsed_documents": 0, "source_directories": []})
                    if relative not in entry["source_directories"]:
                        entry["source_directories"].append(relative)
                    files = [file for file in category.rglob("*")
                             if file.is_file() and not file.is_symlink()
                             and file.resolve().is_relative_to(base.resolve())]
                    if base == self.raw_root:
                        entry["raw_documents"] += sum(file.suffix.lower() == ".pdf" for file in files)
                    else:
                        entry["parsed_documents"] += sum(file.name == "document.json" for file in files)
        return sorted(categories.values(), key=lambda item: (item["category"] != "motor", item["key"]))

    def public_metadata(self) -> dict:
        return {"id": self.dataset_id, "display_name": self.display_name, "short_name": self.short_name,
                "country": self.country, "default_business_line": self.default_business_line,
                "categories": self.discover_categories()}

    def as_dict(self) -> dict:
        """Non-secret server configuration, also used by the local frontend."""
        return {**self.public_metadata(), "collection": self.collection,
                **{name: str(getattr(self, name)) for name in (
                    "project_root", "raw_root", "processed_root", "evaluation_root", "evaluation_dataset",
                    "qdrant_path", "state_path", "api_output_directory", "answer_output_directory")}}


def get_dataset_configuration(dataset_id: str | None = None, *, project_root: Path | None = None,
                              environment: Mapping[str, str] | None = None) -> DatasetConfiguration:
    root = Path(project_root or PROJECT_ROOT).resolve()
    if environment is None:
        load_dotenv(root / ".env", override=False)
        environment = os.environ
    manifest = json.loads((root / "config/datasets.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported dataset configuration schema")
    identifier = dataset_id or environment.get("RAG_DATASET") or manifest["default_dataset"]
    if not _IDENTIFIER.fullmatch(identifier) or identifier not in manifest["datasets"]:
        raise ValueError("RAG_DATASET must name a dataset in config/datasets.json")
    entry = manifest["datasets"][identifier]

    def resolve(value: str) -> Path:
        path = Path(value)
        return (root / path).resolve() if not path.is_absolute() else path.resolve()

    def configured_path(field: str, default: str, variable: str | None = None) -> Path:
        return resolve((environment.get(variable) if variable else None) or entry.get(field) or default)

    collection = environment.get("RAG_QDRANT_COLLECTION") or entry.get("collection") or f"insurance_{identifier}_structure_bge_small_en_v1_5_v2"
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,255}", collection):
        raise ValueError("invalid Qdrant collection name")
    return DatasetConfiguration(
        dataset_id=identifier, display_name=entry["display_name"], short_name=entry["short_name"],
        country=entry["country"], default_business_line=entry.get("default_business_line", "general"),
        project_root=root, raw_root=root / "data/raw" / identifier,
        processed_root=root / "data/processed" / identifier,
        evaluation_root=root / "data/evaluation" / identifier,
        evaluation_dataset=root / "evaluation/datasets" / f"{identifier}_questions_v1.json",
        qdrant_path=configured_path("qdrant_directory", f"data/indexes/{identifier}/qdrant", "RAG_QDRANT_PATH"),
        collection=collection,
        api_output_directory=configured_path("api_output_directory", f"data/outputs/{identifier}/api", "RAG_API_OUTPUT_DIRECTORY"),
        answer_output_directory=configured_path("answer_output_directory", f"data/outputs/{identifier}/answers"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the active dataset configuration without loading any models.")
    parser.add_argument("--json", action="store_true", help="Emit non-secret JSON for the local frontend")
    parser.add_argument("--dataset", help="Override RAG_DATASET for this inspection")
    args = parser.parse_args()
    print(json.dumps(get_dataset_configuration(args.dataset).as_dict(), indent=2))


if __name__ == "__main__":
    main()
