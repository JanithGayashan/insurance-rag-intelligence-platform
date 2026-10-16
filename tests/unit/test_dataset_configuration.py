import json
from pathlib import Path
import shutil

import pytest

from src.configuration.dataset import PROJECT_ROOT, get_dataset_configuration, source_classification


@pytest.fixture
def dataset_root(tmp_path):
    (tmp_path / "config").mkdir()
    shutil.copyfile(PROJECT_ROOT / "config/datasets.json", tmp_path / "config/datasets.json")
    return tmp_path


def test_dataset_defaults_are_isolated_and_independent_of_cwd(dataset_root, monkeypatch):
    monkeypatch.chdir(dataset_root.parent)
    slic = get_dataset_configuration(project_root=dataset_root, environment={})
    allianz = get_dataset_configuration("allianz", project_root=dataset_root, environment={})
    assert slic.dataset_id == "slic"
    assert slic.raw_root == dataset_root / "data/raw/slic"
    assert slic.processed_root == dataset_root / "data/processed/slic"
    assert slic.evaluation_root == dataset_root / "data/evaluation/slic"
    assert slic.qdrant_path != allianz.qdrant_path
    assert slic.collection != allianz.collection
    assert slic.api_output_directory != allianz.api_output_directory
    assert slic.answer_output_directory != allianz.answer_output_directory
    assert allianz.collection == "insurance_structure_bge_small_en_v1_5_v2"
    assert allianz.evaluation_dataset.name == "allianz_questions_v1.json"


def test_storage_overrides_are_resolved_once_against_project_root(dataset_root):
    dataset = get_dataset_configuration(project_root=dataset_root, environment={
        "RAG_QDRANT_PATH": "custom/index", "RAG_QDRANT_COLLECTION": "slic-test",
        "RAG_API_OUTPUT_DIRECTORY": "custom/answers",
    })
    assert dataset.qdrant_path == dataset_root / "custom/index"
    assert dataset.api_output_directory == dataset_root / "custom/answers"
    assert dataset.collection == "slic-test"


@pytest.mark.parametrize("identifier", ["../allianz", "unknown", "SLIC", "slic/motor"])
def test_unknown_or_unsafe_dataset_fails(dataset_root, identifier):
    with pytest.raises(ValueError, match="RAG_DATASET"):
        get_dataset_configuration(identifier, project_root=dataset_root, environment={})


@pytest.mark.parametrize("source,business,category", [
    ("motor/book.pdf", "general", "motor"),
    (r"general\travel\nested\book.PDF", "general", "travel"),
    ("general/health/book.pdf", "general", "health"),
    ("life/protection/book.pdf", "life", "protection"),
    ("book.pdf", "general", "uncategorized"),
])
def test_source_classification(source, business, category):
    assert source_classification(source) == {
        "business_line": business, "category": category, "category_key": f"{business}/{category}",
    }


@pytest.mark.parametrize("source", ["../../secret.pdf", "D:/secret.pdf", "/secret.pdf"])
def test_source_paths_cannot_escape_dataset(source):
    with pytest.raises(ValueError):
        source_classification(source)


def test_discovery_supports_mixed_layouts_empty_folders_and_new_categories(dataset_root):
    dataset = get_dataset_configuration(project_root=dataset_root, environment={})
    for name in ("motor/book.pdf", "general/travel/nested/book.PDF", "general/health/book.pdf", "marine/book.pdf"):
        file = dataset.raw_root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"test PDF")
    (dataset.raw_root / "life/protection").mkdir(parents=True)
    processed = dataset.processed_root / "general/travel/nested/book/document.json"
    processed.parent.mkdir(parents=True)
    processed.write_text("{}", encoding="utf-8")
    (dataset.processed_root / ".indexing").mkdir()
    preview = dataset.processed_root / "parsing-evaluation-proof" / "scores.png"
    preview.parent.mkdir()
    preview.write_bytes(b"preview")
    other = dataset_root / "data/raw/allianz/pet"
    other.mkdir(parents=True)
    (other / "book.pdf").write_bytes(b"other insurer")
    categories = {item["key"]: item for item in dataset.discover_categories()}
    assert set(categories) == {"general/motor", "general/travel", "general/health", "general/marine", "life/protection"}
    assert categories["general/travel"]["raw_documents"] == 1
    assert categories["general/travel"]["parsed_documents"] == 1
    assert categories["life/protection"]["raw_documents"] == 0
    assert categories["general/motor"]["source_directories"] == ["motor"]
    assert "project_root" not in dataset.public_metadata()
    assert json.loads(json.dumps(dataset.as_dict()))["id"] == "slic"
