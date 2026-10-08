import json

import pytest

from src.chunking.parent_child.comparison import compare_document
from src.chunking.parent_child.custom import CustomParentChildChunker
from src.chunking.parent_child.llamaindex import LlamaIndexParentChildChunker
from src.chunking.parent_child.models import ParentChildHierarchy


LONG_TEXT = "\n\n".join(
    (
        f"Section {section}. "
        + " ".join(
            f"Coverage sentence {section}-{sentence} explains a policy condition."
            for sentence in range(18)
        )
    )
    for section in range(8)
)


@pytest.mark.parametrize(
    "chunker_class,strategy",
    [
        (CustomParentChildChunker, "parent-child-custom"),
        (LlamaIndexParentChildChunker, "parent-child-llamaindex"),
    ],
)
def test_parent_child_chunkers_build_valid_two_level_hierarchy(
    chunker_class,
    strategy,
):
    hierarchy = chunker_class(
        parent_size=240,
        child_size=80,
        child_overlap=10,
    ).chunk_text(LONG_TEXT, "policy")

    assert len(hierarchy.parents) > 1
    assert len(hierarchy.children) > len(hierarchy.parents)
    assert hierarchy.strategy == strategy
    assert all(parent.level == 0 and not parent.is_leaf for parent in hierarchy.parents)
    assert all(child.level == 1 and child.is_leaf for child in hierarchy.children)
    assert all(parent.token_count <= 240 for parent in hierarchy.parents)
    assert all(child.token_count <= 80 for child in hierarchy.children)

    parent_by_id = {parent.chunk_id: parent for parent in hierarchy.parents}
    for child in hierarchy.children:
        assert child.parent_id in parent_by_id
        assert child.chunk_id in parent_by_id[child.parent_id].children_ids


@pytest.mark.parametrize(
    "chunker_class",
    [CustomParentChildChunker, LlamaIndexParentChildChunker],
)
def test_parent_child_chunking_uses_deterministic_project_ids(chunker_class):
    chunker = chunker_class(parent_size=240, child_size=80, child_overlap=10)
    first = chunker.chunk_text(LONG_TEXT, "policy")
    second = chunker.chunk_text(LONG_TEXT, "policy")

    assert [chunk.chunk_id for chunk in first.chunks] == [
        chunk.chunk_id for chunk in second.chunks
    ]


@pytest.mark.parametrize(
    "chunker_class",
    [CustomParentChildChunker, LlamaIndexParentChildChunker],
)
def test_parent_child_chunkers_validate_configuration_and_input(chunker_class):
    with pytest.raises(ValueError, match="parent_size"):
        chunker_class(parent_size=80, child_size=80)
    with pytest.raises(ValueError, match="child_overlap"):
        chunker_class(parent_size=200, child_size=80, child_overlap=80)

    chunker = chunker_class(parent_size=200, child_size=80, child_overlap=10)
    with pytest.raises(ValueError, match="text"):
        chunker.chunk_text(" ", "policy")
    with pytest.raises(ValueError, match="document_id"):
        chunker.chunk_text("Policy text.", " ")


def test_hierarchy_model_rejects_broken_relationships():
    valid = CustomParentChildChunker(
        parent_size=240,
        child_size=80,
        child_overlap=10,
    ).chunk_text(LONG_TEXT, "policy")
    payload = valid.model_dump()
    payload["parents"][0]["children_ids"].append("missing-child")

    with pytest.raises(ValueError, match="unknown child"):
        ParentChildHierarchy.model_validate(payload)


def test_custom_chunker_enforces_limits_after_recursive_separator_joining():
    text = "\n\n".join("a" for _ in range(501))
    hierarchy = CustomParentChildChunker(
        parent_size=100,
        child_size=30,
        child_overlap=5,
    ).chunk_text(text, "policy")

    assert all(parent.token_count <= 100 for parent in hierarchy.parents)
    assert all(child.token_count <= 30 for child in hierarchy.children)


def test_comparison_writes_complete_hierarchies_and_readable_samples(tmp_path):
    document_path = tmp_path / "policy" / "document.txt"
    document_path.parent.mkdir()
    document_path.write_text(LONG_TEXT, encoding="utf-8")

    result = compare_document(
        document_path,
        parent_size=240,
        child_size=80,
        child_overlap=10,
    )
    output_directory = document_path.parent / "parent_child_comparison"

    assert result["custom"]["orphan_children"] == 0
    assert result["llamaindex"]["orphan_children"] == 0
    assert result["custom"]["missing_child_references"] == 0
    assert result["llamaindex"]["missing_child_references"] == 0
    for name in (
        "custom_hierarchy.json",
        "llamaindex_hierarchy.json",
        "comparison.json",
        "comparison.md",
    ):
        assert (output_directory / name).exists()

    stored = json.loads(
        (output_directory / "llamaindex_hierarchy.json").read_text(
            encoding="utf-8"
        )
    )
    report = (output_directory / "comparison.md").read_text(encoding="utf-8")
    assert stored["parents"][0]["children_ids"]
    assert stored["children"][0]["parent_id"]
    assert "actual hierarchy sample" in report
