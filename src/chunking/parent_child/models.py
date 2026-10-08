from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HierarchicalChunk(BaseModel):
    """One parent or child chunk and its hierarchy relationships."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    level: Literal[0, 1]
    text: str = Field(min_length=1)
    token_count: int = Field(ge=1)
    page_numbers: list[int] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    is_leaf: bool
    strategy: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_level_relationships(self) -> "HierarchicalChunk":
        if self.level == 0:
            if self.parent_id is not None or self.is_leaf:
                raise ValueError("a parent chunk cannot have a parent and is not a leaf")
        elif self.parent_id is None or not self.is_leaf or self.children_ids:
            raise ValueError(
                "a child chunk must reference a parent, be a leaf, and have no children"
            )
        return self


class ParentChildHierarchy(BaseModel):
    """A validated two-level hierarchy produced for one document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    parent_size: int = Field(gt=0)
    child_size: int = Field(gt=0)
    child_overlap: int = Field(ge=0)
    parents: list[HierarchicalChunk]
    children: list[HierarchicalChunk]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_relationships(self) -> "ParentChildHierarchy":
        if self.parent_size <= self.child_size:
            raise ValueError("parent_size must be greater than child_size")
        if self.child_overlap >= self.child_size:
            raise ValueError("child_overlap must be smaller than child_size")

        parent_ids = [chunk.chunk_id for chunk in self.parents]
        child_ids = [chunk.chunk_id for chunk in self.children]
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("parent chunk IDs must be unique")
        if len(child_ids) != len(set(child_ids)):
            raise ValueError("child chunk IDs must be unique")

        child_parent = {chunk.chunk_id: chunk.parent_id for chunk in self.children}
        expected_children: set[str] = set()
        for parent in self.parents:
            if parent.document_id != self.document_id or parent.strategy != self.strategy:
                raise ValueError("all parent chunks must belong to this hierarchy")
            if len(parent.children_ids) != len(set(parent.children_ids)):
                raise ValueError("a parent cannot reference the same child twice")
            for child_id in parent.children_ids:
                if child_id not in child_parent:
                    raise ValueError("parent references an unknown child")
                if child_parent[child_id] != parent.chunk_id:
                    raise ValueError("parent and child relationship references disagree")
                expected_children.add(child_id)

        if expected_children != set(child_ids):
            raise ValueError("every child must be referenced by exactly one parent")
        if any(
            child.document_id != self.document_id or child.strategy != self.strategy
            for child in self.children
        ):
            raise ValueError("all child chunks must belong to this hierarchy")
        return self

    @property
    def chunks(self) -> list[HierarchicalChunk]:
        return [*self.parents, *self.children]
