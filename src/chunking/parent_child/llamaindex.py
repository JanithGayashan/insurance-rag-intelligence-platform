from __future__ import annotations

from pathlib import Path

import tiktoken
from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.schema import MetadataMode, NodeRelationship

from src.chunking.parent_child.custom import _validate_configuration
from src.chunking.parent_child.models import (
    HierarchicalChunk,
    ParentChildHierarchy,
)


class LlamaIndexParentChildChunker:
    """Build a two-level hierarchy with LlamaIndex node parsers."""

    strategy = "parent-child-llamaindex"

    def __init__(
        self,
        parent_size: int = 1500,
        child_size: int = 300,
        child_overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        _validate_configuration(parent_size, child_size, child_overlap)
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)

        parser_ids = ["parent", "child"]
        parser_map = {
            "parent": SentenceSplitter(
                chunk_size=parent_size,
                chunk_overlap=0,
                tokenizer=self.encoding.encode,
                include_metadata=False,
            ),
            "child": SentenceSplitter(
                chunk_size=child_size,
                chunk_overlap=child_overlap,
                tokenizer=self.encoding.encode,
                include_metadata=False,
            ),
        }
        self.parser = HierarchicalNodeParser(
            node_parser_ids=parser_ids,
            node_parser_map=parser_map,
            include_metadata=False,
        )

    def chunk_text(self, text: str, document_id: str) -> ParentChildHierarchy:
        text = text.strip()
        document_id = document_id.strip()
        if not text:
            raise ValueError("text cannot be empty")
        if not document_id:
            raise ValueError("document_id cannot be empty")

        nodes = self.parser.get_nodes_from_documents(
            [Document(text=text, doc_id=document_id)]
        )
        root_nodes = get_root_nodes(nodes)
        leaf_nodes = get_leaf_nodes(nodes)
        parent_id_by_native_id = {
            node.node_id: f"{document_id}-llamaindex-parent-{index:04d}"
            for index, node in enumerate(root_nodes)
        }

        child_records: list[tuple[object, str, str]] = []
        children_by_parent: dict[str, list[str]] = {
            parent_id: [] for parent_id in parent_id_by_native_id.values()
        }
        for child_index, node in enumerate(leaf_nodes):
            parent_relation = node.relationships.get(NodeRelationship.PARENT)
            if parent_relation is None or isinstance(parent_relation, list):
                raise ValueError("LlamaIndex produced a child without one parent")
            try:
                parent_id = parent_id_by_native_id[parent_relation.node_id]
            except KeyError as error:
                raise ValueError(
                    "LlamaIndex child references an unknown parent node"
                ) from error
            child_id = f"{document_id}-llamaindex-child-{child_index:04d}"
            child_records.append((node, child_id, parent_id))
            children_by_parent[parent_id].append(child_id)

        parents = [
            HierarchicalChunk(
                chunk_id=parent_id_by_native_id[node.node_id],
                document_id=document_id,
                chunk_index=index,
                level=0,
                text=node.get_content(metadata_mode=MetadataMode.NONE).strip(),
                token_count=len(
                    self.encoding.encode(
                        node.get_content(metadata_mode=MetadataMode.NONE).strip()
                    )
                ),
                children_ids=children_by_parent[parent_id_by_native_id[node.node_id]],
                is_leaf=False,
                strategy=self.strategy,
                metadata={"splitter": "llamaindex-sentence-splitter"},
            )
            for index, node in enumerate(root_nodes)
        ]
        parent_index_by_id = {
            parent.chunk_id: parent.chunk_index for parent in parents
        }
        children = [
            HierarchicalChunk(
                chunk_id=child_id,
                document_id=document_id,
                chunk_index=index,
                level=1,
                text=node.get_content(metadata_mode=MetadataMode.NONE).strip(),
                token_count=len(
                    self.encoding.encode(
                        node.get_content(metadata_mode=MetadataMode.NONE).strip()
                    )
                ),
                parent_id=parent_id,
                is_leaf=True,
                strategy=self.strategy,
                metadata={
                    "parent_index": parent_index_by_id[parent_id],
                    "child_index_in_parent": children_by_parent[parent_id].index(
                        child_id
                    ),
                    "splitter": "llamaindex-sentence-splitter",
                },
            )
            for index, (node, child_id, parent_id) in enumerate(child_records)
        ]

        return ParentChildHierarchy(
            document_id=document_id,
            strategy=self.strategy,
            parent_size=self.parent_size,
            child_size=self.child_size,
            child_overlap=self.child_overlap,
            parents=parents,
            children=children,
            metadata={"tokenizer": self.encoding_name, "parent_overlap": 0},
        )

    def chunk_file(
        self,
        document_path: Path,
        document_id: str | None = None,
    ) -> ParentChildHierarchy:
        document_path = Path(document_path)
        return self.chunk_text(
            document_path.read_text(encoding="utf-8"),
            document_id or document_path.parent.name,
        )
