# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LightRAG PostgreSQL Graph Storage - SQLAlchemy Implementation

This module provides a unified implementation using SQLAlchemy ORM instead of raw psycopg3.
Benefits:
- Consistent with OLTP database technology stack
- Unified connection pool and configuration management
- Better ORM abstraction and type safety
- Easier maintenance and testing

Uses asyncio.to_thread to wrap synchronous GraphRepositoryMixin methods.
"""

import asyncio
from dataclasses import dataclass
from typing import final

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger


@final
@dataclass
class PGOpsSyncGraphStorage(BaseGraphStorage):
    """
    PostgreSQL graph storage implementation using unified SQLAlchemy ORM.
    Provides same interface and functionality as PostgreSQLGraphSyncStorage but with unified technology stack.
    """

    def __init__(self, namespace, workspace, embedding_func=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            embedding_func=None,
        )

    async def initialize(self):
        """Initialize storage using unified DatabaseOps."""
        logger.debug(f"PGOpsSyncGraphStorage initialized for workspace '{self.workspace}'")

    async def finalize(self):
        """Clean up resources."""
        logger.debug(f"PGOpsSyncGraphStorage finalized for workspace '{self.workspace}'")

    #################### upsert method ################
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the database - using individual fields for optimal performance."""

        def _sync_upsert_node():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            db_ops.upsert_graph_node(self.workspace, node_id, node_data)

        await asyncio.to_thread(_sync_upsert_node)

        # Log with same format as original
        entity_type = node_data.get("entity_type") or None
        logger.debug(f"Upserted node with entity_id '{node_id}', entity_type '{entity_type}'")

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """Upsert an edge between two nodes - using individual fields for optimal performance."""

        def _sync_upsert_edge():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            db_ops.upsert_graph_edge(self.workspace, source_node_id, target_node_id, edge_data)

        await asyncio.to_thread(_sync_upsert_edge)
        logger.debug(f"Upserted edge from '{source_node_id}' to '{target_node_id}'")

    # Query methods
    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""

        def _sync_has_node():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            return db_ops.has_graph_node(self.workspace, node_id)

        return await asyncio.to_thread(_sync_has_node)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes."""

        def _sync_has_edge():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            return db_ops.has_graph_edge(self.workspace, source_node_id, target_node_id)

        return await asyncio.to_thread(_sync_has_edge)

    async def node_degree(self, node_id: str) -> int:
        """Get the degree of a node."""

        def _sync_node_degree():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            return db_ops.get_graph_node_degree(self.workspace, node_id)

        return await asyncio.to_thread(_sync_node_degree)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of two nodes."""
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its identifier."""

        def _sync_get_node():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            return db_ops.get_graph_node(self.workspace, node_id)

        return await asyncio.to_thread(_sync_get_node)

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """Get edge between two nodes."""

        def _sync_get_edge():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            return db_ops.get_graph_edge(self.workspace, source_node_id, target_node_id)

        return await asyncio.to_thread(_sync_get_edge)

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges for a node."""

        def _sync_get_node_edges():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            edges = db_ops.get_graph_node_edges(self.workspace, source_node_id)
            return edges if edges else None

        return await asyncio.to_thread(_sync_get_node_edges)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieve multiple nodes in batch (simplified implementation)."""
        nodes = {}
        for node_id in node_ids:
            node_data = await self.get_node(node_id)
            if node_data:
                nodes[node_id] = node_data
        return nodes

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieve degrees for multiple nodes."""
        degrees = {}
        for node_id in node_ids:
            degrees[node_id] = await self.node_degree(node_id)
        return degrees

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """Calculate combined degrees for edges."""
        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = await self.edge_degree(src, tgt)
        return edge_degrees

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        """Retrieve edge properties for multiple pairs."""
        edges_dict = {}
        for pair in pairs:
            src, tgt = pair["src"], pair["tgt"]
            edge_data = await self.get_edge(src, tgt)
            if edge_data:
                edges_dict[(src, tgt)] = edge_data
            else:
                # Return default structure with required fields (matching Neo4j behavior)
                edges_dict[(src, tgt)] = {
                    "weight": 0.0,
                    "keywords": None,
                    "description": None,
                    "source_id": None,
                }
        return edges_dict

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """Batch retrieve edges for multiple nodes."""
        edges_dict = {}
        for node_id in node_ids:
            edges = await self.get_node_edges(node_id)
            edges_dict[node_id] = edges or []
        return edges_dict

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its related edges in a single transaction."""

        def _sync_delete_node():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            db_ops.delete_graph_node(self.workspace, node_id)

        await asyncio.to_thread(_sync_delete_node)
        logger.debug(f"Node {node_id} and its related edges have been deleted from the graph")

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes."""
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges in a single transaction."""

        def _sync_remove_edges():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            db_ops.delete_graph_edges(self.workspace, edges)

        await asyncio.to_thread(_sync_remove_edges)

    async def get_all_labels(self) -> list[str]:
        """Get all entity names in the database."""

        def _sync_get_all_labels():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            return db_ops.get_all_graph_labels(self.workspace)

        return await asyncio.to_thread(_sync_get_all_labels)

    async def get_knowledge_graph(self, node_label: str, max_depth: int = 3, max_nodes: int = 1000) -> KnowledgeGraph:
        """
        Get a connected subgraph of nodes matching the specified label.

        Note: This is a simplified implementation that uses the existing Repository pattern.
        For now, it only supports getting nodes by label pattern and their immediate connections.
        Full graph traversal with max_depth would require additional Repository methods.
        """

        def _sync_get_knowledge_graph():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            result = KnowledgeGraph()
            MAX_GRAPH_NODES = max_nodes

            # Get all labels first
            all_labels = db_ops.get_all_graph_labels(self.workspace)

            # Filter based on node_label pattern
            if node_label == "*":
                # Get all nodes (limited by max_nodes)
                matching_labels = all_labels[:MAX_GRAPH_NODES]
            else:
                # Filter by pattern (similar to LIKE operation)
                matching_labels = [label for label in all_labels if node_label in label]
                if len(matching_labels) > MAX_GRAPH_NODES:
                    matching_labels = matching_labels[:MAX_GRAPH_NODES]

            # Get node details for each matching label
            for entity_id in matching_labels:
                node_data = db_ops.get_graph_node(self.workspace, entity_id)
                if node_data:
                    # Assemble properties from individual fields
                    properties = {
                        "entity_id": node_data["entity_id"],
                        "entity_type": node_data.get("entity_type"),
                        "description": node_data.get("description"),
                        "source_id": node_data.get("source_id"),
                        "file_path": node_data.get("file_path"),
                    }
                    # Only include entity_name if it's different from entity_id and not None
                    if "entity_name" in node_data and node_data["entity_name"] != entity_id:
                        properties["entity_name"] = node_data["entity_name"]

                    # Remove None values for cleaner output
                    properties = {k: v for k, v in properties.items() if v is not None}

                    result.nodes.append(
                        KnowledgeGraphNode(
                            id=entity_id,
                            labels=[node_data.get("entity_type", entity_id)],
                            properties=properties,
                        )
                    )

            # Get edges between the selected nodes
            node_names = [node.id for node in result.nodes]
            for i, source_node in enumerate(node_names):
                # Get edges for this node
                edges = db_ops.get_graph_node_edges(self.workspace, source_node)
                if edges:
                    for source_entity_id, target_entity_id in edges:
                        # Only include edges between selected nodes
                        if source_entity_id in node_names and target_entity_id in node_names:
                            # Get edge details
                            edge_data = db_ops.get_graph_edge(self.workspace, source_entity_id, target_entity_id)
                            if edge_data:
                                edge_id = f"{source_entity_id}-{target_entity_id}"

                                # Assemble edge properties from individual fields
                                edge_properties = {
                                    "weight": edge_data.get("weight", 0.0),
                                    "keywords": edge_data.get("keywords"),
                                    "description": edge_data.get("description"),
                                    "source_id": edge_data.get("source_id"),
                                    "file_path": edge_data.get("file_path"),
                                }
                                # Remove None values for cleaner output
                                edge_properties = {k: v for k, v in edge_properties.items() if v is not None}

                                result.edges.append(
                                    KnowledgeGraphEdge(
                                        id=edge_id,
                                        type="DIRECTED",
                                        source=source_entity_id,
                                        target=target_entity_id,
                                        properties=edge_properties,
                                    )
                                )

            return result

        result = await asyncio.to_thread(_sync_get_knowledge_graph)
        logger.info(f"Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}")
        return result

    async def drop(self) -> dict[str, str]:
        """Drop the storage in a single transaction."""

        def _sync_drop():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            return db_ops.drop_graph_workspace(self.workspace)

        result = await asyncio.to_thread(_sync_drop)

        if result.get("status") == "success":
            logger.info(f"Successfully dropped all data for workspace {self.workspace}")
        else:
            logger.error(f"Error dropping graph for workspace {self.workspace}: {result.get('message')}")

        return result
