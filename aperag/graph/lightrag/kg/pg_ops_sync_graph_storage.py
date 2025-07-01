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

    # ========== Optimized Batch Operations ==========

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieve multiple nodes in batch using optimized SQL."""

        def _sync_get_nodes_batch():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            return db_ops.get_graph_nodes_batch(self.workspace, node_ids)

        return await asyncio.to_thread(_sync_get_nodes_batch)

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieve degrees for multiple nodes using optimized SQL."""

        def _sync_node_degrees_batch():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            return db_ops.get_graph_node_degrees_batch(self.workspace, node_ids)

        return await asyncio.to_thread(_sync_node_degrees_batch)

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """Calculate combined degrees for edges using efficient batch processing."""

        def _sync_edge_degrees_batch():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            # Extract unique node IDs from edge pairs
            unique_node_ids = set()
            for src, tgt in edge_pairs:
                unique_node_ids.add(src)
                unique_node_ids.add(tgt)

            # Get all node degrees in one batch call
            node_degrees = db_ops.get_graph_node_degrees_batch(self.workspace, list(unique_node_ids))

            # Calculate edge degrees
            edge_degrees = {}
            for src, tgt in edge_pairs:
                src_degree = node_degrees.get(src, 0)
                tgt_degree = node_degrees.get(tgt, 0)
                edge_degrees[(src, tgt)] = src_degree + tgt_degree

            return edge_degrees

        return await asyncio.to_thread(_sync_edge_degrees_batch)

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        """Retrieve edge properties for multiple pairs using optimized SQL."""

        def _sync_get_edges_batch():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            # Convert pairs format from [{"src": ..., "tgt": ...}] to [(src, tgt), ...]
            edge_pairs = [(pair["src"], pair["tgt"]) for pair in pairs]

            return db_ops.get_graph_edges_batch(self.workspace, edge_pairs)

        return await asyncio.to_thread(_sync_get_edges_batch)

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """Batch retrieve edges for multiple nodes using optimized SQL."""

        def _sync_get_nodes_edges_batch():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            return db_ops.get_graph_nodes_edges_batch(self.workspace, node_ids)

        return await asyncio.to_thread(_sync_get_nodes_edges_batch)

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its related edges in a single transaction."""

        def _sync_delete_node():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            db_ops.delete_graph_node(self.workspace, node_id)

        await asyncio.to_thread(_sync_delete_node)
        logger.debug(f"Node {node_id} and its related edges have been deleted from the graph")

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes using optimized batch SQL."""

        def _sync_remove_nodes():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            db_ops.delete_graph_nodes_batch(self.workspace, nodes)

        await asyncio.to_thread(_sync_remove_nodes)
        logger.debug(f"Batch deleted {len(nodes)} nodes and their related edges")

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges using optimized batch SQL."""

        def _sync_remove_edges():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            db_ops.delete_graph_edges_batch(self.workspace, edges)

        await asyncio.to_thread(_sync_remove_edges)
        logger.debug(f"Batch deleted {len(edges)} edges")

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

            # Get node details for each matching label using batch operation
            if matching_labels:
                nodes_data = db_ops.get_graph_nodes_batch(self.workspace, matching_labels)

                for entity_id, node_data in nodes_data.items():
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

                # Get edges between the selected nodes using batch operation
                node_names = [node.id for node in result.nodes]
                if node_names:
                    nodes_edges = db_ops.get_graph_nodes_edges_batch(self.workspace, node_names)

                    # Collect unique edge pairs that connect nodes in our result set
                    edge_pairs_to_query = set()
                    for source_node in node_names:
                        edges = nodes_edges.get(source_node, [])
                        for source_entity_id, target_entity_id in edges:
                            # Only include edges between selected nodes
                            if source_entity_id in node_names and target_entity_id in node_names:
                                edge_pairs_to_query.add((source_entity_id, target_entity_id))

                    # Get edge details in batch
                    if edge_pairs_to_query:
                        edges_data = db_ops.get_graph_edges_batch(self.workspace, list(edge_pairs_to_query))

                        for (source_entity_id, target_entity_id), edge_data in edges_data.items():
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
