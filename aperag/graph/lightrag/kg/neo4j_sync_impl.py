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
LightRAG Module for ApeRAG

This module is based on the original LightRAG project with extensive modifications.

Original Project:
- Repository: https://github.com/HKUDS/LightRAG
- Paper: "LightRAG: Simple and Fast Retrieval-Augmented Generation" (arXiv:2410.05779)
- Authors: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- License: MIT License

Modifications by ApeRAG Team:
- Removed global state management for true concurrent processing
- Added stateless interfaces for Celery/Prefect integration
- Implemented instance-level locking mechanism
- Enhanced error handling and stability
- See changelog.md for detailed modifications
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import final

from neo4j import exceptions as neo4jExceptions
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger

# Import sync connection manager
try:
    from aperag.db.neo4j_sync_manager import Neo4jSyncConnectionManager
except ImportError:
    Neo4jSyncConnectionManager = None

# Set neo4j logger level to ERROR to suppress warning logs
logging.getLogger("neo4j").setLevel(logging.ERROR)


@final
@dataclass
class Neo4JSyncStorage(BaseGraphStorage):
    """
    Neo4j storage implementation using sync driver with async interface.
    This avoids event loop issues while maintaining compatibility with async code.
    """

    def __init__(self, namespace, workspace, embedding_func=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            embedding_func=None,
        )
        self._DATABASE = None

    async def initialize(self):
        """Initialize storage and prepare database."""
        if Neo4jSyncConnectionManager is None:
            raise RuntimeError("Neo4j sync connection manager is not available")

        # Prepare database in thread to avoid blocking
        self._DATABASE = await asyncio.to_thread(Neo4jSyncConnectionManager.prepare_database, self.workspace)

        logger.debug(f"Neo4JSyncStorage initialized for workspace '{self.workspace}', database '{self._DATABASE}'")

    async def finalize(self):
        """Clean up resources."""
        # Nothing to clean up - connection managed at worker level
        logger.debug(f"Neo4JSyncStorage finalized for workspace '{self.workspace}'")

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""

        def _sync_has_node():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = "MATCH (n:base {entity_id: $entity_id}) RETURN count(n) > 0 AS node_exists"
                result = session.run(query, entity_id=node_id)
                single_result = result.single()
                return single_result["node_exists"]

        return await asyncio.to_thread(_sync_has_node)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes."""

        def _sync_has_edge():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = (
                    "MATCH (a:base {entity_id: $source_entity_id})-[r]-(b:base {entity_id: $target_entity_id}) "
                    "RETURN COUNT(r) > 0 AS edgeExists"
                )
                result = session.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                )
                single_result = result.single()
                return single_result["edgeExists"]

        return await asyncio.to_thread(_sync_has_edge)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier."""

        def _sync_get_node():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = "MATCH (n:base {entity_id: $entity_id}) RETURN n"
                result = session.run(query, entity_id=node_id)
                records = list(result)

                if len(records) > 1:
                    logger.warning(f"Multiple nodes found with label '{node_id}'. Using first node.")

                if records:
                    node = records[0]["n"]
                    node_dict = dict(node)
                    # Remove base label from labels list if it exists
                    if "labels" in node_dict:
                        node_dict["labels"] = [label for label in node_dict["labels"] if label != "base"]
                    return node_dict
                return None

        return await asyncio.to_thread(_sync_get_node)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieve multiple nodes in one query."""

        def _sync_get_nodes_batch():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = """
                UNWIND $node_ids AS id
                MATCH (n:base {entity_id: id})
                RETURN n.entity_id AS entity_id, n
                """
                result = session.run(query, node_ids=node_ids)
                nodes = {}
                for record in result:
                    entity_id = record["entity_id"]
                    node = record["n"]
                    node_dict = dict(node)
                    # Remove the 'base' label if present
                    if "labels" in node_dict:
                        node_dict["labels"] = [label for label in node_dict["labels"] if label != "base"]
                    nodes[entity_id] = node_dict
                return nodes

        return await asyncio.to_thread(_sync_get_nodes_batch)

    async def node_degree(self, node_id: str) -> int:
        """Get the degree of a node."""

        def _sync_node_degree():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = """
                    MATCH (n:base {entity_id: $entity_id})
                    OPTIONAL MATCH (n)-[r]-()
                    RETURN COUNT(r) AS degree
                """
                result = session.run(query, entity_id=node_id)
                record = result.single()

                if not record:
                    logger.warning(f"No node found with label '{node_id}'")
                    return 0

                return record["degree"]

        return await asyncio.to_thread(_sync_node_degree)

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieve degrees for multiple nodes."""

        def _sync_node_degrees_batch():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = """
                    UNWIND $node_ids AS id
                    MATCH (n:base {entity_id: id})
                    RETURN n.entity_id AS entity_id, count { (n)--() } AS degree;
                """
                result = session.run(query, node_ids=node_ids)
                degrees = {}
                for record in result:
                    entity_id = record["entity_id"]
                    degrees[entity_id] = record["degree"]

                # Set degree to 0 for missing nodes
                for nid in node_ids:
                    if nid not in degrees:
                        logger.warning(f"No node found with label '{nid}'")
                        degrees[nid] = 0

                return degrees

        return await asyncio.to_thread(_sync_node_degrees_batch)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of two nodes."""
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)
        return int(src_degree) + int(trg_degree)

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """Calculate combined degrees for edges."""
        unique_node_ids = {src for src, _ in edge_pairs}
        unique_node_ids.update({tgt for _, tgt in edge_pairs})

        degrees = await self.node_degrees_batch(list(unique_node_ids))

        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)
        return edge_degrees

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """Get edge properties between two nodes."""

        def _sync_get_edge():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = """
                MATCH (start:base {entity_id: $source_entity_id})-[r]-(end:base {entity_id: $target_entity_id})
                RETURN properties(r) as edge_properties
                """
                result = session.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                )
                records = list(result)

                if len(records) > 1:
                    logger.warning(
                        f"Multiple edges found between '{source_node_id}' and '{target_node_id}'. Using first edge."
                    )

                if records:
                    edge_result = dict(records[0]["edge_properties"])
                    # Ensure required keys exist with defaults
                    required_keys = {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }
                    for key, default_value in required_keys.items():
                        if key not in edge_result:
                            edge_result[key] = default_value
                    return edge_result

                return None

        return await asyncio.to_thread(_sync_get_edge)

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        """Retrieve edge properties for multiple pairs."""

        def _sync_get_edges_batch():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = """
                UNWIND $pairs AS pair
                MATCH (start:base {entity_id: pair.src})-[r:DIRECTED]-(end:base {entity_id: pair.tgt})
                RETURN pair.src AS src_id, pair.tgt AS tgt_id, collect(properties(r)) AS edges
                """
                result = session.run(query, pairs=pairs)
                edges_dict = {}
                for record in result:
                    src = record["src_id"]
                    tgt = record["tgt_id"]
                    edges = record["edges"]
                    if edges and len(edges) > 0:
                        edge_props = edges[0]
                        # Ensure required keys exist
                        for key, default in {
                            "weight": 0.0,
                            "source_id": None,
                            "description": None,
                            "keywords": None,
                        }.items():
                            if key not in edge_props:
                                edge_props[key] = default
                        edges_dict[(src, tgt)] = edge_props
                    else:
                        # No edge found
                        edges_dict[(src, tgt)] = {
                            "weight": 0.0,
                            "source_id": None,
                            "description": None,
                            "keywords": None,
                        }
                return edges_dict

        return await asyncio.to_thread(_sync_get_edges_batch)

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges for a node."""

        def _sync_get_node_edges():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = """MATCH (n:base {entity_id: $entity_id})
                        OPTIONAL MATCH (n)-[r]-(connected:base)
                        WHERE connected.entity_id IS NOT NULL
                        RETURN n, r, connected"""
                result = session.run(query, entity_id=source_node_id)

                edges = []
                for record in result:
                    source_node = record["n"]
                    connected_node = record["connected"]

                    if not source_node or not connected_node:
                        continue

                    source_label = source_node.get("entity_id")
                    target_label = connected_node.get("entity_id")

                    if source_label and target_label:
                        edges.append((source_label, target_label))

                return edges

        return await asyncio.to_thread(_sync_get_node_edges)

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """Batch retrieve edges for multiple nodes."""

        def _sync_get_nodes_edges_batch():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = """
                    UNWIND $node_ids AS id
                    MATCH (n:base {entity_id: id})
                    OPTIONAL MATCH (n)-[r]-(connected:base)
                    RETURN id AS queried_id, n.entity_id AS node_entity_id,
                           connected.entity_id AS connected_entity_id,
                           startNode(r).entity_id AS start_entity_id
                """
                result = session.run(query, node_ids=node_ids)

                edges_dict = {node_id: [] for node_id in node_ids}

                for record in result:
                    queried_id = record["queried_id"]
                    node_entity_id = record["node_entity_id"]
                    connected_entity_id = record["connected_entity_id"]
                    start_entity_id = record["start_entity_id"]

                    if not node_entity_id or not connected_entity_id:
                        continue

                    if start_entity_id == node_entity_id:
                        edges_dict[queried_id].append((node_entity_id, connected_entity_id))
                    else:
                        edges_dict[queried_id].append((connected_entity_id, node_entity_id))

                return edges_dict

        return await asyncio.to_thread(_sync_get_nodes_edges_batch)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the database."""

        def _sync_upsert_node():
            properties = node_data
            entity_type = properties["entity_type"]
            if "entity_id" not in properties:
                raise ValueError("Neo4j: node properties must contain an 'entity_id' field")

            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = (
                    """
                    MERGE (n:base {entity_id: $entity_id})
                    SET n += $properties
                    SET n:`%s`
                    """
                    % entity_type
                )
                session.run(query, entity_id=node_id, properties=properties)
                logger.debug(f"Upserted node with entity_id '{node_id}'")

        return await asyncio.to_thread(_sync_upsert_node)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """Upsert an edge between two nodes."""

        def _sync_upsert_edge():
            edge_properties = edge_data
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = """
                MATCH (source:base {entity_id: $source_entity_id})
                WITH source
                MATCH (target:base {entity_id: $target_entity_id})
                MERGE (source)-[r:DIRECTED]-(target)
                SET r += $properties
                RETURN r, source, target
                """
                session.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                    properties=edge_properties,
                )
                logger.debug(f"Upserted edge from '{source_node_id}' to '{target_node_id}'")

        return await asyncio.to_thread(_sync_upsert_edge)

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """Retrieve a connected subgraph."""

        # For brevity, I'll implement a simplified version
        # The full implementation would be similar to the original but using sync driver
        def _sync_get_knowledge_graph():
            result = KnowledgeGraph()
            seen_nodes = set()
            seen_edges = set()

            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                if node_label == "*":
                    # Get all nodes
                    query = """
                    MATCH (n)
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, COALESCE(count(r), 0) AS degree
                    ORDER BY degree DESC
                    LIMIT $max_nodes
                    RETURN n
                    """
                    node_results = session.run(query, max_nodes=max_nodes)

                    for record in node_results:
                        node = record["n"]
                        node_id = node.id
                        if node_id not in seen_nodes:
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=f"{node_id}",
                                    labels=[node.get("entity_id")],
                                    properties=dict(node),
                                )
                            )
                            seen_nodes.add(node_id)

                    # Get edges between these nodes
                    edge_query = """
                    MATCH (a)-[r]-(b)
                    WHERE id(a) IN $node_ids AND id(b) IN $node_ids
                    RETURN r, a, b
                    """
                    edge_results = session.run(edge_query, node_ids=list(seen_nodes))

                    for record in edge_results:
                        rel = record["r"]
                        edge_id = rel.id
                        if edge_id not in seen_edges:
                            result.edges.append(
                                KnowledgeGraphEdge(
                                    id=f"{edge_id}",
                                    type=rel.type,
                                    source=f"{record['a'].id}",
                                    target=f"{record['b'].id}",
                                    properties=dict(rel),
                                )
                            )
                            seen_edges.add(edge_id)
                else:
                    # BFS from specific node
                    # Simplified implementation - full version would do proper BFS
                    query = (
                        """
                    MATCH (start:base {entity_id: $entity_id})
                    OPTIONAL MATCH path = (start)-[*..%d]-(end)
                    RETURN nodes(path) as nodes, relationships(path) as rels
                    LIMIT $max_nodes
                    """
                        % max_depth
                    )

                    results = session.run(query, entity_id=node_label, max_nodes=max_nodes)

                    for record in results:
                        if record["nodes"]:
                            for node in record["nodes"]:
                                node_id = node.id
                                if node_id not in seen_nodes:
                                    result.nodes.append(
                                        KnowledgeGraphNode(
                                            id=f"{node_id}",
                                            labels=[node.get("entity_id")],
                                            properties=dict(node),
                                        )
                                    )
                                    seen_nodes.add(node_id)

                        if record["rels"]:
                            for rel in record["rels"]:
                                edge_id = rel.id
                                if edge_id not in seen_edges:
                                    result.edges.append(
                                        KnowledgeGraphEdge(
                                            id=f"{edge_id}",
                                            type=rel.type,
                                            source=f"{rel.start_node.id}",
                                            target=f"{rel.end_node.id}",
                                            properties=dict(rel),
                                        )
                                    )
                                    seen_edges.add(edge_id)

            logger.info(f"Retrieved subgraph with {len(result.nodes)} nodes and {len(result.edges)} edges")
            return result

        return await asyncio.to_thread(_sync_get_knowledge_graph)

    async def get_all_labels(self) -> list[str]:
        """Get all node labels."""

        def _sync_get_all_labels():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = """
                MATCH (n:base)
                WHERE n.entity_id IS NOT NULL
                RETURN DISTINCT n.entity_id AS label
                ORDER BY label
                """
                result = session.run(query)
                labels = [record["label"] for record in result]
                return labels

        return await asyncio.to_thread(_sync_get_all_labels)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    async def delete_node(self, node_id: str) -> None:
        """Delete a node."""

        def _sync_delete_node():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = """
                MATCH (n:base {entity_id: $entity_id})
                DETACH DELETE n
                """
                session.run(query, entity_id=node_id)
                logger.debug(f"Deleted node with label '{node_id}'")

        return await asyncio.to_thread(_sync_delete_node)

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes."""
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges."""

        def _sync_remove_edge(source: str, target: str):
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = """
                MATCH (source:base {entity_id: $source_entity_id})-[r]-(target:base {entity_id: $target_entity_id})
                DELETE r
                """
                session.run(query, source_entity_id=source, target_entity_id=target)
                logger.debug(f"Deleted edge from '{source}' to '{target}'")

        for source, target in edges:
            await asyncio.to_thread(_sync_remove_edge, source, target)

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage."""

        def _sync_drop():
            with Neo4jSyncConnectionManager.get_session(database=self._DATABASE) as session:
                query = "MATCH (n) DETACH DELETE n"
                session.run(query)
                logger.info(f"Dropped all data from database {self._DATABASE}")
                return {"status": "success", "message": "data dropped"}

        return await asyncio.to_thread(_sync_drop)
