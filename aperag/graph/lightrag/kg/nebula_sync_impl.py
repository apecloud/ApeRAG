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
from dataclasses import dataclass
from typing import final

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger

# Import sync connection manager
try:
    from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
except ImportError:
    NebulaSyncConnectionManager = None


@final
@dataclass
class NebulaSyncStorage(BaseGraphStorage):
    """
    NebulaGraph storage implementation using sync driver with async interface.
    This avoids event loop issues while maintaining compatibility with async code.
    """

    def __init__(self, namespace, workspace, embedding_func):
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            embedding_func=embedding_func,
        )
        self._space_name = None

    async def initialize(self):
        """Initialize storage and prepare space."""
        if NebulaSyncConnectionManager is None:
            raise RuntimeError("NebulaGraph sync connection manager is not available")

        # Prepare space in thread to avoid blocking
        self._space_name = await asyncio.to_thread(NebulaSyncConnectionManager.prepare_space, self.workspace)

        logger.debug(f"NebulaSyncStorage initialized for workspace '{self.workspace}', space '{self._space_name}'")

    async def finalize(self):
        """Clean up resources."""
        # Nothing to clean up - connection managed at worker level
        logger.debug(f"NebulaSyncStorage finalized for workspace '{self.workspace}'")

    def _escape_string(self, value: str) -> str:
        """Escape string for NebulaGraph queries."""
        if value is None:
            return "NULL"
        # Escape backslashes first, then quotes
        return value.replace("\\", "\\\\").replace('"', '\\"')

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""

        def _sync_has_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # In NebulaGraph, we need to quote vertex IDs
                query = f'FETCH PROP ON base "{self._escape_string(node_id)}" YIELD vertex as v'
                result = session.execute(query)

                if result.is_succeeded() and result.get_row_size() > 0:
                    return True
                return False

        return await asyncio.to_thread(_sync_has_node)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes."""

        def _sync_has_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Check both directions since we treat edges as undirected
                query = f'''
                FETCH PROP ON DIRECTED "{self._escape_string(source_node_id)}" -> "{self._escape_string(target_node_id)}" YIELD edge as e
                UNION
                FETCH PROP ON DIRECTED "{self._escape_string(target_node_id)}" -> "{self._escape_string(source_node_id)}" YIELD edge as e
                '''
                result = session.execute(query)

                if result.is_succeeded() and result.get_row_size() > 0:
                    return True
                return False

        return await asyncio.to_thread(_sync_has_edge)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier."""

        def _sync_get_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Fetch properties from both base and entity tags
                query = f'''
                FETCH PROP ON base, entity "{self._escape_string(node_id)}" 
                YIELD properties(vertex) as props
                '''
                result = session.execute(query)

                if result.is_succeeded() and result.get_row_size() > 0:
                    row = result.get_rows()[0]
                    props_value = row.values[0]

                    # Parse properties from NebulaGraph format
                    if props_value.is_map():
                        props_map = props_value.get_mVal()
                        node_dict = {}

                        for key, value in props_map.items():
                            key_str = key.decode() if isinstance(key, bytes) else str(key)

                            if value.is_string():
                                node_dict[key_str] = value.get_sVal().decode()
                            elif value.is_int():
                                node_dict[key_str] = value.get_iVal()
                            elif value.is_double():
                                node_dict[key_str] = value.get_dVal()
                            elif value.is_null():
                                node_dict[key_str] = None
                            else:
                                # Handle other types as string
                                node_dict[key_str] = str(value)

                        return node_dict

                return None

        return await asyncio.to_thread(_sync_get_node)

    async def node_degree(self, node_id: str) -> int:
        """Get the degree of a node."""

        def _sync_node_degree():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Count both incoming and outgoing edges
                query = f'''
                GO FROM "{self._escape_string(node_id)}" OVER * BIDIRECT 
                YIELD src(edge) as src, dst(edge) as dst
                | YIELD COUNT(*) as degree
                '''
                result = session.execute(query)

                if result.is_succeeded() and result.get_row_size() > 0:
                    row = result.get_rows()[0]
                    return row.values[0].get_iVal()

                return 0

        return await asyncio.to_thread(_sync_node_degree)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of two nodes."""
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)
        return int(src_degree) + int(trg_degree)

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """Get edge properties between two nodes."""

        def _sync_get_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Try both directions
                query = f'''
                FETCH PROP ON DIRECTED "{self._escape_string(source_node_id)}" -> "{self._escape_string(target_node_id)}" 
                YIELD properties(edge) as props
                '''
                result = session.execute(query)

                if not result.is_succeeded() or result.get_row_size() == 0:
                    # Try reverse direction
                    query = f'''
                    FETCH PROP ON DIRECTED "{self._escape_string(target_node_id)}" -> "{self._escape_string(source_node_id)}" 
                    YIELD properties(edge) as props
                    '''
                    result = session.execute(query)

                if result.is_succeeded() and result.get_row_size() > 0:
                    row = result.get_rows()[0]
                    props_value = row.values[0]

                    # Parse properties
                    if props_value.is_map():
                        props_map = props_value.get_mVal()
                        edge_dict = {
                            "weight": 0.0,
                            "source_id": None,
                            "description": None,
                            "keywords": None,
                        }

                        for key, value in props_map.items():
                            key_str = key.decode() if isinstance(key, bytes) else str(key)

                            if value.is_string():
                                edge_dict[key_str] = value.get_sVal().decode()
                            elif value.is_double():
                                edge_dict[key_str] = value.get_dVal()
                            elif value.is_int():
                                edge_dict[key_str] = value.get_iVal()
                            elif value.is_null():
                                edge_dict[key_str] = None
                            else:
                                edge_dict[key_str] = str(value)

                        return edge_dict

                return None

        return await asyncio.to_thread(_sync_get_edge)

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges for a node."""

        def _sync_get_node_edges():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Get both incoming and outgoing edges
                query = f'''
                GO FROM "{self._escape_string(source_node_id)}" OVER * BIDIRECT 
                YIELD src(edge) as src, dst(edge) as dst
                '''
                result = session.execute(query)

                edges = []
                if result.is_succeeded():
                    for row in result.get_rows():
                        src_val = row.values[0].get_sVal().decode()
                        dst_val = row.values[1].get_sVal().decode()
                        edges.append((src_val, dst_val))

                return edges if edges else None

        return await asyncio.to_thread(_sync_get_node_edges)

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the database."""

        def _sync_upsert_node():
            properties = node_data.copy()
            entity_type = properties.get("entity_type", "entity")

            if "entity_id" not in properties:
                raise ValueError("NebulaGraph: node properties must contain an 'entity_id' field")

            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Build property string
                prop_pairs = []
                for key, value in properties.items():
                    if value is None:
                        prop_pairs.append(f"{key}: NULL")
                    elif isinstance(value, str):
                        prop_pairs.append(f'{key}: "{self._escape_string(value)}"')
                    elif isinstance(value, (int, float)):
                        prop_pairs.append(f"{key}: {value}")
                    else:
                        # Convert other types to string
                        prop_pairs.append(f'{key}: "{self._escape_string(str(value))}"')

                prop_string = ", ".join(prop_pairs)

                # UPSERT vertex with both base and entity tags
                query = f'''
                UPSERT VERTEX ON base "{self._escape_string(node_id)}" 
                SET {prop_string}
                '''
                result = session.execute(query)

                if not result.is_succeeded():
                    logger.error(f"Failed to upsert node: {result.error_msg()}")
                    raise RuntimeError(f"Failed to upsert node: {result.error_msg()}")

                # Also add entity tag if it's an entity type
                if entity_type == "entity":
                    query = f'''
                    UPSERT VERTEX ON entity "{self._escape_string(node_id)}" 
                    SET {prop_string}
                    '''
                    result = session.execute(query)

                    if not result.is_succeeded():
                        logger.warning(f"Failed to add entity tag: {result.error_msg()}")

                logger.debug(f"Upserted node with entity_id '{node_id}'")

        return await asyncio.to_thread(_sync_upsert_node)

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """Upsert an edge between two nodes."""

        def _sync_upsert_edge():
            edge_properties = edge_data.copy()

            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Build property string
                prop_pairs = []
                for key, value in edge_properties.items():
                    if value is None:
                        prop_pairs.append(f"{key}: NULL")
                    elif isinstance(value, str):
                        prop_pairs.append(f'{key}: "{self._escape_string(value)}"')
                    elif isinstance(value, (int, float)):
                        prop_pairs.append(f"{key}: {value}")
                    else:
                        prop_pairs.append(f'{key}: "{self._escape_string(str(value))}"')

                prop_string = ", ".join(prop_pairs)

                # UPSERT edge (NebulaGraph treats edges as directed)
                query = f'''
                UPSERT EDGE ON DIRECTED "{self._escape_string(source_node_id)}" -> "{self._escape_string(target_node_id)}" 
                SET {prop_string}
                '''
                result = session.execute(query)

                if not result.is_succeeded():
                    logger.error(f"Failed to upsert edge: {result.error_msg()}")
                    raise RuntimeError(f"Failed to upsert edge: {result.error_msg()}")

                logger.debug(f"Upserted edge from '{source_node_id}' to '{target_node_id}'")

        return await asyncio.to_thread(_sync_upsert_edge)

    async def delete_node(self, node_id: str) -> None:
        """Delete a node."""

        def _sync_delete_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # DELETE VERTEX will automatically delete associated edges
                query = f'DELETE VERTEX "{self._escape_string(node_id)}" WITH EDGE'
                result = session.execute(query)

                if not result.is_succeeded():
                    logger.error(f"Failed to delete node: {result.error_msg()}")
                    raise RuntimeError(f"Failed to delete node: {result.error_msg()}")

                logger.debug(f"Deleted node with label '{node_id}'")

        return await asyncio.to_thread(_sync_delete_node)

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes."""
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges."""

        def _sync_remove_edge(source: str, target: str):
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Delete edge in both directions to handle undirected nature
                query = f'''
                DELETE EDGE DIRECTED "{self._escape_string(source)}" -> "{self._escape_string(target)}"
                '''
                result = session.execute(query)

                if result.is_succeeded():
                    logger.debug(f"Deleted edge from '{source}' to '{target}'")

                # Also try reverse direction
                query = f'''
                DELETE EDGE DIRECTED "{self._escape_string(target)}" -> "{self._escape_string(source)}"
                '''
                result = session.execute(query)

                if result.is_succeeded():
                    logger.debug(f"Deleted edge from '{target}' to '{source}'")

        for source, target in edges:
            await asyncio.to_thread(_sync_remove_edge, source, target)

    async def get_all_labels(self) -> list[str]:
        """Get all node labels."""

        def _sync_get_all_labels():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # In NebulaGraph, we need to query all vertices and extract entity_id
                query = """
                MATCH (n:base)
                RETURN DISTINCT n.base.entity_id AS label
                ORDER BY label
                """
                result = session.execute(query)

                labels = []
                if result.is_succeeded():
                    for row in result.get_rows():
                        if row.values[0].is_string():
                            label = row.values[0].get_sVal().decode()
                            labels.append(label)

                return labels

        return await asyncio.to_thread(_sync_get_all_labels)

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """Retrieve a connected subgraph."""

        def _sync_get_knowledge_graph():
            result = KnowledgeGraph()
            seen_nodes = set()
            seen_edges = set()

            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if node_label == "*":
                    # Get all nodes up to max_nodes
                    query = f"""
                    MATCH (n:base)
                    RETURN id(n) as node_id, n.base.entity_id as entity_id, properties(n) as props
                    LIMIT {max_nodes}
                    """
                else:
                    # BFS from specific node using GO statement
                    query = f'''
                    GO {max_depth} STEPS FROM "{self._escape_string(node_label)}" OVER * BIDIRECT 
                    YIELD DISTINCT dst(edge) as node_id
                    | FETCH PROP ON base, entity $-.node_id 
                    YIELD id(vertex) as node_id, properties(vertex) as props
                    | LIMIT {max_nodes}
                    '''

                result_nodes = session.execute(query)

                if result_nodes.is_succeeded():
                    node_ids = []

                    for row in result_nodes.get_rows():
                        node_id = row.values[0].get_sVal().decode() if row.values[0].is_string() else str(row.values[0])

                        if node_id not in seen_nodes:
                            # Parse properties
                            props = {}
                            if len(row.values) > 1 and row.values[1].is_map():
                                props_map = row.values[1].get_mVal()
                                for key, value in props_map.items():
                                    key_str = key.decode() if isinstance(key, bytes) else str(key)
                                    if value.is_string():
                                        props[key_str] = value.get_sVal().decode()
                                    elif value.is_int():
                                        props[key_str] = value.get_iVal()
                                    elif value.is_double():
                                        props[key_str] = value.get_dVal()
                                    else:
                                        props[key_str] = str(value)

                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=node_id,
                                    labels=[props.get("entity_id", node_id)],
                                    properties=props,
                                )
                            )
                            seen_nodes.add(node_id)
                            node_ids.append(node_id)

                    # Get edges between collected nodes
                    if node_ids:
                        # Build a query to get all edges between the nodes
                        node_id_list = ", ".join([f'"{self._escape_string(nid)}"' for nid in node_ids])
                        edge_query = f"""
                        GO FROM {node_id_list} OVER * 
                        WHERE dst(edge) IN [{node_id_list}]
                        YIELD src(edge) as src, dst(edge) as dst, properties(edge) as props, type(edge) as edge_type
                        """

                        result_edges = session.execute(edge_query)

                        if result_edges.is_succeeded():
                            for row in result_edges.get_rows():
                                src = row.values[0].get_sVal().decode()
                                dst = row.values[1].get_sVal().decode()
                                edge_type = row.values[3].get_sVal().decode()

                                edge_id = f"{src}-{edge_type}-{dst}"
                                if edge_id not in seen_edges:
                                    # Parse edge properties
                                    edge_props = {}
                                    if row.values[2].is_map():
                                        props_map = row.values[2].get_mVal()
                                        for key, value in props_map.items():
                                            key_str = key.decode() if isinstance(key, bytes) else str(key)
                                            if value.is_string():
                                                edge_props[key_str] = value.get_sVal().decode()
                                            elif value.is_double():
                                                edge_props[key_str] = value.get_dVal()
                                            elif value.is_int():
                                                edge_props[key_str] = value.get_iVal()
                                            else:
                                                edge_props[key_str] = str(value)

                                    result.edges.append(
                                        KnowledgeGraphEdge(
                                            id=edge_id,
                                            type=edge_type,
                                            source=src,
                                            target=dst,
                                            properties=edge_props,
                                        )
                                    )
                                    seen_edges.add(edge_id)

            logger.info(f"Retrieved subgraph with {len(result.nodes)} nodes and {len(result.edges)} edges")
            return result

        return await asyncio.to_thread(_sync_get_knowledge_graph)

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage."""

        def _sync_drop():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Delete all vertices (which will also delete edges)
                query = "DELETE VERTEX * WITH EDGE"
                result = session.execute(query)

                if result.is_succeeded():
                    logger.info(f"Dropped all data from space {self._space_name}")
                    return {"status": "success", "message": "data dropped"}
                else:
                    logger.error(f"Failed to drop data: {result.error_msg()}")
                    return {"status": "error", "message": result.error_msg()}

        return await asyncio.to_thread(_sync_drop)
