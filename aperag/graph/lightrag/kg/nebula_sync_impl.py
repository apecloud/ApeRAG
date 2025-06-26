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
    from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
    from nebula3.Exception import IOErrorException, AuthFailedException
except ImportError:
    NebulaSyncConnectionManager = None
    IOErrorException = None
    AuthFailedException = None

# Set nebula logger level to ERROR to suppress warning logs
logging.getLogger("nebula3").setLevel(logging.ERROR)


def _safe_error_msg(result) -> str:
    """Safely extract error message from Nebula result, handling UTF-8 decode errors."""
    try:
        return result.error_msg()
    except UnicodeDecodeError as e:
        logger.warning(f"Failed to decode Nebula error message: {e}")
        return f"Nebula operation failed (error code: {result.error_code()}, UTF-8 decode error)"
    except Exception as e:
        logger.warning(f"Failed to get Nebula error message: {e}")
        return f"Nebula operation failed (error code: {result.error_code()})"


@final
@dataclass
class NebulaSyncStorage(BaseGraphStorage):
    """
    Nebula storage implementation using sync driver with async interface.
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
        """Initialize storage and prepare database."""
        if NebulaSyncConnectionManager is None:
            raise RuntimeError("Nebula sync connection manager is not available")

        # Prepare space in thread to avoid blocking
        self._space_name = await asyncio.to_thread(NebulaSyncConnectionManager.prepare_space, self.workspace)

        logger.debug(f"NebulaSyncStorage initialized for workspace '{self.workspace}', space '{self._space_name}'")

    async def finalize(self):
        """Clean up resources."""
        # Nothing to clean up - connection managed at worker level
        logger.debug(f"NebulaSyncStorage finalized for workspace '{self.workspace}'")

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""

        def _sync_has_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = f"FETCH PROP ON base '{node_id}' YIELD properties(vertex)"
                result = session.execute(query)
                if result.is_succeeded() and result.row_size() > 0:
                    return True
                return False

        return await asyncio.to_thread(_sync_has_node)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes."""

        def _sync_has_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Try to fetch edge properties between the two nodes
                query = f"""
                FETCH PROP ON DIRECTED '{source_node_id}' -> '{target_node_id}' 
                YIELD properties(edge) as props
                """
                result = session.execute(query)
                if result.is_succeeded() and result.row_size() > 0:
                    return True
                
                # Try reverse direction
                query = f"""
                FETCH PROP ON DIRECTED '{target_node_id}' -> '{source_node_id}' 
                YIELD properties(edge) as props
                """
                result = session.execute(query)
                if result.is_succeeded() and result.row_size() > 0:
                    return True
                
                return False

        return await asyncio.to_thread(_sync_has_edge)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier."""

        def _sync_get_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = f"FETCH PROP ON base '{node_id}' YIELD properties(vertex) as props"
                result = session.execute(query)
                
                if result.is_succeeded() and result.row_size() > 0:
                    for row in result:
                        props = row.values()[0].as_map()
                        node_dict = {}
                        for key, value in props.items():
                            key_str = key
                            if value.is_string():
                                node_dict[key_str] = value.as_string()
                            elif value.is_int():
                                node_dict[key_str] = value.as_int()
                            elif value.is_double():
                                node_dict[key_str] = value.as_double()
                        
                        # Add entity_id which is the node ID itself
                        node_dict['entity_id'] = node_id
                        return node_dict
                return None

        return await asyncio.to_thread(_sync_get_node)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieve multiple nodes in one query."""

        def _sync_get_nodes_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Build the list of IDs for the query
                id_list = ', '.join([f"'{node_id}'" for node_id in node_ids])
                query = f"FETCH PROP ON base {id_list} YIELD id(vertex) as id, properties(vertex) as props"
                result = session.execute(query)
                
                nodes = {}
                if result.is_succeeded():
                    for row in result:
                        node_id = row.values()[0].as_string()
                        props = row.values()[1].as_map()
                        node_dict = {}
                        for key, value in props.items():
                            key_str = key
                            if value.is_string():
                                node_dict[key_str] = value.as_string()
                            elif value.is_int():
                                node_dict[key_str] = value.as_int()
                            elif value.is_double():
                                node_dict[key_str] = value.as_double()
                        
                        # Add entity_id
                        node_dict['entity_id'] = node_id
                        nodes[node_id] = node_dict
                
                return nodes

        return await asyncio.to_thread(_sync_get_nodes_batch)

    async def node_degree(self, node_id: str) -> int:
        """Get the degree of a node."""

        def _sync_node_degree():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = f"""
                GO FROM '{node_id}' OVER * BIDIRECT 
                YIELD src(edge) as src, dst(edge) as dst 
                | YIELD count(*) as degree
                """
                result = session.execute(query)
                
                if result.is_succeeded() and result.row_size() > 0:
                    for row in result:
                        return row.values()[0].as_int()
                return 0

        return await asyncio.to_thread(_sync_node_degree)

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieve degrees for multiple nodes."""

        def _sync_node_degrees_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Build conditions for batch query
                conditions = []
                for node_id in node_ids:
                    conditions.append(f"id(n) == '{node_id}'")
                
                where_clause = " OR ".join(conditions)
                query = f"""
                MATCH (n:base)
                WHERE {where_clause}
                RETURN id(n) AS node_id, count((n)--()) AS degree
                """
                result = session.execute(query)
                
                degrees = {}
                if result.is_succeeded():
                    for row in result:
                        node_id = row.values()[0].as_string()
                        degree = row.values()[1].as_int()
                        degrees[node_id] = degree
                
                # Set degree to 0 for missing nodes
                for nid in node_ids:
                    if nid not in degrees:
                        logger.warning(f"No node found with id '{nid}'")
                        degrees[nid] = 0
                
                return degrees

        return await asyncio.to_thread(_sync_node_degrees_batch)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of two nodes."""
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return int(src_degree) + int(tgt_degree)

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
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = f"""
                FETCH PROP ON DIRECTED '{source_node_id}' -> '{target_node_id}' 
                YIELD properties(edge) as props
                """
                result = session.execute(query)
                
                if result.is_succeeded() and result.row_size() > 0:
                    for row in result:
                        props = row.values()[0].as_map()
                        edge_dict = {}
                        for key, value in props.items():
                            key_str = key
                            if value.is_string():
                                edge_dict[key_str] = value.as_string()
                            elif value.is_int():
                                edge_dict[key_str] = value.as_int()
                            elif value.is_double():
                                edge_dict[key_str] = value.as_double()
                        
                        # Ensure required keys exist with defaults
                        required_keys = {
                            "weight": 0.0,
                            "source_id": None,
                            "description": None,
                            "keywords": None,
                        }
                        for key, default_value in required_keys.items():
                            if key not in edge_dict:
                                edge_dict[key] = default_value
                        
                        return edge_dict
                
                return None

        return await asyncio.to_thread(_sync_get_edge)

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        """Retrieve edge properties for multiple pairs."""

        def _sync_get_edges_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                edges_dict = {}
                
                # Build edge specifications for batch fetch
                edge_specs = []
                for pair in pairs:
                    src = pair["src"]
                    tgt = pair["tgt"]
                    edge_specs.append(f"'{src}' -> '{tgt}'")
                
                edge_list = ', '.join(edge_specs)
                query = f"""
                FETCH PROP ON DIRECTED {edge_list} 
                YIELD src(edge) as src, dst(edge) as dst, properties(edge) as props
                """
                result = session.execute(query)
                
                if result.is_succeeded():
                    for row in result:
                        src = row.values()[0].as_string()
                        tgt = row.values()[1].as_string()
                        props = row.values()[2].as_map()
                        
                        edge_dict = {}
                        for key, value in props.items():
                            key_str = key
                            if value.is_string():
                                edge_dict[key_str] = value.as_string()
                            elif value.is_int():
                                edge_dict[key_str] = value.as_int()
                            elif value.is_double():
                                edge_dict[key_str] = value.as_double()
                        
                        # Ensure required keys
                        for key, default in {
                            "weight": 0.0,
                            "source_id": None,
                            "description": None,
                            "keywords": None,
                        }.items():
                            if key not in edge_dict:
                                edge_dict[key] = default
                        
                        edges_dict[(src, tgt)] = edge_dict
                
                return edges_dict

        return await asyncio.to_thread(_sync_get_edges_batch)

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges for a node."""

        def _sync_get_node_edges():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = f"""
                GO FROM '{source_node_id}' OVER * BIDIRECT 
                YIELD src(edge) as src, dst(edge) as dst
                """
                result = session.execute(query)
                
                edges = []
                if result.is_succeeded():
                    for row in result:
                        src = row.values()[0].as_string()
                        tgt = row.values()[1].as_string()
                        edges.append((src, tgt))
                
                return edges if edges else None

        return await asyncio.to_thread(_sync_get_node_edges)

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """Batch retrieve edges for multiple nodes."""

        def _sync_get_nodes_edges_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Build conditions
                conditions = []
                for node_id in node_ids:
                    conditions.append(f"id(n) == '{node_id}'")
                
                where_clause = " OR ".join(conditions)
                query = f"""
                MATCH (n:base)-[r]-(connected:base)
                WHERE {where_clause}
                RETURN id(n) as node_id, id(connected) as connected_id, src(edge) as edge_src
                """
                result = session.execute(query)
                
                edges_dict = {node_id: [] for node_id in node_ids}
                
                if result.is_succeeded():
                    for row in result:
                        node_id = row.values()[0].as_string()
                        connected_id = row.values()[1].as_string()
                        # For Nebula, we need to check the edge direction
                        # Here we assume DIRECTED edges, adjust as needed
                        edges_dict[node_id].append((node_id, connected_id))
                
                return edges_dict

        return await asyncio.to_thread(_sync_get_nodes_edges_batch)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((IOErrorException,)) if IOErrorException else ()
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the database."""

        def _sync_upsert_node():
            # Ensure entity_type exists for Nebula tag creation
            entity_type = node_data.get("entity_type", "base")
            if "entity_id" not in node_data:
                raise ValueError("Nebula: node properties must contain an 'entity_id' field")

            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Build property names and values separately for Nebula syntax
                prop_names = []
                prop_values = []
                
                for key, value in node_data.items():
                    if value is not None:
                        prop_names.append(key)
                        # Escape single quotes in string values
                        if isinstance(value, str):
                            escaped_value = value.replace("'", "\\'")
                            prop_values.append(f"'{escaped_value}'")
                        else:
                            prop_values.append(str(value))
                
                names_str = ", ".join(prop_names)
                values_str = ", ".join(prop_values)
                
                # Insert/Update vertex with base tag using correct Nebula syntax
                query = f"""
                INSERT VERTEX base({names_str}) 
                VALUES '{node_id}':({values_str})
                """
                result = session.execute(query)
                
                if not result.is_succeeded():
                    logger.error(f"Failed to upsert node {node_id}: {_safe_error_msg(result)}")
                    raise RuntimeError(f"Failed to upsert node: {_safe_error_msg(result)}")
                
                logger.debug(f"Upserted node with id '{node_id}'")

        return await asyncio.to_thread(_sync_upsert_node)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((IOErrorException,)) if IOErrorException else ()
    )
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """Upsert an edge between two nodes."""

        def _sync_upsert_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Build property names and values separately for Nebula syntax
                prop_names = []
                prop_values = []
                
                for key, value in edge_data.items():
                    if value is not None:
                        prop_names.append(key)
                        # Escape single quotes in string values
                        if isinstance(value, str):
                            escaped_value = value.replace("'", "\\'")
                            prop_values.append(f"'{escaped_value}'")
                        else:
                            prop_values.append(str(value))
                
                names_str = ", ".join(prop_names)
                values_str = ", ".join(prop_values)
                
                # Insert/Update edge using correct Nebula syntax
                query = f"""
                INSERT EDGE DIRECTED({names_str}) 
                VALUES '{source_node_id}' -> '{target_node_id}':({values_str})
                """
                result = session.execute(query)
                
                if not result.is_succeeded():
                    logger.error(f"Failed to upsert edge from {source_node_id} to {target_node_id}: {_safe_error_msg(result)}")
                    raise RuntimeError(f"Failed to upsert edge: {_safe_error_msg(result)}")
                
                logger.debug(f"Upserted edge from '{source_node_id}' to '{target_node_id}'")

        return await asyncio.to_thread(_sync_upsert_edge)

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
                    # Get all nodes
                    query = f"""
                    MATCH (n:base)
                    RETURN id(n) as id, properties(n) as props
                    LIMIT {max_nodes}
                    """
                    node_result = session.execute(query)
                    
                    if node_result.is_succeeded():
                        for row in node_result:
                            node_id = row.values()[0].as_string()
                            props = row.values()[1].as_map()
                            
                            node_dict = {}
                            for key, value in props.items():
                                key_str = key
                                if value.is_string():
                                    node_dict[key_str] = value.as_string()
                                elif value.is_int():
                                    node_dict[key_str] = value.as_int()
                                elif value.is_double():
                                    node_dict[key_str] = value.as_double()
                            
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=node_id,
                                    labels=[node_dict.get("entity_id", node_id)],
                                    properties=node_dict,
                                )
                            )
                            seen_nodes.add(node_id)
                    
                    # Get edges between these nodes
                    if seen_nodes:
                        edge_query = f"""
                        MATCH (a:base)-[r:DIRECTED]->(b:base)
                        WHERE id(a) IN [{', '.join([f"'{n}'" for n in seen_nodes])}] 
                          AND id(b) IN [{', '.join([f"'{n}'" for n in seen_nodes])}]
                        RETURN id(a) as src, id(b) as tgt, properties(r) as props
                        """
                        edge_result = session.execute(edge_query)
                        
                        if edge_result.is_succeeded():
                            for row in edge_result:
                                src = row.values()[0].as_string()
                                tgt = row.values()[1].as_string()
                                props = row.values()[2].as_map()
                                
                                edge_dict = {}
                                for key, value in props.items():
                                    key_str = key
                                    if value.is_string():
                                        edge_dict[key_str] = value.as_string()
                                    elif value.is_int():
                                        edge_dict[key_str] = value.as_int()
                                    elif value.is_double():
                                        edge_dict[key_str] = value.as_double()
                                
                                edge_id = f"{src}-{tgt}"
                                result.edges.append(
                                    KnowledgeGraphEdge(
                                        id=edge_id,
                                        type="DIRECTED",
                                        source=src,
                                        target=tgt,
                                        properties=edge_dict,
                                    )
                                )
                                seen_edges.add(edge_id)
                else:
                    # BFS from specific node
                    query = f"""
                    MATCH (start:base)-[*0..{max_depth}]-(end:base)
                    WHERE id(start) == '{node_label}'
                    RETURN DISTINCT id(end) as id, properties(end) as props
                    LIMIT {max_nodes}
                    """
                    node_result = session.execute(query)
                    
                    if node_result.is_succeeded():
                        for row in node_result:
                            node_id = row.values()[0].as_string()
                            props = row.values()[1].as_map()
                            
                            node_dict = {}
                            for key, value in props.items():
                                key_str = key
                                if value.is_string():
                                    node_dict[key_str] = value.as_string()
                                elif value.is_int():
                                    node_dict[key_str] = value.as_int()
                                elif value.is_double():
                                    node_dict[key_str] = value.as_double()
                            
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=node_id,
                                    labels=[node_dict.get("entity_id", node_id)],
                                    properties=node_dict,
                                )
                            )
                            seen_nodes.add(node_id)
                    
                    # Get edges in the subgraph
                    if seen_nodes:
                        edge_query = f"""
                        MATCH (a:base)-[r:DIRECTED]->(b:base)
                        WHERE id(a) IN [{', '.join([f"'{n}'" for n in seen_nodes])}] 
                          AND id(b) IN [{', '.join([f"'{n}'" for n in seen_nodes])}]
                        RETURN id(a) as src, id(b) as tgt, properties(r) as props
                        """
                        edge_result = session.execute(edge_query)
                        
                        if edge_result.is_succeeded():
                            for row in edge_result:
                                src = row.values()[0].as_string()
                                tgt = row.values()[1].as_string()
                                props = row.values()[2].as_map()
                                
                                edge_dict = {}
                                for key, value in props.items():
                                    key_str = key
                                    if value.is_string():
                                        edge_dict[key_str] = value.as_string()
                                    elif value.is_int():
                                        edge_dict[key_str] = value.as_int()
                                    elif value.is_double():
                                        edge_dict[key_str] = value.as_double()
                                
                                edge_id = f"{src}-{tgt}"
                                result.edges.append(
                                    KnowledgeGraphEdge(
                                        id=edge_id,
                                        type="DIRECTED",
                                        source=src,
                                        target=tgt,
                                        properties=edge_dict,
                                    )
                                )
                                seen_edges.add(edge_id)

            logger.info(f"Retrieved subgraph with {len(result.nodes)} nodes and {len(result.edges)} edges")
            return result

        return await asyncio.to_thread(_sync_get_knowledge_graph)

    async def get_all_labels(self) -> list[str]:
        """Get all node labels."""

        def _sync_get_all_labels():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = """
                FETCH PROP ON base * 
                YIELD properties(vertex).entity_id as label
                | WHERE $-.label IS NOT NULL 
                | YIELD DISTINCT $-.label 
                | ORDER BY $-.label
                """
                result = session.execute(query)
                
                labels = []
                if result.is_succeeded():
                    for row in result:
                        label = row.values()[0].as_string()
                        labels.append(label)
                
                return labels

        return await asyncio.to_thread(_sync_get_all_labels)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((IOErrorException,)) if IOErrorException else ()
    )
    async def delete_node(self, node_id: str) -> None:
        """Delete a node."""

        def _sync_delete_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = f"DELETE VERTEX '{node_id}' WITH EDGE"
                result = session.execute(query)
                
                if not result.is_succeeded():
                    logger.error(f"Failed to delete node {node_id}: {_safe_error_msg(result)}")
                    raise RuntimeError(f"Failed to delete node: {_safe_error_msg(result)}")
                
                logger.debug(f"Deleted node with id '{node_id}'")

        return await asyncio.to_thread(_sync_delete_node)

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes."""
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges."""

        def _sync_remove_edge(source: str, target: str):
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = f"DELETE EDGE DIRECTED '{source}' -> '{target}'"
                result = session.execute(query)
                
                if not result.is_succeeded():
                    logger.error(f"Failed to delete edge from {source} to {target}: {_safe_error_msg(result)}")
                else:
                    logger.debug(f"Deleted edge from '{source}' to '{target}'")

        for source, target in edges:
            await asyncio.to_thread(_sync_remove_edge, source, target)

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage."""

        def _sync_drop():
            with NebulaSyncConnectionManager.get_session() as session:
                query = f"DROP SPACE IF EXISTS {self._space_name}"
                result = session.execute(query)
                
                if result.is_succeeded():
                    logger.info(f"Dropped space {self._space_name}")
                    return {"status": "success", "message": "data dropped"}
                else:
                    logger.error(f"Failed to drop space {self._space_name}: {_safe_error_msg(result)}")
                    return {"status": "error", "message": _safe_error_msg(result)}

        return await asyncio.to_thread(_sync_drop)
