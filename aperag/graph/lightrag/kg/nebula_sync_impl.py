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

from nebula3.Exception import IOErrorException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger

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
                        node_dict["entity_id"] = node_id
                        return node_dict
                return None

        return await asyncio.to_thread(_sync_get_node)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieve multiple nodes in one query."""

        def _sync_get_nodes_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Build the list of IDs for the query
                id_list = ", ".join([f"'{node_id}'" for node_id in node_ids])
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
                        node_dict["entity_id"] = node_id
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
        """Retrieve degrees for multiple nodes, optimized for Nebula characteristics."""

        def _sync_node_degrees_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if not node_ids:
                    return {}

                degrees = {node_id: 0 for node_id in node_ids}

                # 提高批次大小，利用fallback机制确保稳定性
                batch_size = 25  # 增加批次大小，提高效率

                for i in range(0, len(node_ids), batch_size):
                    batch = node_ids[i : i + batch_size]

                    # 使用简洁的UNION语法，增加UNION容量
                    union_queries = []
                    for node_id in batch:
                        # 使用管道语法计算度数，更符合Nebula习惯
                        union_queries.append(f"""
                        GO FROM '{node_id}' OVER * BIDIRECT 
                        YIELD '{node_id}' as node_id, count(*) as degree
                        """)

                    # 增加UNION查询数量限制，提高批量效率
                    if len(union_queries) <= 15:  # 从5个增加到15个
                        query = " UNION ".join(union_queries)
                        result = session.execute(query)

                        if result.is_succeeded():
                            found_nodes = set()
                            for row in result:
                                node_id = row.values()[0].as_string()
                                degree = row.values()[1].as_int()
                                degrees[node_id] = degree
                                found_nodes.add(node_id)

                            # 处理未找到的节点（度数为0）
                            for node_id in batch:
                                if node_id not in found_nodes:
                                    degrees[node_id] = 0
                        else:
                            logger.debug(f"UNION degree query failed: {_safe_error_msg(result)}")
                            # 降级为单个查询
                            for node_id in batch:
                                single_query = f"""
                                GO FROM '{node_id}' OVER * BIDIRECT 
                                YIELD count(*) as degree
                                """
                                single_result = session.execute(single_query)

                                if single_result.is_succeeded() and single_result.row_size() > 0:
                                    for row in single_result:
                                        degrees[node_id] = row.values()[0].as_int()
                                        break
                                else:
                                    degrees[node_id] = 0
                    else:
                        # 批次过大，直接使用单个查询避免复杂UNION
                        for node_id in batch:
                            try:
                                query = f"""
                                GO FROM '{node_id}' OVER * BIDIRECT 
                                YIELD count(*) as degree
                                """
                                result = session.execute(query)

                                if result.is_succeeded() and result.row_size() > 0:
                                    for row in result:
                                        degrees[node_id] = row.values()[0].as_int()
                                        break
                                else:
                                    degrees[node_id] = 0
                            except Exception as e:
                                logger.warning(f"Failed to get degree for node '{node_id}': {e}")
                                degrees[node_id] = 0

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
                # Try direction: source -> target
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

                # Try reverse direction: target -> source
                query = f"""
                FETCH PROP ON DIRECTED '{target_node_id}' -> '{source_node_id}' 
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
        """Retrieve edge properties for multiple pairs with optimized batch query."""

        def _sync_get_edges_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                edges_dict = {}

                if not pairs:
                    return edges_dict

                # Initialize all pairs with default values
                for pair in pairs:
                    src, tgt = pair["src"], pair["tgt"]
                    edges_dict[(src, tgt)] = {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }

                # Optimized: Use batch processing with fewer queries
                chunk_size = 50  # 从25增加到50，提高批量效率
                for i in range(0, len(pairs), chunk_size):
                    chunk = pairs[i : i + chunk_size]

                    # Create UNION query for batch edge processing (both directions)
                    union_queries = []
                    for pair in chunk:
                        src, tgt = pair["src"], pair["tgt"]
                        # Query both directions in a single UNION
                        union_queries.append(f"""
                        FETCH PROP ON DIRECTED '{src}' -> '{tgt}' 
                        YIELD '{src}' as src, '{tgt}' as tgt, properties(edge) as props
                        """)
                        union_queries.append(f"""
                        FETCH PROP ON DIRECTED '{tgt}' -> '{src}' 
                        YIELD '{tgt}' as reverse_src, '{src}' as reverse_tgt, properties(edge) as props
                        """)

                    query = " UNION ".join(union_queries)
                    result = session.execute(query)

                    if result.is_succeeded():
                        for row in result:
                            edge_src = row.values()[0].as_string()
                            edge_tgt = row.values()[1].as_string()
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

                            # Find the original query pair and update it
                            for pair in chunk:
                                src, tgt = pair["src"], pair["tgt"]
                                if (edge_src == src and edge_tgt == tgt) or (edge_src == tgt and edge_tgt == src):
                                    edges_dict[(src, tgt)] = edge_dict
                                    break
                    else:
                        logger.warning(f"Batch edge query chunk failed: {_safe_error_msg(result)}")
                        # Fall back to individual queries for this chunk
                        for pair in chunk:
                            src, tgt = pair["src"], pair["tgt"]

                            # Try direction: src -> tgt
                            query = f"""
                            FETCH PROP ON DIRECTED '{src}' -> '{tgt}' 
                            YIELD properties(edge) as props
                            """
                            result = session.execute(query)

                            edge_found = False
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
                                    edge_found = True
                                    break

                            # If not found, try reverse direction: tgt -> src
                            if not edge_found:
                                query = f"""
                                FETCH PROP ON DIRECTED '{tgt}' -> '{src}' 
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
                                        break

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
        """Batch retrieve edges for multiple nodes with optimized batch query."""

        def _sync_get_nodes_edges_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if not node_ids:
                    return {}

                edges_dict = {node_id: [] for node_id in node_ids}

                # Optimized: Use batch processing with fewer queries
                chunk_size = 60  # 从30增加到60，提高批量效率
                for i in range(0, len(node_ids), chunk_size):
                    chunk = node_ids[i : i + chunk_size]

                    # Create UNION query for batch edge processing
                    union_queries = []
                    for node_id in chunk:
                        union_queries.append(f"""
                        GO FROM '{node_id}' OVER * BIDIRECT 
                        YIELD '{node_id}' as query_node, src(edge) as src, dst(edge) as dst
                        """)

                    query = " UNION ".join(union_queries)
                    result = session.execute(query)

                    if result.is_succeeded():
                        for row in result:
                            query_node = row.values()[0].as_string()
                            src = row.values()[1].as_string()
                            dst = row.values()[2].as_string()
                            edges_dict[query_node].append((src, dst))
                    else:
                        logger.warning(f"Batch node edges query chunk failed: {_safe_error_msg(result)}")
                        # Fall back to individual queries for this chunk
                        for node_id in chunk:
                            query = f"""
                            GO FROM '{node_id}' OVER * BIDIRECT 
                            YIELD src(edge) as src, dst(edge) as dst
                            """
                            result = session.execute(query)

                            if result.is_succeeded():
                                for row in result:
                                    src = row.values()[0].as_string()
                                    dst = row.values()[1].as_string()
                                    edges_dict[node_id].append((src, dst))

                return edges_dict

        return await asyncio.to_thread(_sync_get_nodes_edges_batch)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((IOErrorException,)) if IOErrorException else (),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the database."""

        def _sync_upsert_node():
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
        retry=retry_if_exception_type((IOErrorException,)) if IOErrorException else (),
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
                    logger.error(
                        f"Failed to upsert edge from {source_node_id} to {target_node_id}: {_safe_error_msg(result)}"
                    )
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
                    # Get all nodes using LOOKUP
                    query = f"""
                    LOOKUP ON base 
                    YIELD id(vertex) as id, properties(vertex) as props
                    | LIMIT {max_nodes}
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

                    # Optimized: Get edges between nodes using batch query instead of N² individual queries
                    if seen_nodes:
                        nodes_list = list(seen_nodes)

                        # Create optimized batch query for all edges between these nodes
                        union_queries = []
                        for src in nodes_list:
                            union_queries.append(f"""
                            GO FROM '{src}' OVER DIRECTED 
                            WHERE dst(edge) IN [{", ".join([f"'{n}'" for n in nodes_list])}]
                            YIELD src(edge) as src, dst(edge) as dst, properties(edge) as props
                            """)

                        if union_queries:
                            # Process in chunks to avoid query size limits
                            chunk_size = 50  # 从20增加到50，提高批量效率
                            for i in range(0, len(union_queries), chunk_size):
                                chunk_queries = union_queries[i : i + chunk_size]
                                query = " UNION ".join(chunk_queries)
                                edge_result = session.execute(query)

                                if edge_result.is_succeeded():
                                    for row in edge_result:
                                        edge_src = row.values()[0].as_string()
                                        edge_dst = row.values()[1].as_string()
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

                                        edge_id = f"{edge_src}-{edge_dst}"
                                        if edge_id not in seen_edges:
                                            result.edges.append(
                                                KnowledgeGraphEdge(
                                                    id=edge_id,
                                                    type="DIRECTED",
                                                    source=edge_src,
                                                    target=edge_dst,
                                                    properties=edge_dict,
                                                )
                                            )
                                            seen_edges.add(edge_id)
                                else:
                                    logger.warning("Batch edge query failed, falling back to individual queries")
                                    # Fall back to original approach only if batch fails
                                    for src in nodes_list[i * chunk_size // 20 : (i + 1) * chunk_size // 20]:
                                        for tgt in seen_nodes:
                                            if src != tgt:  # Avoid self-loops
                                                edge_query = f"""
                                                FETCH PROP ON DIRECTED '{src}' -> '{tgt}' 
                                                YIELD src(edge) as src, dst(edge) as dst, properties(edge) as props
                                                """
                                                edge_result = session.execute(edge_query)

                                                if edge_result.is_succeeded() and edge_result.row_size() > 0:
                                                    for row in edge_result:
                                                        edge_src = row.values()[0].as_string()
                                                        edge_dst = row.values()[1].as_string()
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

                                                        edge_id = f"{edge_src}-{edge_dst}"
                                                        if edge_id not in seen_edges:
                                                            result.edges.append(
                                                                KnowledgeGraphEdge(
                                                                    id=edge_id,
                                                                    type="DIRECTED",
                                                                    source=edge_src,
                                                                    target=edge_dst,
                                                                    properties=edge_dict,
                                                                )
                                                            )
                                                            seen_edges.add(edge_id)
                else:
                    # BFS from specific node using GO FROM
                    from collections import deque

                    # Start with the specific node
                    queue = deque([node_label])
                    visited = set()
                    current_depth = 0

                    while queue and current_depth <= max_depth and len(seen_nodes) < max_nodes:
                        level_size = len(queue)
                        next_level_nodes = set()

                        for _ in range(level_size):
                            if not queue or len(seen_nodes) >= max_nodes:
                                break

                            current_node = queue.popleft()
                            if current_node in visited:
                                continue

                            visited.add(current_node)

                            # Get node properties
                            node_query = f"FETCH PROP ON base '{current_node}' YIELD properties(vertex) as props"
                            node_result = session.execute(node_query)

                            if node_result.is_succeeded() and node_result.row_size() > 0:
                                for row in node_result:
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

                                    node_dict["entity_id"] = current_node
                                    result.nodes.append(
                                        KnowledgeGraphNode(
                                            id=current_node,
                                            labels=[node_dict.get("entity_id", current_node)],
                                            properties=node_dict,
                                        )
                                    )
                                    seen_nodes.add(current_node)
                                    break

                            # Get neighbors if not at max depth
                            if current_depth < max_depth:
                                neighbor_query = f"""
                                GO FROM '{current_node}' OVER * BIDIRECT 
                                YIELD src(edge) as src, dst(edge) as dst
                                """
                                neighbor_result = session.execute(neighbor_query)

                                if neighbor_result.is_succeeded():
                                    for row in neighbor_result:
                                        src = row.values()[0].as_string()
                                        dst = row.values()[1].as_string()

                                        # Add edge to result
                                        edge_id = f"{src}-{dst}"
                                        if edge_id not in seen_edges:
                                            # Get edge properties
                                            edge_props_query = f"""
                                            FETCH PROP ON DIRECTED '{src}' -> '{dst}' 
                                            YIELD properties(edge) as props
                                            """
                                            edge_props_result = session.execute(edge_props_query)

                                            edge_dict = {}
                                            if edge_props_result.is_succeeded() and edge_props_result.row_size() > 0:
                                                for edge_row in edge_props_result:
                                                    props = edge_row.values()[0].as_map()
                                                    for key, value in props.items():
                                                        key_str = key
                                                        if value.is_string():
                                                            edge_dict[key_str] = value.as_string()
                                                        elif value.is_int():
                                                            edge_dict[key_str] = value.as_int()
                                                        elif value.is_double():
                                                            edge_dict[key_str] = value.as_double()
                                                    break

                                            result.edges.append(
                                                KnowledgeGraphEdge(
                                                    id=edge_id,
                                                    type="DIRECTED",
                                                    source=src,
                                                    target=dst,
                                                    properties=edge_dict,
                                                )
                                            )
                                            seen_edges.add(edge_id)

                                        # Add neighbor to next level
                                        neighbor = dst if src == current_node else src
                                        if neighbor not in visited and neighbor not in next_level_nodes:
                                            next_level_nodes.add(neighbor)

                        # Add next level nodes to queue
                        for node in next_level_nodes:
                            if len(seen_nodes) < max_nodes:
                                queue.append(node)

                        current_depth += 1

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
        retry=retry_if_exception_type((IOErrorException,)) if IOErrorException else (),
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
