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

from nebula3.common import ttypes

from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph
from ..utils import logger

# Set nebula logger level to ERROR to suppress warning logs
logging.getLogger("nebula3").setLevel(logging.WARNING)


def _prepare_nebula_params(params: dict) -> dict:
    """Convert Python values to Nebula ttypes.Value objects."""
    nebula_params = {}
    for key, value in params.items():
        param_value = ttypes.Value()
        if isinstance(value, str):
            param_value.set_sVal(value)
        elif isinstance(value, int):
            param_value.set_iVal(value)
        elif isinstance(value, float):
            param_value.set_fVal(value)
        elif isinstance(value, bool):
            param_value.set_bVal(value)
        elif isinstance(value, list):
            # For list parameters, create NList
            value_list = []
            for item in value:
                item_value = ttypes.Value()
                if isinstance(item, str):
                    item_value.set_sVal(item)
                elif isinstance(item, int):
                    item_value.set_iVal(item)
                elif isinstance(item, float):
                    item_value.set_fVal(item)
                value_list.append(item_value)
            nlist = ttypes.NList(values=value_list)
            param_value.set_lVal(nlist)
        else:
            # Fallback: convert to string
            param_value.set_sVal(str(value))
        nebula_params[key] = param_value
    return nebula_params


def _quote_vid(vid: str) -> str:
    """
    Safely quote a VID for nGQL queries.
    Escape backslash and double quote, and wrap with double quotes.
    """
    escaped = vid.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _convert_nebula_value(value) -> any:
    """Convert a single Nebula Value to Python type."""
    if value.is_null():
        return None
    elif value.is_string():
        return value.as_string()
    elif value.is_int():
        return value.as_int()
    elif value.is_double():
        return value.as_double()
    elif value.is_bool():
        return value.as_bool()
    elif value.is_list():
        return [_convert_nebula_value(item) for item in value.as_list()]
    else:
        return str(value)  # 兜底转换


def _safe_error_msg(result) -> str:
    """Safely extract error message from Nebula result, handling UTF-8 decode errors."""
    try:
        error_code = result.error_code()

        # Try to get error message safely
        try:
            error_msg = result.error_msg()
        except Exception as msg_error:
            logger.warning(f"Failed to extract error message: {msg_error}")
            return f"Nebula operation failed (error code: {error_code})"

        # Handle the message based on its type
        if error_msg is None:
            return f"Nebula operation failed (error code: {error_code})"
        elif isinstance(error_msg, str):
            return f"Nebula error (code: {error_code}): {error_msg}"
        elif isinstance(error_msg, bytes):
            # Try different encodings safely
            decoded_msg = None
            for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
                try:
                    decoded_msg = error_msg.decode(encoding)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue

            # If all encodings fail, use safe replacement
            if decoded_msg is None:
                try:
                    decoded_msg = error_msg.decode("utf-8", errors="replace")
                except Exception:
                    decoded_msg = str(error_msg)

            return f"Nebula error (code: {error_code}): {decoded_msg}"
        else:
            return f"Nebula error (code: {error_code}): {str(error_msg)}"

    except Exception as e:
        logger.warning(f"Failed to process Nebula error: {e}")
        try:
            error_code = result.error_code()
            return f"Nebula operation failed (error code: {error_code})"
        except Exception:
            return "Nebula operation failed (unknown error)"


@final
@dataclass
class NebulaSyncStorage(BaseGraphStorage):
    """
    Nebula storage implementation using sync driver with async interface.
    This avoids event loop issues while maintaining compatibility with async code.

    Security Strategy (Mixed Approach):
    - Query Operations: Use MATCH syntax + parameterized queries (fully secure)
    - Modify Operations: Use nGQL syntax + _quote_vid (VID parameterization not supported)

    Nebula Graph supports parameterized queries in MATCH statements but not in VID positions
    for operations like DELETE, UPSERT, FETCH, GO. This is a known limitation.
    """

    def __init__(self, namespace, workspace, embedding_func=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            embedding_func=None,
        )
        self._space_name = None

    def _convert_nebula_value_map(self, value_map: dict) -> dict[str, any]:
        """统一的类型转换函数"""
        result = {}
        for key, value in value_map.items():
            result[key] = _convert_nebula_value(value)
        return result

    async def initialize(self):
        """Initialize storage and prepare database."""
        if NebulaSyncConnectionManager is None:
            raise RuntimeError("Nebula sync connection manager is not available")

        # Use optimized space preparation with reliable schema checking
        # max_wait=30s for balanced performance and reliability
        # fail_on_timeout=True ensures we get clear feedback if initialization fails
        self._space_name = await asyncio.to_thread(
            NebulaSyncConnectionManager.prepare_space, self.workspace, max_wait=30, fail_on_timeout=True
        )

        logger.debug(f"NebulaSyncStorage initialized for workspace '{self.workspace}', space '{self._space_name}'")

    async def finalize(self):
        """Clean up resources."""
        # Nothing to clean up - connection managed at worker level
        logger.debug(f"NebulaSyncStorage finalized for workspace '{self.workspace}'")

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists using MATCH syntax (Nebula supports both nGQL and Cypher-like syntax)."""

        def _sync_has_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Use MATCH syntax with parameterized query (Nebula supports this!)
                query = "MATCH (v) WHERE id(v) == $vid RETURN v LIMIT 1"
                params = {"vid": node_id}

                nebula_params = _prepare_nebula_params(params)
                result = session.execute_parameter(query, nebula_params)
                return result.is_succeeded() and result.row_size() > 0

        return await asyncio.to_thread(_sync_has_node)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes using parameterized MATCH query."""

        def _sync_has_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Use MATCH syntax with parameterized query instead of FETCH
                query = """
                MATCH (src)-[e:DIRECTED]-(dst) 
                WHERE (id(src) == $src_id AND id(dst) == $dst_id) 
                   OR (id(src) == $dst_id AND id(dst) == $src_id)
                RETURN e LIMIT 1
                """
                params = {"src_id": source_node_id, "dst_id": target_node_id}

                nebula_params = _prepare_nebula_params(params)
                result = session.execute_parameter(query, nebula_params)

                return result.is_succeeded() and result.row_size() > 0

        return await asyncio.to_thread(_sync_has_edge)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier using parameterized MATCH query."""

        def _sync_get_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Use MATCH syntax with parameterized query instead of FETCH
                query = "MATCH (v:base) WHERE id(v) == $node_id RETURN properties(v) as props"
                params = {"node_id": node_id}

                nebula_params = _prepare_nebula_params(params)
                result = session.execute_parameter(query, nebula_params)

                if result.is_succeeded() and result.row_size() > 0:
                    for row in result:
                        props = row.values()[0].as_map()
                        node_dict = self._convert_nebula_value_map(props)

                        # Add entity_id which is the node ID itself
                        node_dict["entity_id"] = node_id
                        return node_dict
                return None

        return await asyncio.to_thread(_sync_get_node)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """
        Optimized batch node retrieval using IN operator.
        Processes up to 100 nodes per batch for better performance.
        """

        def _sync_get_nodes_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                nodes = {}

                if not node_ids:
                    return nodes

                # Process in batches of 100
                batch_size = 100
                for i in range(0, len(node_ids), batch_size):
                    batch_ids = node_ids[i : i + batch_size]

                    # Use IN operator for batch query - much faster!
                    query = "MATCH (v:base) WHERE id(v) IN $node_ids RETURN id(v) as vid, properties(v) as props"
                    params = {"node_ids": batch_ids}

                    nebula_params = _prepare_nebula_params(params)
                    result = session.execute_parameter(query, nebula_params)

                    if result.is_succeeded():
                        for row in result:
                            node_id = row.values()[0].as_string()
                            props = row.values()[1].as_map()
                            node_dict = self._convert_nebula_value_map(props)
                            node_dict["entity_id"] = node_id
                            nodes[node_id] = node_dict

                return nodes

        return await asyncio.to_thread(_sync_get_nodes_batch)

    async def node_degree(self, node_id: str) -> int:
        """Get the degree of a node."""

        def _sync_node_degree():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Use MATCH syntax for more reliable degree calculation
                query = """
                MATCH (v)-[r]-(other) 
                WHERE id(v) == $node_id 
                RETURN COUNT(r) AS degree
                """
                nebula_params = _prepare_nebula_params({"node_id": node_id})
                result = session.execute_parameter(query, nebula_params)

                if result.is_succeeded() and result.row_size() > 0:
                    for row in result:
                        return row.values()[0].as_int()
                return 0

        return await asyncio.to_thread(_sync_node_degree)

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """
        Optimized batch degree calculation using batch queries.
        Processes up to 100 nodes per batch.
        """

        def _sync_node_degrees_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if not node_ids:
                    return {}

                degrees = {}
                batch_size = 100

                for i in range(0, len(node_ids), batch_size):
                    batch_ids = node_ids[i : i + batch_size]

                    # Use MATCH syntax for more reliable batch degree calculation
                    query = """
                    UNWIND $node_ids AS node_id 
                    MATCH (v)-[r]-(other) 
                    WHERE id(v) == node_id 
                    RETURN node_id, COUNT(r) AS degree
                    """
                    params = {"node_ids": batch_ids}
                    nebula_params = _prepare_nebula_params(params)
                    result = session.execute_parameter(query, nebula_params)

                    if result.is_succeeded():
                        for row in result:
                            node_id = row.values()[0].as_string()
                            degree = row.values()[1].as_int()
                            degrees[node_id] = degree

                    # Set degree 0 for nodes not found in the result
                    for node_id in batch_ids:
                        if node_id not in degrees:
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
        """Get edge properties between two nodes using parameterized MATCH query."""

        def _sync_get_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Use MATCH syntax with parameterized query instead of FETCH
                query = """
                MATCH (src)-[e:DIRECTED]-(dst) 
                WHERE (id(src) == $src_id AND id(dst) == $dst_id) 
                   OR (id(src) == $dst_id AND id(dst) == $src_id)
                RETURN properties(e) as props LIMIT 1
                """
                params = {"src_id": source_node_id, "dst_id": target_node_id}

                nebula_params = _prepare_nebula_params(params)
                result = session.execute_parameter(query, nebula_params)

                if result.is_succeeded() and result.row_size() > 0:
                    for row in result:
                        props = row.values()[0].as_map()
                        edge_dict = self._convert_nebula_value_map(props)

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
        """
        Optimized batch edge retrieval using batch queries.
        Processes up to 100 edge pairs per batch.
        """

        def _sync_get_edges_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                edges_dict = {}

                if not pairs:
                    return edges_dict

                # Initialize all edge pairs with default values
                for pair in pairs:
                    src, tgt = pair["src"], pair["tgt"]
                    edges_dict[(src, tgt)] = {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }

                # Process in batches of 100
                batch_size = 100
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i : i + batch_size]

                    # Build batch query using UNION ALL for better performance
                    union_queries = []
                    all_params = {}

                    for j, pair in enumerate(batch_pairs):
                        src, tgt = pair["src"], pair["tgt"]
                        src_param = f"src_{j}"
                        tgt_param = f"tgt_{j}"
                        src_return_param = f"src_return_{j}"
                        tgt_return_param = f"tgt_return_{j}"

                        union_queries.append(f"""
                        MATCH (src)-[e:DIRECTED]-(dst) 
                        WHERE (id(src) == ${src_param} AND id(dst) == ${tgt_param}) 
                           OR (id(src) == ${tgt_param} AND id(dst) == ${src_param})
                        RETURN ${src_return_param} as src_id, ${tgt_return_param} as tgt_id, properties(e) as props
                        """)

                        all_params[src_param] = src
                        all_params[tgt_param] = tgt
                        all_params[src_return_param] = src
                        all_params[tgt_return_param] = tgt

                    # Combine with UNION ALL
                    batch_query = " UNION ALL ".join(union_queries)

                    nebula_params = _prepare_nebula_params(all_params)
                    result = session.execute_parameter(batch_query, nebula_params)

                    if result.is_succeeded():
                        for row in result:
                            src_id = row.values()[0].as_string()
                            tgt_id = row.values()[1].as_string()
                            props = row.values()[2].as_map()

                            edge_dict = self._convert_nebula_value_map(props)

                            # Ensure required keys exist
                            for key, default in {
                                "weight": 0.0,
                                "source_id": None,
                                "description": None,
                                "keywords": None,
                            }.items():
                                if key not in edge_dict:
                                    edge_dict[key] = default

                            edges_dict[(src_id, tgt_id)] = edge_dict

                return edges_dict

        return await asyncio.to_thread(_sync_get_edges_batch)

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges for a node."""

        def _sync_get_node_edges():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = """
                MATCH (v)-[r]-(connected) 
                WHERE id(v) == $source_node_id 
                RETURN id(v) as src, id(connected) as dst
                """
                nebula_params = _prepare_nebula_params({"source_node_id": source_node_id})
                result = session.execute_parameter(query, nebula_params)

                edges = []
                edges_set = set()  # For deduplication to match Neo4j behavior
                if result.is_succeeded():
                    for row in result:
                        src = row.values()[0].as_string()
                        tgt = row.values()[1].as_string()
                        # Deduplicate bidirectional edges to match Neo4j behavior
                        if (tgt, src) not in edges_set:
                            edges.append((src, tgt))
                            edges_set.add((src, tgt))

                return edges if edges else None

        return await asyncio.to_thread(_sync_get_node_edges)

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """
        Optimized batch node edges retrieval using batch queries.
        Processes up to 100 nodes per batch.
        """

        def _sync_get_nodes_edges_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if not node_ids:
                    return {}

                edges_dict = {node_id: [] for node_id in node_ids}
                batch_size = 100

                for i in range(0, len(node_ids), batch_size):
                    batch_ids = node_ids[i : i + batch_size]

                    # Use MATCH syntax for more reliable batch edge retrieval
                    query = """
                    UNWIND $node_ids AS node_id
                    MATCH (v)-[r]-(connected) 
                    WHERE id(v) == node_id 
                    RETURN node_id, id(v) as src, id(connected) as dst
                    """
                    params = {"node_ids": batch_ids}
                    nebula_params = _prepare_nebula_params(params)
                    result = session.execute_parameter(query, nebula_params)

                    # Track edges per node for deduplication
                    node_edges_sets = {node_id: set() for node_id in batch_ids}

                    if result.is_succeeded():
                        for row in result:
                            source_node_id = row.values()[0].as_string()
                            src = row.values()[1].as_string()
                            dst = row.values()[2].as_string()

                            # Deduplicate bidirectional edges to match Neo4j behavior
                            if (dst, src) not in node_edges_sets[source_node_id]:
                                edges_dict[source_node_id].append((src, dst))
                                node_edges_sets[source_node_id].add((src, dst))

                return edges_dict

        return await asyncio.to_thread(_sync_get_nodes_edges_batch)

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the database."""

        def _sync_upsert_node():
            if "entity_id" not in node_data:
                raise ValueError("Nebula: node properties must contain an 'entity_id' field")

            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Build property names and parameter mapping for Nebula syntax
                prop_names = []
                param_dict = {"node_id": node_id}

                for key, value in node_data.items():
                    if value is not None:
                        prop_names.append(key)
                        param_dict[f"prop_{key}"] = value

                if not prop_names:
                    # No properties to insert
                    logger.warning(f"No properties to insert for node {node_id}")
                    return

                # Build Nebula UPSERT syntax for true upsert semantics (like Neo4j MERGE)
                # VID cannot be parameterized in UPSERT statements
                # Use safe VID quoting to handle special characters properly
                node_id_quoted = _quote_vid(node_id)

                # Build SET clause with parameterized values
                set_items = []
                for key in prop_names:
                    set_items.append(f"base.{key} = $prop_{key}")
                set_clause = ", ".join(set_items)

                query = f"UPSERT VERTEX {node_id_quoted} SET {set_clause}"

                # Remove node_id from params since VID is not parameterized
                param_dict_without_vid = {k: v for k, v in param_dict.items() if k != "node_id"}
                nebula_params = _prepare_nebula_params(param_dict_without_vid)
                result = session.execute_parameter(query, nebula_params)

                if not result.is_succeeded():
                    logger.error(f"Failed to upsert node {node_id}: {_safe_error_msg(result)}")
                    raise RuntimeError(f"Failed to upsert node: {_safe_error_msg(result)}")

                logger.debug(f"Upserted node with id '{node_id}'")

        return await asyncio.to_thread(_sync_upsert_node)

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """True UPSERT implementation using correct Nebula syntax."""

        def _sync_upsert_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Filter out None values
                valid_props = {k: v for k, v in edge_data.items() if v is not None}

                if not valid_props:
                    logger.warning(f"No valid properties to upsert for edge {source_node_id} -> {target_node_id}")
                    return

                source_quoted = _quote_vid(source_node_id)
                target_quoted = _quote_vid(target_node_id)

                # Build SET clauses with parameterized values
                set_clauses = []
                param_dict = {}

                for key, value in valid_props.items():
                    param_key = f"prop_{key}"
                    set_clauses.append(f"{key} = ${param_key}")
                    param_dict[param_key] = value

                set_clause = ", ".join(set_clauses)

                # Use correct UPSERT EDGE syntax: UPSERT EDGE "src" -> "dst" OF edge_type SET ...
                query = f"UPSERT EDGE {source_quoted} -> {target_quoted} OF DIRECTED SET {set_clause}"

                # Prepare Nebula parameters
                nebula_params = _prepare_nebula_params(param_dict)

                logger.debug(f"UPSERT edge query: {query}")
                logger.debug(f"UPSERT edge params: {list(param_dict.keys())}")

                result = session.execute_parameter(query, nebula_params)

                if not result.is_succeeded():
                    logger.error(
                        f"Failed to upsert edge from {source_node_id} to {target_node_id}: {_safe_error_msg(result)}"
                    )
                    raise RuntimeError(f"Failed to upsert edge: {_safe_error_msg(result)}")

                logger.debug(f"Successfully upserted edge: '{source_node_id}' -> '{target_node_id}'")

        return await asyncio.to_thread(_sync_upsert_edge)

    def _sync_check_node_exists(self, node_id: str, session) -> bool:
        """Synchronous helper to check if a node exists using parameterized query."""
        query = "MATCH (v:base) WHERE id(v) == $node_id RETURN v LIMIT 1"
        params = {"node_id": node_id}

        nebula_params = _prepare_nebula_params(params)
        result = session.execute_parameter(query, nebula_params)
        return result.is_succeeded() and result.row_size() > 0

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """This function is not used"""
        """Don't implement it"""
        raise NotImplementedError

    async def get_all_labels(self) -> list[str]:
        """Get all node labels."""

        def _sync_get_all_labels():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Use simple LOOKUP syntax that actually works
                query = "LOOKUP ON base YIELD properties(vertex).entity_id as label"
                result = session.execute(query)

                labels = []
                if result.is_succeeded():
                    for row in result:
                        label = row.values()[0].as_string()
                        if label:  # Ensure label is not empty
                            labels.append(label)

                return list(set(labels))  # Remove duplicates and return

        return await asyncio.to_thread(_sync_get_all_labels)

    async def delete_node(self, node_id: str) -> None:
        """
        Delete a node using nGQL syntax.
        VID parameters are not supported in Nebula, so we must use _quote_vid.
        """

        def _sync_delete_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # VID cannot be parameterized in DELETE statements
                # Use safe VID quoting to handle special characters properly
                node_id_quoted = _quote_vid(node_id)
                query = f"DELETE VERTEX {node_id_quoted} WITH EDGE"
                result = session.execute(query)

                if not result.is_succeeded():
                    logger.error(f"Failed to delete node {node_id}: {_safe_error_msg(result)}")
                    raise RuntimeError(f"Failed to delete node: {_safe_error_msg(result)}")

                logger.debug(f"Deleted node with id '{node_id}'")

        return await asyncio.to_thread(_sync_delete_node)

    async def remove_nodes(self, nodes: list[str]):
        """
        Batch node deletion with small batch size to avoid query plan tree depth limit.
        Processes up to 10 nodes per batch to prevent Nebula optimization stage errors.
        """

        def _sync_remove_nodes_batch(batch_nodes: list[str]):
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if not batch_nodes:
                    return

                # For very small batches, use individual deletes to avoid query plan depth issues
                for node_id in batch_nodes:
                    node_id_quoted = _quote_vid(node_id)
                    query = f"DELETE VERTEX {node_id_quoted} WITH EDGE"
                    result = session.execute(query)

                    if not result.is_succeeded():
                        logger.error(f"Failed to delete node {node_id}: {_safe_error_msg(result)}")
                        # Continue with other nodes instead of raising exception
                    else:
                        logger.debug(f"Successfully deleted node {node_id}")

                logger.debug(f"Processed deletion of {len(batch_nodes)} nodes")

        # Process in very small batches of 10 to avoid query plan tree depth limit
        batch_size = 10
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            await asyncio.to_thread(_sync_remove_nodes_batch, batch)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """
        Batch edge deletion with small batch size to avoid query plan tree depth limit.
        Processes up to 10 edges per batch to prevent Nebula optimization stage errors.
        """

        def _sync_remove_edges_batch(batch_edges: list[tuple[str, str]]):
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if not batch_edges:
                    return

                # For very small batches, use individual deletes to avoid query plan depth issues
                for source, target in batch_edges:
                    source_quoted = _quote_vid(source)
                    target_quoted = _quote_vid(target)
                    query = f"DELETE EDGE DIRECTED {source_quoted} -> {target_quoted}"
                    result = session.execute(query)

                    if not result.is_succeeded():
                        logger.error(f"Failed to delete edge {source} -> {target}: {_safe_error_msg(result)}")
                        # Continue with other edges instead of raising exception
                    else:
                        logger.debug(f"Successfully deleted edge {source} -> {target}")

                logger.debug(f"Processed deletion of {len(batch_edges)} edges")

        # Process in very small batches of 10 to avoid query plan tree depth limit
        batch_size = 10
        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]
            await asyncio.to_thread(_sync_remove_edges_batch, batch)

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage."""

        def _sync_drop():
            with NebulaSyncConnectionManager.get_session() as session:
                # DROP SPACE doesn't support parameterized space names
                query = f"DROP SPACE IF EXISTS {self._space_name}"
                result = session.execute(query)

                if result.is_succeeded():
                    logger.info(f"Dropped space {self._space_name}")
                    return {"status": "success", "message": "data dropped"}
                else:
                    logger.error(f"Failed to drop space {self._space_name}: {_safe_error_msg(result)}")
                    return {"status": "error", "message": _safe_error_msg(result)}

        return await asyncio.to_thread(_sync_drop)
