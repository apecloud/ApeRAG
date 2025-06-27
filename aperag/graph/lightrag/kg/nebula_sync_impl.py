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
                # VID cannot be parameterized in FETCH statements
                # Use safe VID quoting to handle special characters properly
                node_id_quoted = _quote_vid(node_id)
                query = f"FETCH PROP ON base {node_id_quoted} YIELD properties(vertex)"
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
                # VID cannot be parameterized in FETCH statements
                source_quoted = _quote_vid(source_node_id)
                target_quoted = _quote_vid(target_node_id)
                query = f"""
                FETCH PROP ON DIRECTED {source_quoted} -> {target_quoted}
                YIELD properties(edge) as props
                """
                result = session.execute(query)
                if result.is_succeeded() and result.row_size() > 0:
                    return True

                # Try reverse direction - correct the variable assignment
                # VID cannot be parameterized in FETCH statements
                source_quoted_rev = _quote_vid(target_node_id)  # Now target becomes source
                target_quoted_rev = _quote_vid(source_node_id)  # Now source becomes target
                query = f"""
                FETCH PROP ON DIRECTED {source_quoted_rev} -> {target_quoted_rev}
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
                # VID cannot be parameterized in FETCH statements
                # Use safe VID quoting to handle special characters properly
                node_id_quoted = _quote_vid(node_id)
                query = f"FETCH PROP ON base {node_id_quoted} YIELD properties(vertex) as props"
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
        """使用参数化查询的高效批量节点查询"""

        def _sync_get_nodes_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                nodes = {}

                if not node_ids:
                    return nodes

                # Build FETCH query with explicit node IDs (FETCH doesn't support parameterized lists)
                # Use safe VID quoting to handle special characters properly
                node_ids_str = ", ".join([_quote_vid(node_id) for node_id in node_ids])
                query = f"FETCH PROP ON base {node_ids_str} YIELD id(vertex) as id, properties(vertex) as props"
                result = session.execute(query)

                if result.is_succeeded():
                    for row in result:
                        node_id = row.values()[0].as_string()
                        props = row.values()[1].as_map()
                        node_dict = {}
                        for key, value in props.items():
                            if value.is_string():
                                node_dict[key] = value.as_string()
                            elif value.is_int():
                                node_dict[key] = value.as_int()
                            elif value.is_double():
                                node_dict[key] = value.as_double()

                        node_dict["entity_id"] = node_id
                        nodes[node_id] = node_dict

                return nodes

        return await asyncio.to_thread(_sync_get_nodes_batch)

    async def node_degree(self, node_id: str) -> int:
        """Get the degree of a node."""

        def _sync_node_degree():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Use proper nGQL aggregation syntax with GROUP BY
                query = """
                GO FROM $node_id OVER * BIDIRECT 
                YIELD dst(edge) AS n
                | GROUP BY $-.n YIELD COUNT(*) AS degree
                """
                nebula_params = _prepare_nebula_params({"node_id": node_id})
                result = session.execute_parameter(query, nebula_params)

                if result.is_succeeded() and result.row_size() > 0:
                    for row in result:
                        return row.values()[0].as_int()
                return 0

        return await asyncio.to_thread(_sync_node_degree)

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """使用简单查询的批量度数查询"""

        def _sync_node_degrees_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if not node_ids:
                    return {}

                degrees = {}

                # Use simple individual queries (complex UNWIND aggregation is problematic)
                for node_id in node_ids:
                    # Use proper nGQL aggregation syntax with GROUP BY
                    query = """
                    GO FROM $node_id OVER * BIDIRECT 
                    YIELD dst(edge) AS n
                    | GROUP BY $-.n YIELD COUNT(*) AS degree
                    """
                    nebula_params = _prepare_nebula_params({"node_id": node_id})
                    result = session.execute_parameter(query, nebula_params)

                    if result.is_succeeded() and result.row_size() > 0:
                        for row in result:
                            degree = row.values()[0].as_int()
                            degrees[node_id] = degree
                            break  # Only need the first (and should be only) row
                    else:
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
                # VID cannot be parameterized in FETCH statements
                source_quoted = _quote_vid(source_node_id)
                target_quoted = _quote_vid(target_node_id)
                query = f"""
                FETCH PROP ON DIRECTED {source_quoted} -> {target_quoted}
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

                # Try reverse direction: target -> source - fix variable assignment
                # VID cannot be parameterized in FETCH statements
                source_quoted_rev = _quote_vid(target_node_id)  # Now target becomes source
                target_quoted_rev = _quote_vid(source_node_id)  # Now source becomes target
                query = f"""
                FETCH PROP ON DIRECTED {source_quoted_rev} -> {target_quoted_rev}
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
        """使用简单查询的批量边查询"""

        def _sync_get_edges_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                edges_dict = {}

                if not pairs:
                    return edges_dict

                # 初始化所有边对
                for pair in pairs:
                    src, tgt = pair["src"], pair["tgt"]
                    edges_dict[(src, tgt)] = {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }

                # Use individual FETCH queries for each edge pair (both directions)
                for pair in pairs:
                    src, tgt = pair["src"], pair["tgt"]

                    # Try both directions for each pair
                    for direction_src, direction_tgt in [(src, tgt), (tgt, src)]:
                        # VID cannot be parameterized in FETCH statements
                        # Use safe VID quoting to handle special characters properly
                        direction_src_quoted = _quote_vid(direction_src)
                        direction_tgt_quoted = _quote_vid(direction_tgt)
                        query = f"""
                        FETCH PROP ON DIRECTED {direction_src_quoted} -> {direction_tgt_quoted}
                        YIELD src(edge) as src, dst(edge) as dst, properties(edge) as props
                        """
                        result = session.execute(query)

                        if result.is_succeeded() and result.row_size() > 0:
                            for row in result:
                                _edge_src = row.values()[0].as_string()  # Not used but required for structure
                                _edge_dst = row.values()[1].as_string()  # Not used but required for structure
                                props = row.values()[2].as_map()

                                edge_dict = {}
                                for key, value in props.items():
                                    if value.is_string():
                                        edge_dict[key] = value.as_string()
                                    elif value.is_int():
                                        edge_dict[key] = value.as_int()
                                    elif value.is_double():
                                        edge_dict[key] = value.as_double()

                                # 确保必需的键存在
                                for key, default in {
                                    "weight": 0.0,
                                    "source_id": None,
                                    "description": None,
                                    "keywords": None,
                                }.items():
                                    if key not in edge_dict:
                                        edge_dict[key] = default

                                # Update the original pair's result
                                edges_dict[(src, tgt)] = edge_dict
                                break  # Found edge, no need to check more rows

                return edges_dict

        return await asyncio.to_thread(_sync_get_edges_batch)

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges for a node."""

        def _sync_get_node_edges():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = """
                GO FROM $source_node_id OVER * BIDIRECT 
                YIELD src(edge) as src, dst(edge) as dst
                """
                nebula_params = _prepare_nebula_params({"source_node_id": source_node_id})
                result = session.execute_parameter(query, nebula_params)

                edges = []
                edges_set = set()  # For deduplication to match Neo4j behavior
                if result.is_succeeded():
                    for row in result:
                        src = row.values()[0].as_string()
                        tgt = row.values()[1].as_string()
                        # Deduplicate bidirectional edges (BIDIRECT returns both A->B and B->A)
                        # Neo4j only returns one direction, so we match that behavior
                        if (tgt, src) not in edges_set:
                            edges.append((src, tgt))
                            edges_set.add((src, tgt))

                return edges if edges else None

        return await asyncio.to_thread(_sync_get_node_edges)

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """使用简单查询的批量邻居查询"""

        def _sync_get_nodes_edges_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if not node_ids:
                    return {}

                edges_dict = {node_id: [] for node_id in node_ids}

                # Use individual GO queries for each node
                for node_id in node_ids:
                    query = """
                    GO FROM $node_id OVER * BIDIRECT 
                    YIELD src(edge) as src, dst(edge) as dst
                    """
                    nebula_params = _prepare_nebula_params({"node_id": node_id})
                    result = session.execute_parameter(query, nebula_params)

                    edges_set = set()  # For deduplication to match Neo4j behavior
                    if result.is_succeeded():
                        for row in result:
                            src = row.values()[0].as_string()
                            dst = row.values()[1].as_string()
                            # Deduplicate bidirectional edges (BIDIRECT returns both A->B and B->A)
                            # Neo4j only returns one direction, so we match that behavior
                            if (dst, src) not in edges_set:
                                edges_dict[node_id].append((src, dst))
                                edges_set.add((src, dst))

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((IOErrorException,)) if IOErrorException else (),
    )
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """Upsert an edge between two nodes."""

        def _sync_upsert_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # First, verify that both nodes exist
                source_exists = self._sync_check_node_exists(source_node_id, session)
                target_exists = self._sync_check_node_exists(target_node_id, session)
                
                if not source_exists:
                    logger.error(f"Source node does not exist: {source_node_id}")
                    raise RuntimeError(f"Source node does not exist: {source_node_id}")
                    
                if not target_exists:
                    logger.error(f"Target node does not exist: {target_node_id}")
                    raise RuntimeError(f"Target node does not exist: {target_node_id}")

                # Build property names and parameter mapping for Nebula syntax
                prop_names = []
                param_dict = {"source_node_id": source_node_id, "target_node_id": target_node_id}

                for key, value in edge_data.items():
                    if value is not None:
                        prop_names.append(key)
                        param_dict[f"prop_{key}"] = value

                if not prop_names:
                    # No properties to insert
                    logger.warning(f"No properties to insert for edge {source_node_id} -> {target_node_id}")
                    return

                # Use INSERT EDGE with ON DUPLICATE UPDATE for upsert semantics
                # VID cannot be parameterized in INSERT statements  
                # Use safe VID quoting to handle special characters properly
                source_quoted = _quote_vid(source_node_id)
                target_quoted = _quote_vid(target_node_id)

                # Build property value list - avoid parameterization for now
                prop_values = []
                for key in prop_names:
                    value = edge_data[key]
                    if isinstance(value, str):
                        # Escape special characters in string values for Nebula
                        escaped_value = value.replace("\\", "\\\\")  # Escape backslashes first
                        escaped_value = escaped_value.replace("'", "\\'")  # Escape single quotes
                        escaped_value = escaped_value.replace("\n", "\\n")  # Escape newlines
                        escaped_value = escaped_value.replace("\r", "\\r")  # Escape carriage returns
                        escaped_value = escaped_value.replace("\t", "\\t")  # Escape tabs
                        prop_values.append(f"'{escaped_value}'")
                    elif isinstance(value, (int, float)):
                        prop_values.append(str(value))
                    else:
                        # Convert to string and escape special characters
                        str_value = str(value)
                        escaped_value = str_value.replace("\\", "\\\\")  # Escape backslashes first
                        escaped_value = escaped_value.replace("'", "\\'")  # Escape single quotes
                        escaped_value = escaped_value.replace("\n", "\\n")  # Escape newlines
                        escaped_value = escaped_value.replace("\r", "\\r")  # Escape carriage returns
                        escaped_value = escaped_value.replace("\t", "\\t")  # Escape tabs
                        prop_values.append(f"'{escaped_value}'")

                # Build property names and values for INSERT
                prop_names_str = "(" + ", ".join(prop_names) + ")"
                prop_values_str = "(" + ", ".join(prop_values) + ")"

                query = f"INSERT EDGE DIRECTED{prop_names_str} VALUES {source_quoted} -> {target_quoted}:{prop_values_str}"

                # No parameters needed since we're not using parameterization
                nebula_params = {}
                
                # Debug logging
                logger.debug(f"Insert edge query: {query}")
                
                result = session.execute(query)

                if not result.is_succeeded():
                    logger.error(
                        f"Failed to upsert edge from {source_node_id} to {target_node_id}: {_safe_error_msg(result)}"
                    )
                    raise RuntimeError(f"Failed to upsert edge: {_safe_error_msg(result)}")

                logger.debug(f"Upserted edge from '{source_node_id}' to '{target_node_id}'")

        return await asyncio.to_thread(_sync_upsert_edge)
    
    def _sync_check_node_exists(self, node_id: str, session) -> bool:
        """Synchronous helper to check if a node exists."""
        node_id_quoted = _quote_vid(node_id)
        query = f"FETCH PROP ON base {node_id_quoted} YIELD properties(vertex)"
        result = session.execute(query)
        return result.is_succeeded() and result.row_size() > 0

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
                    # Get all nodes using LOOKUP (LIMIT doesn't support parameters)
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

                    # Get edges between nodes using individual queries
                    if seen_nodes:
                        nodes_list = list(seen_nodes)

                        # Use individual GO queries for each node to find edges
                        for src_node in nodes_list:
                            query = """
                            GO FROM $src_node OVER DIRECTED 
                            YIELD src(edge) as src, dst(edge) as dst, properties(edge) as props
                            """
                            nebula_params = _prepare_nebula_params({"src_node": src_node})
                            edge_result = session.execute_parameter(query, nebula_params)

                            if edge_result.is_succeeded():
                                for row in edge_result:
                                    edge_src = row.values()[0].as_string()
                                    edge_dst = row.values()[1].as_string()
                                    props = row.values()[2].as_map()

                                    # Only include edges where destination is also in our node set
                                    if edge_dst in seen_nodes:
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
                    # BFS from specific node using parameterized queries
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

                            # Get node properties using quoted VID
                            # VID cannot be parameterized in FETCH statements
                            # Use safe VID quoting to handle special characters properly
                            current_node_quoted = _quote_vid(current_node)
                            node_query = f"FETCH PROP ON base {current_node_quoted} YIELD properties(vertex) as props"
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
                                neighbor_query = """
                                GO FROM $current_node OVER * BIDIRECT 
                                YIELD src(edge) as src, dst(edge) as dst, properties(edge) as props
                                """
                                nebula_params = _prepare_nebula_params({"current_node": current_node})
                                neighbor_result = session.execute_parameter(neighbor_query, nebula_params)

                                if neighbor_result.is_succeeded():
                                    for row in neighbor_result:
                                        src = row.values()[0].as_string()
                                        dst = row.values()[1].as_string()
                                        props = row.values()[2].as_map()

                                        # Add edge to result
                                        edge_id = f"{src}-{dst}"
                                        if edge_id not in seen_edges:
                                            edge_dict = {}
                                            for key, value in props.items():
                                                key_str = key
                                                if value.is_string():
                                                    edge_dict[key_str] = value.as_string()
                                                elif value.is_int():
                                                    edge_dict[key_str] = value.as_int()
                                                elif value.is_double():
                                                    edge_dict[key_str] = value.as_double()

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((IOErrorException,)) if IOErrorException else (),
    )
    async def delete_node(self, node_id: str) -> None:
        """Delete a node."""

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
        """Delete multiple nodes."""
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges."""

        def _sync_remove_edge(source: str, target: str):
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # VID cannot be parameterized in DELETE statements
                # Use safe VID quoting to handle special characters properly
                source_quoted = _quote_vid(source)
                target_quoted = _quote_vid(target)
                query = f"DELETE EDGE DIRECTED {source_quoted} -> {target_quoted}"
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
