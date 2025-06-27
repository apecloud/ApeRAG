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
    """

    def __init__(self, namespace, workspace, embedding_func):
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            embedding_func=embedding_func,
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
                        node_dict = self._convert_nebula_value_map(props)

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
                        node_dict = self._convert_nebula_value_map(props)

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

                                edge_dict = self._convert_nebula_value_map(props)

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
                    logger.error(f"Failed to upsert edge from {source_node_id} to {target_node_id}: {_safe_error_msg(result)}")
                    raise RuntimeError(f"Failed to upsert edge: {_safe_error_msg(result)}")

                logger.debug(f"Successfully upserted edge: '{source_node_id}' -> '{target_node_id}'")

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
