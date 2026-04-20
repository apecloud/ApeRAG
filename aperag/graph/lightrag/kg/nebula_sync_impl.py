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

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, final

from nebula3.common import ttypes

from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph
from ..utils import logger

logging.getLogger("nebula3").setLevel(logging.WARNING)


def _prepare_nebula_params(params: dict[str, Any]) -> dict[str, ttypes.Value]:
    """Convert Python values to Nebula ttypes.Value objects."""
    nebula_params: dict[str, ttypes.Value] = {}
    for key, value in params.items():
        param_value = ttypes.Value()
        if isinstance(value, bool):
            param_value.set_bVal(value)
        elif isinstance(value, str):
            param_value.set_sVal(value)
        elif isinstance(value, int):
            param_value.set_iVal(value)
        elif isinstance(value, float):
            param_value.set_fVal(value)
        elif isinstance(value, list):
            value_list = []
            for item in value:
                item_value = ttypes.Value()
                if isinstance(item, bool):
                    item_value.set_bVal(item)
                elif isinstance(item, str):
                    item_value.set_sVal(item)
                elif isinstance(item, int):
                    item_value.set_iVal(item)
                elif isinstance(item, float):
                    item_value.set_fVal(item)
                else:
                    item_value.set_sVal(str(item))
                value_list.append(item_value)
            param_value.set_lVal(ttypes.NList(values=value_list))
        else:
            param_value.set_sVal(str(value))
        nebula_params[key] = param_value
    return nebula_params


def _quote_vid(vid: str) -> str:
    """Safely quote a VID for nGQL queries."""
    escaped = vid.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _convert_nebula_value(value) -> Any:
    """Convert a single Nebula Value to Python type."""
    if value.is_null():
        return None
    if value.is_string():
        return value.as_string()
    if value.is_int():
        return value.as_int()
    if value.is_double():
        return value.as_double()
    if value.is_bool():
        return value.as_bool()
    if value.is_list():
        return [_convert_nebula_value(item) for item in value.as_list()]
    return str(value)


def _safe_error_msg(result) -> str:
    """Safely extract error message from Nebula result, handling UTF-8 decode errors."""
    try:
        error_code = result.error_code()
        try:
            error_msg = result.error_msg()
        except Exception as msg_error:
            logger.warning(f"Failed to extract error message: {msg_error}")
            return f"Nebula operation failed (error code: {error_code})"

        if error_msg is None:
            return f"Nebula operation failed (error code: {error_code})"
        if isinstance(error_msg, str):
            return f"Nebula error (code: {error_code}): {error_msg}"
        if isinstance(error_msg, bytes):
            decoded_msg = None
            for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
                try:
                    decoded_msg = error_msg.decode(encoding)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            if decoded_msg is None:
                try:
                    decoded_msg = error_msg.decode("utf-8", errors="replace")
                except Exception:
                    decoded_msg = str(error_msg)
            return f"Nebula error (code: {error_code}): {decoded_msg}"
        return f"Nebula error (code: {error_code}): {str(error_msg)}"
    except Exception as exc:
        logger.warning(f"Failed to process Nebula error: {exc}")
        try:
            return f"Nebula operation failed (error code: {result.error_code()})"
        except Exception:
            return "Nebula operation failed (unknown error)"


@final
@dataclass
class NebulaSyncStorage(BaseGraphStorage):
    """
    Nebula storage implementation using sync driver with async interface.

    Security Strategy:
    - Query operations use parameterized MATCH queries.
    - Mutating operations use nGQL where VID parameterization is unavailable, so VIDs are quoted safely.
    """

    def __init__(self, namespace, workspace, embedding_func=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            embedding_func=None,
        )
        self._space_name = None

    def _convert_nebula_value_map(self, value_map: dict) -> dict[str, Any]:
        result = {}
        for key, value in value_map.items():
            result[key] = _convert_nebula_value(value)
        return result

    async def initialize(self):
        """Initialize storage and prepare database."""
        if NebulaSyncConnectionManager is None:
            raise RuntimeError("Nebula sync connection manager is not available")

        self._space_name = await asyncio.to_thread(
            NebulaSyncConnectionManager.prepare_space, self.workspace, max_wait=30, fail_on_timeout=True
        )
        logger.debug(f"NebulaSyncStorage initialized for workspace '{self.workspace}', space '{self._space_name}'")

    async def finalize(self):
        """Clean up resources."""
        logger.debug(f"NebulaSyncStorage finalized for workspace '{self.workspace}'")

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists using MATCH syntax."""

        def _sync_has_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = "MATCH (v) WHERE id(v) == $vid RETURN v LIMIT 1"
                result = session.execute_parameter(query, _prepare_nebula_params({"vid": node_id}))
                return result.is_succeeded() and result.row_size() > 0

        return await asyncio.to_thread(_sync_has_node)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes."""

        def _sync_has_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = """
                MATCH (src)-[e:DIRECTED]-(dst)
                WHERE (id(src) == $src_id AND id(dst) == $dst_id)
                   OR (id(src) == $dst_id AND id(dst) == $src_id)
                RETURN e LIMIT 1
                """
                result = session.execute_parameter(
                    query,
                    _prepare_nebula_params({"src_id": source_node_id, "dst_id": target_node_id}),
                )
                return result.is_succeeded() and result.row_size() > 0

        return await asyncio.to_thread(_sync_has_edge)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its identifier."""

        def _sync_get_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = "MATCH (v:base) WHERE id(v) == $node_id RETURN properties(v) as props"
                result = session.execute_parameter(query, _prepare_nebula_params({"node_id": node_id}))

                if result.is_succeeded() and result.row_size() > 0:
                    for row in result:
                        props = row.values()[0].as_map()
                        node_dict = self._convert_nebula_value_map(props)
                        node_dict["entity_id"] = node_id
                        return node_dict
                return None

        return await asyncio.to_thread(_sync_get_node)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieve multiple nodes in one query."""

        def _sync_get_nodes_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                nodes = {}
                if not node_ids:
                    return nodes

                batch_size = 100
                for i in range(0, len(node_ids), batch_size):
                    batch_ids = node_ids[i : i + batch_size]
                    query = "MATCH (v:base) WHERE id(v) IN $node_ids RETURN id(v) as vid, properties(v) as props"
                    result = session.execute_parameter(query, _prepare_nebula_params({"node_ids": batch_ids}))

                    if result.is_succeeded():
                        for row in result:
                            batch_node_id = row.values()[0].as_string()
                            props = row.values()[1].as_map()
                            node_dict = self._convert_nebula_value_map(props)
                            node_dict["entity_id"] = batch_node_id
                            nodes[batch_node_id] = node_dict

                return nodes

        return await asyncio.to_thread(_sync_get_nodes_batch)

    async def node_degree(self, node_id: str) -> int:
        """Get the degree of a node."""

        def _sync_node_degree():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = """
                MATCH (v)-[r]-(other)
                WHERE id(v) == $node_id
                RETURN COUNT(r) AS degree
                """
                result = session.execute_parameter(query, _prepare_nebula_params({"node_id": node_id}))
                if result.is_succeeded() and result.row_size() > 0:
                    for row in result:
                        return row.values()[0].as_int()
                return 0

        return await asyncio.to_thread(_sync_node_degree)

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieve degrees for multiple nodes."""

        def _sync_node_degrees_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if not node_ids:
                    return {}

                degrees = {}
                batch_size = 100
                for i in range(0, len(node_ids), batch_size):
                    batch_ids = node_ids[i : i + batch_size]
                    query = """
                    UNWIND $node_ids AS node_id
                    MATCH (v)-[r]-(other)
                    WHERE id(v) == node_id
                    RETURN node_id, COUNT(r) AS degree
                    """
                    result = session.execute_parameter(query, _prepare_nebula_params({"node_ids": batch_ids}))
                    if result.is_succeeded():
                        for row in result:
                            batch_node_id = row.values()[0].as_string()
                            degree = row.values()[1].as_int()
                            degrees[batch_node_id] = degree

                    for batch_node_id in batch_ids:
                        if batch_node_id not in degrees:
                            degrees[batch_node_id] = 0

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
        return {(src, tgt): degrees.get(src, 0) + degrees.get(tgt, 0) for src, tgt in edge_pairs}

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """Get edge properties between two nodes."""

        def _sync_get_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = """
                MATCH (src)-[e:DIRECTED]-(dst)
                WHERE (id(src) == $src_id AND id(dst) == $dst_id)
                   OR (id(src) == $dst_id AND id(dst) == $src_id)
                RETURN properties(e) as props LIMIT 1
                """
                result = session.execute_parameter(
                    query,
                    _prepare_nebula_params({"src_id": source_node_id, "dst_id": target_node_id}),
                )

                if result.is_succeeded() and result.row_size() > 0:
                    for row in result:
                        props = row.values()[0].as_map()
                        edge_dict = self._convert_nebula_value_map(props)
                        for key, default_value in {
                            "weight": 0.0,
                            "source_id": None,
                            "description": None,
                            "keywords": None,
                        }.items():
                            if key not in edge_dict:
                                edge_dict[key] = default_value
                        return edge_dict
                return None

        return await asyncio.to_thread(_sync_get_edge)

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        """Retrieve multiple edges in batches."""

        def _sync_get_edges_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                edges_dict = {}
                if not pairs:
                    return edges_dict

                for pair in pairs:
                    src, tgt = pair["src"], pair["tgt"]
                    edges_dict[(src, tgt)] = {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }

                batch_size = 100
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i : i + batch_size]
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

                    batch_query = " UNION ALL ".join(union_queries)
                    result = session.execute_parameter(batch_query, _prepare_nebula_params(all_params))

                    if result.is_succeeded():
                        for row in result:
                            src_id = row.values()[0].as_string()
                            tgt_id = row.values()[1].as_string()
                            props = row.values()[2].as_map()
                            edge_dict = self._convert_nebula_value_map(props)

                            for key, default_value in {
                                "weight": 0.0,
                                "source_id": None,
                                "description": None,
                                "keywords": None,
                            }.items():
                                if key not in edge_dict:
                                    edge_dict[key] = default_value

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
                result = session.execute_parameter(query, _prepare_nebula_params({"source_node_id": source_node_id}))

                edges = []
                edges_set = set()
                if result.is_succeeded():
                    for row in result:
                        src = row.values()[0].as_string()
                        tgt = row.values()[1].as_string()
                        if (tgt, src) not in edges_set:
                            edges.append((src, tgt))
                            edges_set.add((src, tgt))

                return edges if edges else None

        return await asyncio.to_thread(_sync_get_node_edges)

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """Retrieve edges for multiple nodes in batches."""

        def _sync_get_nodes_edges_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if not node_ids:
                    return {}

                edges_dict = {node_id: [] for node_id in node_ids}
                batch_size = 100

                for i in range(0, len(node_ids), batch_size):
                    batch_ids = node_ids[i : i + batch_size]
                    query = """
                    UNWIND $node_ids AS node_id
                    MATCH (v)-[r]-(connected)
                    WHERE id(v) == node_id
                    RETURN node_id, id(v) as src, id(connected) as dst
                    """
                    result = session.execute_parameter(query, _prepare_nebula_params({"node_ids": batch_ids}))
                    node_edges_sets = {node_id: set() for node_id in batch_ids}

                    if result.is_succeeded():
                        for row in result:
                            source_node_id = row.values()[0].as_string()
                            src = row.values()[1].as_string()
                            dst = row.values()[2].as_string()

                            if (dst, src) not in node_edges_sets[source_node_id]:
                                edges_dict[source_node_id].append((src, dst))
                                node_edges_sets[source_node_id].add((src, dst))

                return edges_dict

        return await asyncio.to_thread(_sync_get_nodes_edges_batch)

    async def get_incident_edges_with_data_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str, dict]]]:
        """Retrieve incident edges with edge payloads for multiple nodes in batches."""

        def _sync_get_incident_edges_with_data_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if not node_ids:
                    return {}

                edges_dict = {node_id: [] for node_id in node_ids}
                batch_size = 100

                for i in range(0, len(node_ids), batch_size):
                    batch_ids = node_ids[i : i + batch_size]
                    query = """
                    UNWIND $node_ids AS node_id
                    MATCH (v)-[r:DIRECTED]-(connected)
                    WHERE id(v) == node_id
                    RETURN node_id, id(v) as src, id(connected) as dst, properties(r) as props
                    """
                    result = session.execute_parameter(query, _prepare_nebula_params({"node_ids": batch_ids}))
                    node_edges_sets = {node_id: set() for node_id in batch_ids}

                    if result.is_succeeded():
                        for row in result:
                            source_node_id = row.values()[0].as_string()
                            src = row.values()[1].as_string()
                            dst = row.values()[2].as_string()
                            props = row.values()[3].as_map()
                            edge_pair = (src, dst)
                            if edge_pair in node_edges_sets[source_node_id]:
                                continue
                            node_edges_sets[source_node_id].add(edge_pair)

                            edge_data = self._convert_nebula_value_map(props)
                            for key, default_value in {
                                "weight": 0.0,
                                "source_id": None,
                                "description": None,
                                "keywords": None,
                            }.items():
                                if key not in edge_data:
                                    edge_data[key] = default_value

                            edges_dict[source_node_id].append((src, dst, edge_data))

                return edges_dict

        return await asyncio.to_thread(_sync_get_incident_edges_with_data_batch)

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the database."""

        def _sync_upsert_node():
            if "entity_id" not in node_data:
                raise ValueError("Nebula: node properties must contain an 'entity_id' field")

            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                prop_names = []
                param_dict = {}

                for key, value in node_data.items():
                    if value is not None:
                        prop_names.append(key)
                        param_dict[f"prop_{key}"] = value

                if not prop_names:
                    logger.warning(f"No properties to insert for node {node_id}")
                    return

                set_clause = ", ".join([f"base.{key} = $prop_{key}" for key in prop_names])
                query = f"UPSERT VERTEX {_quote_vid(node_id)} SET {set_clause}"

                result = session.execute_parameter(query, _prepare_nebula_params(param_dict))
                if not result.is_succeeded():
                    logger.error(f"Failed to upsert node {node_id}: {_safe_error_msg(result)}")
                    raise RuntimeError(f"Failed to upsert node: {_safe_error_msg(result)}")

        return await asyncio.to_thread(_sync_upsert_node)

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """Upsert an edge in the database."""

        def _sync_upsert_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                valid_props = {k: v for k, v in edge_data.items() if v is not None}
                if not valid_props:
                    logger.warning(f"No valid properties to upsert for edge {source_node_id} -> {target_node_id}")
                    return

                set_clauses = []
                param_dict = {}
                for key, value in valid_props.items():
                    param_key = f"prop_{key}"
                    set_clauses.append(f"{key} = ${param_key}")
                    param_dict[param_key] = value

                query = (
                    f"UPSERT EDGE {_quote_vid(source_node_id)} -> {_quote_vid(target_node_id)} "
                    f"OF DIRECTED SET {', '.join(set_clauses)}"
                )
                result = session.execute_parameter(query, _prepare_nebula_params(param_dict))
                if not result.is_succeeded():
                    logger.error(
                        f"Failed to upsert edge from {source_node_id} to {target_node_id}: {_safe_error_msg(result)}"
                    )
                    raise RuntimeError(f"Failed to upsert edge: {_safe_error_msg(result)}")

        return await asyncio.to_thread(_sync_upsert_edge)

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """This function is not used in ApeRAG today."""
        raise NotImplementedError

    async def get_all_labels(self) -> list[str]:
        """Get all node labels."""

        def _sync_get_all_labels():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                result = session.execute("LOOKUP ON base YIELD properties(vertex).entity_id as label")
                labels = []
                if result.is_succeeded():
                    for row in result:
                        label = row.values()[0].as_string()
                        if label:
                            labels.append(label)
                return list(set(labels))

        return await asyncio.to_thread(_sync_get_all_labels)

    async def get_node_ids(self, limit: int | None = None) -> list[str] | None:
        """Get node IDs directly without fetching full node payloads."""

        def _sync_get_node_ids():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = "MATCH (v:base) RETURN id(v) as vid ORDER BY vid"
                if limit is not None:
                    query += " LIMIT $limit"
                    result = session.execute_parameter(query, _prepare_nebula_params({"limit": limit}))
                else:
                    result = session.execute(query)

                node_ids = []
                if result.is_succeeded():
                    for row in result:
                        node_id = row.values()[0].as_string()
                        if node_id:
                            node_ids.append(node_id)
                return node_ids

        return await asyncio.to_thread(_sync_get_node_ids)

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its incident edges."""

        def _sync_delete_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                result = session.execute(f"DELETE VERTEX {_quote_vid(node_id)} WITH EDGE")
                if not result.is_succeeded():
                    logger.error(f"Failed to delete node {node_id}: {_safe_error_msg(result)}")
                    raise RuntimeError(f"Failed to delete node: {_safe_error_msg(result)}")

        return await asyncio.to_thread(_sync_delete_node)

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes in small batches."""

        def _sync_remove_nodes_batch(batch_nodes: list[str]):
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                for node_id in batch_nodes:
                    result = session.execute(f"DELETE VERTEX {_quote_vid(node_id)} WITH EDGE")
                    if not result.is_succeeded():
                        logger.error(f"Failed to delete node {node_id}: {_safe_error_msg(result)}")

        batch_size = 10
        for i in range(0, len(nodes), batch_size):
            await asyncio.to_thread(_sync_remove_nodes_batch, nodes[i : i + batch_size])

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges in small batches."""

        def _sync_remove_edges_batch(batch_edges: list[tuple[str, str]]):
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                for source, target in batch_edges:
                    result = session.execute(f"DELETE EDGE DIRECTED {_quote_vid(source)} -> {_quote_vid(target)}")
                    if not result.is_succeeded():
                        logger.error(f"Failed to delete edge {source} -> {target}: {_safe_error_msg(result)}")

        batch_size = 10
        for i in range(0, len(edges), batch_size):
            await asyncio.to_thread(_sync_remove_edges_batch, edges[i : i + batch_size])

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage."""

        def _sync_drop():
            with NebulaSyncConnectionManager.get_session() as session:
                result = session.execute(f"DROP SPACE IF EXISTS {self._space_name}")
                if result.is_succeeded():
                    NebulaSyncConnectionManager.discard_space(self._space_name)
                    logger.info(f"Dropped space {self._space_name}")
                    return {"status": "success", "message": "data dropped"}
                logger.error(f"Failed to drop space {self._space_name}: {_safe_error_msg(result)}")
                return {"status": "error", "message": _safe_error_msg(result)}

        return await asyncio.to_thread(_sync_drop)
