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
from collections import deque
from dataclasses import dataclass
from typing import Any, final

from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager

try:
    from nebula3.common import ttypes
except ImportError:
    ttypes = None

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger

logging.getLogger("nebula3").setLevel(logging.WARNING)


def _prepare_nebula_params(params: dict) -> dict:
    if ttypes is None:
        raise RuntimeError("nebula3-python is not installed")

    nebula_params = {}
    for key, value in params.items():
        param_value = ttypes.Value()
        if isinstance(value, str):
            param_value.set_sVal(value)
        elif isinstance(value, bool):
            param_value.set_bVal(value)
        elif isinstance(value, int):
            param_value.set_iVal(value)
        elif isinstance(value, float):
            param_value.set_fVal(value)
        elif isinstance(value, list):
            value_list = []
            for item in value:
                item_value = ttypes.Value()
                if isinstance(item, str):
                    item_value.set_sVal(item)
                elif isinstance(item, bool):
                    item_value.set_bVal(item)
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
    escaped = vid.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _convert_nebula_value(value) -> Any:
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
    try:
        error_code = result.error_code()
        try:
            error_msg = result.error_msg()
        except Exception as msg_error:
            logger.warning(f"Failed to extract Nebula error message: {msg_error}")
            return f"Nebula operation failed (error code: {error_code})"

        if error_msg is None:
            return f"Nebula operation failed (error code: {error_code})"
        if isinstance(error_msg, str):
            return f"Nebula error (code: {error_code}): {error_msg}"
        if isinstance(error_msg, bytes):
            for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
                try:
                    return f"Nebula error (code: {error_code}): {error_msg.decode(encoding)}"
                except (UnicodeDecodeError, LookupError):
                    continue
            return f"Nebula error (code: {error_code}): {error_msg.decode('utf-8', errors='replace')}"
        return f"Nebula error (code: {error_code}): {str(error_msg)}"
    except Exception as exc:
        logger.warning(f"Failed to process Nebula error: {exc}")
        return "Nebula operation failed (unknown error)"


@final
@dataclass
class NebulaSyncStorage(BaseGraphStorage):
    """
    NebulaGraph storage implementation using the sync driver with async wrappers.
    """

    def __init__(self, namespace, workspace, embedding_func=None):
        super().__init__(namespace=namespace, workspace=workspace, embedding_func=None)
        self._space_name = None

    def _convert_nebula_value_map(self, value_map: dict) -> dict[str, Any]:
        return {key: _convert_nebula_value(value) for key, value in value_map.items()}

    async def initialize(self):
        if NebulaSyncConnectionManager is None:
            raise RuntimeError("Nebula sync connection manager is not available")

        self._space_name = await asyncio.to_thread(
            NebulaSyncConnectionManager.prepare_space,
            self.workspace,
            30,
            True,
        )
        logger.debug(f"NebulaSyncStorage initialized for workspace '{self.workspace}', space '{self._space_name}'")

    async def finalize(self):
        logger.debug(f"NebulaSyncStorage finalized for workspace '{self.workspace}'")

    async def has_node(self, node_id: str) -> bool:
        def _sync_has_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = "MATCH (v) WHERE id(v) == $vid RETURN v LIMIT 1"
                result = session.execute_parameter(query, _prepare_nebula_params({"vid": node_id}))
                return result.is_succeeded() and result.row_size() > 0

        return await asyncio.to_thread(_sync_has_node)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        def _sync_has_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = """
                MATCH (src)-[e:DIRECTED]-(dst)
                WHERE (id(src) == $src_id AND id(dst) == $dst_id)
                   OR (id(src) == $dst_id AND id(dst) == $src_id)
                RETURN e LIMIT 1
                """
                params = {"src_id": source_node_id, "dst_id": target_node_id}
                result = session.execute_parameter(query, _prepare_nebula_params(params))
                return result.is_succeeded() and result.row_size() > 0

        return await asyncio.to_thread(_sync_has_edge)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
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
        def _sync_get_nodes_batch():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if not node_ids:
                    return {}
                nodes = {}
                batch_size = 100
                for i in range(0, len(node_ids), batch_size):
                    batch_ids = node_ids[i : i + batch_size]
                    query = "MATCH (v:base) WHERE id(v) IN $node_ids RETURN id(v) as vid, properties(v) as props"
                    result = session.execute_parameter(query, _prepare_nebula_params({"node_ids": batch_ids}))
                    if result.is_succeeded():
                        for row in result:
                            vid = row.values()[0].as_string()
                            props = row.values()[1].as_map()
                            node_dict = self._convert_nebula_value_map(props)
                            node_dict["entity_id"] = vid
                            nodes[vid] = node_dict
                return nodes

        return await asyncio.to_thread(_sync_get_nodes_batch)

    async def node_degree(self, node_id: str) -> int:
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
                            degrees[row.values()[0].as_string()] = row.values()[1].as_int()
                    for node_id in batch_ids:
                        degrees.setdefault(node_id, 0)
                return degrees

        return await asyncio.to_thread(_sync_node_degrees_batch)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return int(src_degree) + int(tgt_degree)

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        unique_node_ids = {src for src, _ in edge_pairs}
        unique_node_ids.update({tgt for _, tgt in edge_pairs})
        degrees = await self.node_degrees_batch(list(unique_node_ids))
        return {(src, tgt): degrees.get(src, 0) + degrees.get(tgt, 0) for src, tgt in edge_pairs}

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        def _sync_get_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = """
                MATCH (src)-[e:DIRECTED]-(dst)
                WHERE (id(src) == $src_id AND id(dst) == $dst_id)
                   OR (id(src) == $dst_id AND id(dst) == $src_id)
                RETURN properties(e) as props LIMIT 1
                """
                params = {"src_id": source_node_id, "dst_id": target_node_id}
                result = session.execute_parameter(query, _prepare_nebula_params(params))
                if result.is_succeeded() and result.row_size() > 0:
                    for row in result:
                        props = row.values()[0].as_map()
                        edge_dict = self._convert_nebula_value_map(props)
                        edge_dict.setdefault("weight", 0.0)
                        edge_dict.setdefault("source_id", None)
                        edge_dict.setdefault("description", None)
                        edge_dict.setdefault("keywords", None)
                        return edge_dict
                return None

        return await asyncio.to_thread(_sync_get_edge)

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        result = await super().get_edges_batch(pairs)
        for pair in pairs:
            result.setdefault(
                (pair["src"], pair["tgt"]),
                {"weight": 0.0, "source_id": None, "description": None, "keywords": None},
            )
        return result

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        def _sync_get_node_edges():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                query = """
                MATCH (v)-[r]-(connected)
                WHERE id(v) == $source_node_id
                RETURN id(v) as src, id(connected) as dst
                """
                result = session.execute_parameter(
                    query,
                    _prepare_nebula_params({"source_node_id": source_node_id}),
                )
                edges = []
                seen = set()
                if result.is_succeeded():
                    for row in result:
                        src = row.values()[0].as_string()
                        dst = row.values()[1].as_string()
                        if (dst, src) in seen:
                            continue
                        seen.add((src, dst))
                        edges.append((src, dst))
                return edges if edges else None

        return await asyncio.to_thread(_sync_get_node_edges)

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
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
                    seen = {node_id: set() for node_id in batch_ids}
                    if result.is_succeeded():
                        for row in result:
                            node_id = row.values()[0].as_string()
                            src = row.values()[1].as_string()
                            dst = row.values()[2].as_string()
                            if (dst, src) in seen[node_id]:
                                continue
                            seen[node_id].add((src, dst))
                            edges_dict[node_id].append((src, dst))
                return edges_dict

        return await asyncio.to_thread(_sync_get_nodes_edges_batch)

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        def _sync_upsert_node():
            if "entity_id" not in node_data:
                raise ValueError("Nebula node properties must contain an 'entity_id' field")

            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                valid_props = {k: v for k, v in node_data.items() if v is not None}
                if not valid_props:
                    logger.warning(f"No properties to insert for node {node_id}")
                    return

                set_clause = ", ".join(f"base.{key} = $prop_{key}" for key in valid_props)
                params = {f"prop_{key}": value for key, value in valid_props.items()}
                query = f"UPSERT VERTEX {_quote_vid(node_id)} SET {set_clause}"
                result = session.execute_parameter(query, _prepare_nebula_params(params))
                if not result.is_succeeded():
                    raise RuntimeError(f"Failed to upsert node: {_safe_error_msg(result)}")

        return await asyncio.to_thread(_sync_upsert_node)

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        def _sync_upsert_edge():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                valid_props = {k: v for k, v in edge_data.items() if v is not None}
                if not valid_props:
                    logger.warning(f"No valid properties to upsert for edge {source_node_id} -> {target_node_id}")
                    return

                set_clause = ", ".join(f"{key} = $prop_{key}" for key in valid_props)
                params = {f"prop_{key}": value for key, value in valid_props.items()}
                query = (
                    f"UPSERT EDGE {_quote_vid(source_node_id)} -> {_quote_vid(target_node_id)} "
                    f"OF DIRECTED SET {set_clause}"
                )
                result = session.execute_parameter(query, _prepare_nebula_params(params))
                if not result.is_succeeded():
                    raise RuntimeError(f"Failed to upsert edge: {_safe_error_msg(result)}")

        return await asyncio.to_thread(_sync_upsert_edge)

    async def delete_node(self, node_id: str) -> None:
        def _sync_delete_node():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                result = session.execute(f"DELETE VERTEX {_quote_vid(node_id)} WITH EDGE")
                if not result.is_succeeded():
                    raise RuntimeError(f"Failed to delete node: {_safe_error_msg(result)}")

        return await asyncio.to_thread(_sync_delete_node)

    async def remove_nodes(self, nodes: list[str]):
        def _sync_remove_nodes_batch(batch_nodes: list[str]):
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                for node_id in batch_nodes:
                    result = session.execute(f"DELETE VERTEX {_quote_vid(node_id)} WITH EDGE")
                    if not result.is_succeeded():
                        logger.error(f"Failed to delete node {node_id}: {_safe_error_msg(result)}")

        for i in range(0, len(nodes), 10):
            await asyncio.to_thread(_sync_remove_nodes_batch, nodes[i : i + 10])

    async def remove_edges(self, edges: list[tuple[str, str]]):
        def _sync_remove_edges_batch(batch_edges: list[tuple[str, str]]):
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                for source, target in batch_edges:
                    query = f"DELETE EDGE DIRECTED {_quote_vid(source)} -> {_quote_vid(target)}"
                    result = session.execute(query)
                    if not result.is_succeeded():
                        logger.error(f"Failed to delete edge {source} -> {target}: {_safe_error_msg(result)}")

        for i in range(0, len(edges), 10):
            await asyncio.to_thread(_sync_remove_edges_batch, edges[i : i + 10])

    async def get_all_labels(self) -> list[str]:
        def _sync_get_all_labels():
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                result = session.execute("MATCH (v:base) RETURN DISTINCT id(v) as label")
                labels = []
                if result.is_succeeded():
                    for row in result:
                        label = row.values()[0].as_string()
                        if label:
                            labels.append(label)
                return sorted(set(labels))

        return await asyncio.to_thread(_sync_get_all_labels)

    async def get_knowledge_graph(self, node_label: str, max_depth: int = 3, max_nodes: int = 1000) -> KnowledgeGraph:
        if max_nodes <= 0:
            return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

        selected_ids: list[str] = []
        is_truncated = False

        if node_label == "*":
            all_labels = await self.get_all_labels()
            selected_ids = all_labels[:max_nodes]
            is_truncated = len(all_labels) > len(selected_ids)
        else:
            if not await self.has_node(node_label):
                return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

            visited = {node_label}
            queue: deque[tuple[str, int]] = deque([(node_label, 0)])

            while queue and len(selected_ids) < max_nodes:
                current, depth = queue.popleft()
                selected_ids.append(current)
                if depth >= max_depth:
                    continue

                for src, dst in await self.get_node_edges(current) or []:
                    neighbor = dst if src == current else src
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    if len(visited) > max_nodes:
                        is_truncated = True
                    queue.append((neighbor, depth + 1))

            if queue:
                is_truncated = True

        nodes_map = await self.get_nodes_batch(selected_ids)
        selected_set = set(selected_ids)
        nodes = []
        for node_id in selected_ids:
            node_props = nodes_map.get(node_id)
            if not node_props:
                continue
            labels = [node_props["entity_type"]] if node_props.get("entity_type") else []
            nodes.append(KnowledgeGraphNode(id=node_id, labels=labels, properties=node_props))

        edges = []
        seen_edges = set()
        for node_id in selected_ids:
            for src, dst in await self.get_node_edges(node_id) or []:
                if src not in selected_set or dst not in selected_set:
                    continue
                edge_key = tuple(sorted((src, dst)))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edge_props = await self.get_edge(src, dst) or {}
                edges.append(
                    KnowledgeGraphEdge(
                        id=f"{src}->{dst}",
                        type="DIRECTED",
                        source=src,
                        target=dst,
                        properties=edge_props,
                    )
                )

        return KnowledgeGraph(nodes=nodes, edges=edges, is_truncated=is_truncated)

    async def drop(self) -> dict[str, str]:
        def _sync_drop():
            return NebulaSyncConnectionManager.drop_space(self._space_name)

        return await asyncio.to_thread(_sync_drop)
