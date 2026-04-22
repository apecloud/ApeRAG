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

"""Graph-index business service.

Thin wrapper around ``aperag.graphindex.GraphIndexService`` that handles
the ApeRAG-specific concerns: collection lookup, auth, and mapping
graphindex DTOs to the shapes the REST UI expects. All the real logic
(SQL, LLM, summarization) lives inside graphindex.

Three endpoints route through here:

* ``get_graph_labels`` — list entity types for the UI dropdown.
* ``get_knowledge_graph`` — fetch a subgraph for visualization.
* ``merge_entities`` — consolidate multiple entities into one, with an
  LLM-generated summary description.

Two LightRAG-era curation features are not re-exposed in v2:

* ``generate_merge_suggestions`` / ``handle_suggestion_action`` — the
  "discover clusters to merge" pipeline. It was a 500-line LLM
  orchestration over LightRAG internals; its v2 replacement (if the
  product needs one) should be a dedicated module, not part of
  graphindex.
* ``export_for_kg_eval`` — snapshot export for the KG eval harness.
  Can be rebuilt as a thin SQL dump whenever we want it back.

The REST routes for those two actions still exist under
``aperag/views/graph.py`` and return HTTP 410 until the frontend is
updated. See ``docs/zh-CN/design/graphindex_rewrite.md``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from aperag.db.ops import async_db_ops
from aperag.exceptions import CollectionNotFoundException
from aperag.graphindex.dto import Entity as GraphIndexEntity
from aperag.graphindex.dto import Relation as GraphIndexRelation
from aperag.schema import view_models

logger = logging.getLogger(__name__)


class GraphService:
    """Read-side graph service used by the UI."""

    def __init__(self) -> None:
        # Local import avoids an import cycle with ``collection_service``.
        from aperag.service.collection_service import collection_service

        self.collection_service = collection_service
        self.db_ops = async_db_ops

    # ================================================================== labels
    async def get_graph_labels(self, user_id: str, collection_id: str) -> view_models.GraphLabelsResponse:
        """List distinct entity types for a collection.

        Empty list when the collection has not been indexed yet — that
        is the *correct* answer, not a cue to fall back to anything.
        """
        db_collection = await self._get_and_validate_collection(user_id, collection_id)

        from aperag.graphindex.integration import make_service_for_collection

        svc = make_service_for_collection(db_collection)
        labels = await svc.get_labels(collection_id=collection_id)
        return view_models.GraphLabelsResponse(labels=labels)

    # ============================================================= subgraph UI
    async def get_knowledge_graph(
        self,
        user_id: str,
        collection_id: str,
        label: str = None,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> Dict[str, Any]:
        """Fetch a knowledge-graph subgraph for visualization.

        ``label`` is either a specific entity type or "*" / empty for
        overview mode. In overview mode we oversample 2x and apply a
        degree-based picker so the visualiser gets well-connected
        nodes; the subsequent truncation sets ``is_truncated`` on the
        response.
        """
        db_collection = await self._get_and_validate_collection(user_id, collection_id)

        from aperag.graphindex.integration import make_service_for_collection

        svc = make_service_for_collection(db_collection)

        normalized_label = None if not label or label == "*" else label
        query_max_nodes = max_nodes * 2 if normalized_label is None else max_nodes
        mode_description = "overview" if normalized_label is None else f"subgraph from '{label}'"

        kg = await svc.get_knowledge_graph(
            collection_id=collection_id,
            label=normalized_label,
            max_depth=max_depth,
            max_nodes=query_max_nodes,
        )
        raw_nodes = _adapt_nodes(kg.nodes)
        raw_edges = _adapt_edges(kg.edges)
        is_truncated = bool(getattr(kg, "is_truncated", False))

        # Overview mode: prune to top-degree nodes when we asked the
        # storage for more than the UI limit. Sub-label mode passes
        # through unchanged.
        if normalized_label is None and len(raw_nodes) > max_nodes:
            optimized_nodes, optimized_edges = self._optimize_graph_for_visualization(raw_nodes, raw_edges, max_nodes)
            is_truncated = True
        else:
            optimized_nodes = raw_nodes
            optimized_edges = raw_edges

        result = self._to_ui_dict(optimized_nodes, optimized_edges, is_truncated)
        logger.info(
            "Retrieved %s graph for collection %s: %d nodes, %d edges",
            mode_description,
            collection_id,
            len(result["nodes"]),
            len(result["edges"]),
        )
        return result

    # ================================================================== merge
    async def merge_entities(
        self,
        user_id: str,
        collection_id: str,
        *,
        target_entity_id: str,
        source_entity_ids: List[str],
    ) -> Dict[str, Any]:
        """Merge entities and return a summary payload for the UI.

        The heavy lifting (structural merge + LLM summary of the merged
        description) happens inside
        ``GraphIndexService.merge_entities``. This layer only validates
        the collection and reshapes the DTO into the dict the existing
        frontend expects.
        """
        db_collection = await self._get_and_validate_collection(user_id, collection_id)

        from aperag.graphindex.integration import make_service_for_collection

        svc = make_service_for_collection(db_collection)
        result = await svc.merge_entities(
            collection_id=collection_id,
            target_entity_id=target_entity_id,
            source_entity_ids=source_entity_ids,
        )
        return {
            "target_entity_id": result.target_entity_id,
            "merged_source_ids": list(result.merged_source_ids),
            "description": result.description,
            "source_chunk_ids": list(result.source_chunk_ids),
            "edges_redirected": result.edges_redirected,
            "edges_collapsed": result.edges_collapsed,
        }

    # --------------------------------------------------------- internal helpers
    def _optimize_graph_for_visualization(self, nodes, edges, max_nodes):
        """Pick the top ``max_nodes`` entities by degree, filter edges to
        those whose endpoints survived the pick.

        Used only in overview mode (no label filter). Degree is
        computed from the edge list we already fetched, not with a
        separate SQL query, because overview mode always fetches the
        full subset anyway.
        """
        if len(nodes) <= max_nodes:
            return nodes, edges

        degree_map = {node.id: 0 for node in nodes}
        for edge in edges:
            if edge.source in degree_map and edge.target in degree_map:
                degree_map[edge.source] += 1
                degree_map[edge.target] += 1

        sorted_nodes = sorted(nodes, key=lambda n: (-degree_map[n.id], n.id))
        selected_nodes = sorted_nodes[:max_nodes]
        selected_ids = {n.id for n in selected_nodes}
        optimized_edges = [edge for edge in edges if edge.source in selected_ids and edge.target in selected_ids]
        return selected_nodes, optimized_edges

    def _to_ui_dict(self, nodes, edges, is_truncated: bool) -> Dict[str, Any]:
        """Convert adapted nodes/edges into the JSON the frontend expects.

        The schema is preserved exactly as it was under LightRAG so the
        web UI doesn't need to change: each node carries ``id``,
        ``labels``, and a flat ``properties`` dict; each edge has
        ``source``, ``target``, ``type``, and ``properties``.
        """
        default_node_fields = [
            "entity_id",
            "entity_name",
            "entity_type",
            "description",
            "source_id",
            "file_path",
        ]
        default_edge_fields = ["weight", "description", "keywords", "source_id", "file_path"]

        def extract_properties(obj, fields):
            if hasattr(obj, "properties") and obj.properties:
                return obj.properties
            return {f: getattr(obj, f, None) for f in fields if hasattr(obj, f)}

        def node_to_item(node):
            props = extract_properties(node, default_node_fields)
            if getattr(node, "labels", None) and node.labels:
                labels = node.labels
            else:
                display = props.get("entity_id") or props.get("entity_name")
                labels = [display] if display is not None else ([node.id] if hasattr(node, "id") else [])
            return {"id": node.id, "labels": labels, "properties": props}

        return {
            "nodes": [node_to_item(n) for n in nodes],
            "edges": [
                {
                    "id": edge.id,
                    "type": getattr(edge, "type", "DIRECTED"),
                    "source": edge.source,
                    "target": edge.target,
                    "properties": extract_properties(edge, default_edge_fields),
                }
                for edge in edges
            ],
            "is_truncated": is_truncated,
        }

    async def _get_and_validate_collection(self, user_id: str, collection_id: str):
        """Resolve the DB collection row and enforce that graph is enabled."""
        try:
            view_collection = await self.collection_service.get_collection(user_id, collection_id)
        except Exception:
            raise CollectionNotFoundException(collection_id)

        if not view_collection.config or not view_collection.config.enable_knowledge_graph:
            raise ValueError(f"Knowledge graph is not enabled for collection {collection_id}")

        db_collection = await self.collection_service.db_ops.query_collection(user_id, collection_id)
        if not db_collection:
            raise CollectionNotFoundException(collection_id)

        return db_collection


# ---------------------------------------------------------------------------
# Adapter helpers: graphindex DTOs → UI shape
# ---------------------------------------------------------------------------
#
# ``_to_ui_dict`` and ``_optimize_graph_for_visualization`` were originally
# written for the LightRAG ``KnowledgeGraph`` schema. Rather than rewrite
# the UI contract, we wrap graphindex DTOs in lightweight proxy objects
# that expose the attributes those helpers read. Keeps the frontend
# contract unchanged and the adapter isolated in one place.

from types import SimpleNamespace  # noqa: E402  (intentional in-line import)


def _adapt_nodes(nodes: List[GraphIndexEntity]) -> List[SimpleNamespace]:
    """Wrap ``graphindex.Entity`` for the UI dict builder."""
    out: List[SimpleNamespace] = []
    for n in nodes:
        if not isinstance(n, GraphIndexEntity):
            # Already in legacy shape (defensive; should not happen after
            # the LightRAG removal but cheap to keep).
            out.append(n)  # type: ignore[arg-type]
            continue
        props = {
            "entity_id": n.entity_id,
            "entity_name": n.name,
            "entity_type": n.type,
            "description": n.description,
            "source_id": ",".join(n.source_chunk_ids) if n.source_chunk_ids else "",
            "file_path": "",
        }
        out.append(
            SimpleNamespace(
                id=n.entity_id,
                labels=[n.type] if n.type else [n.name],
                properties=props,
                entity_id=n.entity_id,
                entity_name=n.name,
                entity_type=n.type,
                description=n.description,
                source_id=props["source_id"],
                file_path="",
            )
        )
    return out


def _adapt_edges(edges: List[GraphIndexRelation]) -> List[SimpleNamespace]:
    """Wrap ``graphindex.Relation`` for the UI dict builder."""
    out: List[SimpleNamespace] = []
    for e in edges:
        if not isinstance(e, GraphIndexRelation):
            out.append(e)  # type: ignore[arg-type]
            continue
        props = {
            "weight": float(e.weight),
            "description": e.description,
            "keywords": "",
            "source_id": ",".join(e.source_chunk_ids) if e.source_chunk_ids else "",
            "file_path": "",
        }
        out.append(
            SimpleNamespace(
                id=f"{e.source_id}->{e.target_id}",
                type="DIRECTED",
                source=e.source_id,
                target=e.target_id,
                properties=props,
                weight=props["weight"],
                description=e.description,
                keywords="",
                source_id=props["source_id"],
                file_path="",
            )
        )
    return out


# Global service instance
graph_service = GraphService()
