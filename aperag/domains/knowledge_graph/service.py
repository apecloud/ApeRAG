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

"""Graph-index business service (canonical ``knowledge_graph`` domain).

Relocated from ``aperag/service/graph_service.py`` by the Phase 2
hard-cut. Preserves the HTTP-facing behavior byte-for-byte: only the
module path and a couple of imports change (``aperag.db.models.Collection``
→ ``aperag.domains.knowledge_graph.ports.CollectionRow`` structural
Protocol, ``aperag.schema.view_models.*`` → ``.schemas.*``).

Three read-side endpoints route through here:

* ``get_graph_labels`` — list entity types for the UI dropdown.
* ``get_knowledge_graph`` — fetch a subgraph for visualization.
* ``merge_entities`` — consolidate multiple entities into one, with an
  LLM-generated summary description.

Merge-suggestion discovery itself still lives in the dedicated
``aperag.graph_curation`` infrastructure module; that module is **not**
covered by the Phase 0 strict ban (which only forbids
``aperag.service.*`` / ``aperag.schema.view_models`` /
``aperag.db.models``), so the domain can call into it directly without
crossing the boundary.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List

from aperag.db.ops import async_db_ops
from aperag.domains.knowledge_graph.graphindex.dto import Entity as GraphIndexEntity
from aperag.domains.knowledge_graph.graphindex.dto import Relation as GraphIndexRelation
from aperag.domains.knowledge_graph.ports import CollectionRow
from aperag.domains.knowledge_graph.schemas import GraphLabelsResponse, KnowledgeGraph
from aperag.exceptions import CollectionNotFoundException

if TYPE_CHECKING:
    from aperag.domains.knowledge_graph.schemas import (
        GraphEntitiesSearchResponse,
        GraphRelationView,
        GraphSearchEntity,
        GraphSubgraphExpandResponse,
    )

logger = logging.getLogger(__name__)


class GraphService:
    """Read-side graph service used by the UI."""

    def __init__(self) -> None:
        self.db_ops = async_db_ops

    # ================================================================== labels
    async def get_graph_labels(self, user_id: str, collection_id: str) -> GraphLabelsResponse:
        """List distinct entity types for a collection.

        Empty list when the collection has not been indexed yet — that
        is the *correct* answer, not a cue to fall back to anything.

        Wave 6 #40 narrow replacement (per architect ruling
        msg=3efdf906): reads from the canonical
        :class:`LineageGraphStore` Protocol via the worker_factory's
        backend dispatch (mirroring ``retrieval/pipeline.py:_graph_search``
        from #33 chunk 3). The legacy
        ``GraphIndexService.list_labels`` callsite is retired here;
        the legacy ``graphindex/`` package stays alive only for
        ``get_knowledge_graph`` + ``merge_entities`` (deferred per
        Option C).
        """
        db_collection = await self._get_and_validate_collection(user_id, collection_id)

        # Lazy import keeps service module free of indexing-layer
        # dependencies at import time (consistent with
        # ``retrieval/pipeline.py:_build_lineage_graph_store_for``).
        from aperag.indexing.worker_factory import (
            _build_lineage_graph_store,
            _resolve_graph_backend_type,
        )

        backend_type = _resolve_graph_backend_type(db_collection)
        store = _build_lineage_graph_store(backend_type=backend_type, collection=db_collection)
        labels = await store.list_entity_labels()
        return GraphLabelsResponse(labels=labels)

    # ============================================================= subgraph UI
    async def get_knowledge_graph(
        self,
        user_id: str,
        collection_id: str,
        label: str = None,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """Fetch a knowledge-graph subgraph for visualization.

        ``label`` is either a specific entity type or "*" / empty for
        overview mode. In overview mode we oversample 2x and apply a
        degree-based picker so the visualiser gets well-connected
        nodes; the subsequent truncation sets ``is_truncated`` on the
        response.
        """
        db_collection = await self._get_and_validate_collection(user_id, collection_id)

        from aperag.domains.knowledge_graph.graphindex.integration import make_service_for_collection

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

        result = KnowledgeGraph.model_validate(self._to_ui_dict(optimized_nodes, optimized_edges, is_truncated))
        logger.info(
            "Retrieved %s graph for collection %s: %d nodes, %d edges",
            mode_description,
            collection_id,
            len(result.nodes),
            len(result.edges),
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
        """Merge entities — Wave 7 §K.12.8 task #8 cutover to
        :class:`LineageEntityMerger` (PR #1758).

        The merger runs the 8-step orchestration inside the new lineage
        layer: alias-map writes for transparent indexer redirect (steps
        1-2), per-doc parts re-anchor under the canonical name (step 6a,
        preserves invariant #1 lineage), unified-description LLM merge
        + compaction (steps 4-5 + 6b), vector point upsert with the
        canonical name + 3-field payload + uuid5 id (step 7), and
        finally L1 + vector deletion of source entities (step 8).

        Response shape is preserved byte-for-byte for backward-compat
        with the existing frontend (cuiwenbo task #9 stays at
        OpenAPI-regen + flow smoke):

        * ``target_entity_id`` ← ``LineageMergeResult.final_target``
        * ``description`` ← ``compacted_description`` if present, else
          ``unified_description``
        * ``source_chunk_ids`` ← chunk ids contributed by the source
          entities, recovered from the target's lineage *after* the
          merge has re-anchored them under the canonical name
        * ``edges_redirected`` / ``edges_collapsed`` — the new merger
          does not return per-edge stats (the alias-redirect decorator
          handles edge re-anchoring transparently at indexer write
          time, not at merge time). We surface ``0`` for both fields
          to keep the response shape stable.
        """
        db_collection = await self._get_and_validate_collection(user_id, collection_id)

        from aperag.graph_curation.alias_map import AliasCycleError
        from aperag.graph_curation.lineage_merge import build_lineage_entity_merger_for

        merger = build_lineage_entity_merger_for(db_collection)
        try:
            merge_result = await merger.merge_entities(
                target_name=target_entity_id,
                source_names=source_entity_ids,
                merged_by=user_id,
            )
        except AliasCycleError as exc:
            # Surface as a validation error so the route returns 400
            # (per the existing ``ValueError`` → 400 mapping in
            # ``merge_nodes_view``).
            raise ValueError(str(exc)) from exc

        try:
            from aperag.graph_curation import graph_curation_service

            await graph_curation_service.expire_pending_for_entities(
                collection_id=collection_id,
                entity_ids=[target_entity_id, *source_entity_ids],
                reason="manual_merge",
            )
        except Exception:
            logger.exception("Failed to expire stale graph-curation suggestions after manual merge")

        # Recover source chunk ids from the target's lineage after the
        # merge — step 6a re-anchored each source's parts under the
        # canonical name with their original ``(document_id,
        # parse_version, chunk_ids)`` so the union of those chunks is
        # what the UI expects.
        source_chunk_ids = await self._collect_source_chunk_ids(
            db_collection,
            entity_name=merge_result.final_target,
        )

        description = merge_result.compacted_description or merge_result.unified_description or ""

        return {
            "target_entity_id": merge_result.final_target,
            "merged_source_ids": list(merge_result.merged_source_ids),
            "description": description,
            "source_chunk_ids": source_chunk_ids,
            # Edge re-anchoring is handled transparently by the
            # alias-redirect decorator at indexer write time (Wave 7
            # §K.12 invariant #9), not as part of the merge action,
            # so per-edge counts are not surfaced. ``0`` keeps the
            # response shape stable for backward-compat.
            "edges_redirected": 0,
            "edges_collapsed": 0,
        }

    async def _collect_source_chunk_ids(self, collection: Any, *, entity_name: str) -> List[str]:
        """Return the union of chunk ids attached to ``entity_name``'s
        lineage members after the merge has run. Used by
        :meth:`merge_entities` to populate the backward-compat
        ``source_chunk_ids`` field."""
        import asyncio

        from aperag.indexing.worker_factory import (
            _build_lineage_graph_store_inner,
            _resolve_graph_backend_type,
        )

        backend_type = _resolve_graph_backend_type(collection)
        store = await asyncio.to_thread(
            _build_lineage_graph_store_inner, backend_type=backend_type, collection=collection
        )
        entity = await store.get_entity(entity_name)
        if entity is None:
            return []
        chunk_ids: list[str] = []
        seen: set[str] = set()
        for member in entity.source_lineage or ():
            for cid in getattr(member, "chunk_ids", ()) or ():
                cid_str = str(cid)
                if cid_str in seen:
                    continue
                seen.add(cid_str)
                chunk_ids.append(cid_str)
        return chunk_ids

    # ============================================================ Wave 7 §K.12.6
    async def search_entities(
        self,
        user_id: str,
        collection_id: str,
        *,
        query: str,
        top_k: int = 10,
    ) -> "GraphEntitiesSearchResponse":
        """Vector-recall the top-K entities for ``query``.

        Backed by :class:`GraphSearchService.search_entities`. Internal
        endpoint serving the ``query_graph_entities`` MCP tool.
        """
        import asyncio

        from aperag.domains.knowledge_graph.schemas import GraphEntitiesSearchResponse
        from aperag.indexing.graph_search_service import build_graph_search_service_for

        db_collection = await self._get_and_validate_collection(user_id, collection_id)
        service = await asyncio.to_thread(build_graph_search_service_for, db_collection)
        entities = await service.search_entities(query=query, top_k=top_k)
        return GraphEntitiesSearchResponse(entities=[_entity_to_search_view(e) for e in entities])

    async def expand_subgraph(
        self,
        user_id: str,
        collection_id: str,
        *,
        entity_names: List[str],
        hops: int = 1,
    ) -> "GraphSubgraphExpandResponse":
        """Expand neighbours from ``entity_names`` within ``hops``.

        Backed by :class:`GraphSearchService.get_subgraph` (delegates to
        :meth:`LineageGraphStore.expand_neighbors_n_hops`).
        """
        import asyncio

        from aperag.domains.knowledge_graph.schemas import GraphSubgraphExpandResponse
        from aperag.indexing.graph_search_service import build_graph_search_service_for

        db_collection = await self._get_and_validate_collection(user_id, collection_id)
        service = await asyncio.to_thread(build_graph_search_service_for, db_collection)
        entities, relations = await service.get_subgraph(entity_names=entity_names, hops=hops)
        return GraphSubgraphExpandResponse(
            entities=[_entity_to_search_view(e) for e in entities],
            relations=[_relation_to_view(r) for r in relations],
        )

    async def get_entity_detail(
        self,
        user_id: str,
        collection_id: str,
        *,
        entity_name: str,
    ) -> "GraphSearchEntity | None":
        """Return the full record for a single entity.

        Returns ``None`` when the entity is not found in the lineage
        store — the route handler turns that into a 404.
        """
        import asyncio

        from aperag.indexing.worker_factory import _build_lineage_graph_store, _resolve_graph_backend_type

        db_collection = await self._get_and_validate_collection(user_id, collection_id)
        backend_type = _resolve_graph_backend_type(db_collection)
        store = await asyncio.to_thread(_build_lineage_graph_store, backend_type=backend_type, collection=db_collection)
        entity = await store.get_entity(entity_name)
        if entity is None:
            return None
        return _entity_to_search_view(entity)

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
        """Convert adapted nodes/edges into the canonical public graph DTO.

        Keep the visualizer shape (id/labels/properties and source/target),
        but strip storage internals such as source chunk IDs, file paths, and
        timestamps before the data crosses the API boundary.
        """
        default_node_fields = [
            "entity_id",
            "entity_name",
            "entity_type",
            "description",
            "source_chunk_count",
        ]
        default_edge_fields = ["weight", "description", "keywords", "source_chunk_count"]

        def extract_properties(obj, fields):
            raw = {}
            if hasattr(obj, "properties") and obj.properties:
                raw = obj.properties
                if hasattr(raw, "model_dump"):
                    raw = raw.model_dump(exclude_none=True)
                elif not isinstance(raw, dict):
                    raw = dict(raw)
            else:
                raw = {f: getattr(obj, f, None) for f in fields if hasattr(obj, f)}
            return {f: raw.get(f) for f in fields if raw.get(f) is not None}

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

    async def _get_and_validate_collection(self, user_id: str, collection_id: str) -> CollectionRow:
        """Resolve the collection row and enforce that graph is enabled.

        Legacy ``aperag.service.collection_service`` would fetch the
        view-model collection first and then the SQLAlchemy row. Inside
        the domain we short-circuit to a single ``db_ops`` lookup: the
        domain binds its dependency only to the narrow ``CollectionRow``
        Protocol declared in ``.ports``, not to the concrete
        SQLAlchemy ``Collection`` type.
        """
        from aperag.schema.utils import parseCollectionConfig

        db_collection = await self.db_ops.query_collection(user_id, collection_id)
        if not db_collection:
            raise CollectionNotFoundException(collection_id)

        config = parseCollectionConfig(db_collection.config)
        if not config or not config.enable_knowledge_graph:
            raise ValueError(f"Knowledge graph is not enabled for collection {collection_id}")

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
            "source_chunk_count": len(n.source_chunk_ids),
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
                source_chunk_count=props["source_chunk_count"],
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
            "source_chunk_count": len(e.source_chunk_ids),
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
                source_chunk_count=props["source_chunk_count"],
            )
        )
    return out


# ---------------------------------------------------------------------------
# Wave 7 §K.12.6 — EntityWithLineage / RelationWithLineage → public view
# ---------------------------------------------------------------------------


def _entity_to_search_view(entity: Any) -> "GraphSearchEntity":
    """Project an :class:`EntityWithLineage` to the public search shape.

    Uses the same compacted-or-fallback rule as
    :func:`GraphSearchService.compose_context` so MCP callers see the
    same description independent of which path (search vs subgraph
    vs detail) returned it.
    """
    from aperag.domains.knowledge_graph.schemas import GraphSearchEntity
    from aperag.indexing.graph_search_service import GraphSearchService

    description = GraphSearchService._render_description(
        getattr(entity, "compacted_description", None),
        entity.description_parts or (),
    )
    # ``source_chunk_count`` aggregates chunk ids across all lineage
    # members (per :class:`LineageMember.chunk_ids`). Description parts
    # only carry text, not chunk ids — those live on the lineage side.
    chunk_ids: set[str] = set()
    for member in entity.source_lineage or ():
        for cid in getattr(member, "chunk_ids", ()) or ():
            chunk_ids.add(str(cid))
    return GraphSearchEntity(
        name=entity.name,
        entity_type=entity.entity_type or "entity",
        description=description,
        source_chunk_count=len(chunk_ids),
    )


def _relation_to_view(relation: Any) -> "GraphRelationView":
    """Project a :class:`RelationWithLineage` to the public view."""
    from aperag.domains.knowledge_graph.schemas import GraphRelationView
    from aperag.indexing.graph_search_service import GraphSearchService

    description = GraphSearchService._render_description(
        getattr(relation, "compacted_description", None),
        relation.description_parts or (),
    )
    return GraphRelationView(
        source=relation.source,
        target=relation.target,
        description=description,
    )


# Global service instance
graph_service = GraphService()


__all__ = ["GraphService", "graph_service"]
