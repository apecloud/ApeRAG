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
from aperag.domains.knowledge_graph.ports import CollectionRow
from aperag.domains.knowledge_graph.schemas import GraphLabelsResponse, KnowledgeGraph
from aperag.exceptions import CollectionNotFoundException

if TYPE_CHECKING:
    from aperag.domains.knowledge_graph.schemas import (
        GraphEmbeddingMapResponse,
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

        Wave 7 W7-10 cutover (per architect Q1 ratify msg=838d57c3):
        2-step pipeline replacing the legacy
        ``GraphIndexService.get_knowledge_graph``:

        1. ``store.list_entities(label, limit=query_max_nodes)`` —
           label-filtered entity list (primary work; new Protocol
           method shipped in this PR).
        2. ``GraphSearchService.get_subgraph(names, hops=max_depth)`` —
           optional edge expansion when ``max_depth > 0``.

        Each layer has clean semantics (W7-5 ``get_subgraph`` is
        anchor-expansion, NOT label-filter; using it as the primary
        entry point would force a wrapper that re-list_entities just
        to compute anchors — drift the architect catches in
        msg=838d57c3 own-up).
        """
        db_collection = await self._get_and_validate_collection(user_id, collection_id)

        normalized_label = None if not label or label == "*" else label
        query_max_nodes = max_nodes * 2 if normalized_label is None else max_nodes
        mode_description = "overview" if normalized_label is None else f"subgraph from '{label}'"

        # Lazy imports keep the service module free of indexing-layer
        # dependencies at import time (mirror ``get_graph_labels``).
        from aperag.indexing.graph_search_service import build_graph_search_service_for
        from aperag.indexing.worker_factory import (
            _build_lineage_graph_store,
            _resolve_graph_backend_type,
        )

        backend_type = _resolve_graph_backend_type(db_collection)
        store = _build_lineage_graph_store(backend_type=backend_type, collection=db_collection)
        # Step 1 — label-filtered entity list (paged single call; UI
        # asks for at most ``query_max_nodes`` so one page suffices).
        entities = await store.list_entities(label=normalized_label, limit=query_max_nodes)
        # Step 2 — optional edge expansion. ``get_subgraph`` walks
        # 1+ hops from the seed entities and returns relations that
        # touch them; we keep only those edges whose endpoints are in
        # the requested entity set so the UI does not show stray
        # neighbours past the ``label`` filter.
        relations: list[Any] = []
        if entities and max_depth > 0:
            search = build_graph_search_service_for(db_collection)
            seed_names = [entity.name for entity in entities]
            _, subgraph_relations = await search.get_subgraph(
                entity_names=seed_names,
                hops=max_depth,
            )
            allowed = set(seed_names)
            relations = [rel for rel in subgraph_relations if rel.source in allowed and rel.target in allowed]

        raw_nodes = _adapt_lineage_entities(entities)
        raw_edges = _adapt_lineage_relations(relations)
        is_truncated = False

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

    async def get_embedding_map(
        self,
        user_id: str,
        collection_id: str,
        *,
        max_entities: int = 1000,
    ) -> "GraphEmbeddingMapResponse":
        """Return a 2-D PCA projection of graph entity vectors.

        The hybrid graph view joins these stable coordinates with the
        existing topology response. Keeping coordinates and topology
        separate lets the frontend reuse the normal graph endpoint for
        edge metadata while this endpoint stays bounded to vector-backed
        entities only.
        """
        import asyncio
        import math
        import uuid as _uuid

        import numpy as np

        from aperag.domains.knowledge_graph.schemas import (
            GraphEmbeddingMapResponse,
            GraphEmbeddingPoint,
        )
        from aperag.indexing.worker_factory import (
            _build_collection_qdrant_connector,
            _build_lineage_graph_store,
            _resolve_graph_backend_type,
        )

        db_collection = await self._get_and_validate_collection(user_id, collection_id)
        backend_type = _resolve_graph_backend_type(db_collection)
        store = await asyncio.to_thread(
            _build_lineage_graph_store,
            backend_type=backend_type,
            collection=db_collection,
        )
        adaptor, _embedder, _vector_size = await asyncio.to_thread(
            _build_collection_qdrant_connector,
            db_collection,
        )
        connector = adaptor.connector

        max_entities = max(1, min(int(max_entities), 5000))
        entities = await store.list_entities(limit=max_entities, offset=0)
        if not entities:
            return GraphEmbeddingMapResponse(points=[], relations=[], cluster_labels={})

        ids: list[str] = []
        id_to_entity: dict[str, Any] = {}
        seen_names: set[str] = set()
        for entity in entities:
            if entity.name in seen_names:
                continue
            seen_names.add(entity.name)
            point_id = str(
                _uuid.uuid5(
                    _uuid.NAMESPACE_DNS,
                    f"graph_entity:{collection_id}:{entity.name}",
                )
            )
            ids.append(point_id)
            id_to_entity[point_id] = entity

        retrieved = await asyncio.to_thread(connector.retrieve, ids, with_vectors=True)
        if not retrieved:
            return GraphEmbeddingMapResponse(points=[], relations=[], cluster_labels={})

        vectors: list[list[float]] = []
        kept_entities: list[Any] = []
        for vector_point in retrieved:
            if vector_point.vector is None:
                continue
            vector = list(vector_point.vector)
            if not vector:
                continue
            entity = id_to_entity.get(str(vector_point.id))
            if entity is None:
                continue
            vectors.append(vector)
            kept_entities.append(entity)
        if not vectors:
            return GraphEmbeddingMapResponse(points=[], relations=[], cluster_labels={})

        matrix = np.asarray(vectors, dtype=np.float32)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return GraphEmbeddingMapResponse(points=[], relations=[], cluster_labels={})

        components = vh[:2]
        coords = centered @ components.T
        if coords.shape[1] == 0:
            coords = np.zeros((len(kept_entities), 2), dtype=np.float32)
        elif coords.shape[1] == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0], dtype=coords.dtype)])

        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        span = np.maximum(coords_max - coords_min, 1e-6)
        scale = 2000.0 / float(span.max())

        cluster_for_type: dict[str, int] = {}
        cluster_labels: dict[str, str] = {}
        points: list[GraphEmbeddingPoint] = []
        for entity, xy in zip(kept_entities, coords, strict=True):
            entity_type = (entity.entity_type or "unknown") or "unknown"
            cluster_idx = cluster_for_type.get(entity_type)
            if cluster_idx is None:
                cluster_idx = len(cluster_for_type)
                cluster_for_type[entity_type] = cluster_idx
                cluster_labels[str(cluster_idx)] = entity_type

            x = float(xy[0]) * scale
            y = float(xy[1]) * scale
            if not math.isfinite(x) or not math.isfinite(y):
                continue

            chunk_ids: set[str] = set()
            for member in entity.source_lineage or ():
                for chunk_id in getattr(member, "chunk_ids", ()) or ():
                    chunk_ids.add(str(chunk_id))

            points.append(
                GraphEmbeddingPoint(
                    name=entity.name,
                    entity_type=entity_type,
                    cluster=cluster_idx,
                    x=x,
                    y=y,
                    source_chunk_count=len(chunk_ids),
                )
            )

        return GraphEmbeddingMapResponse(
            points=points,
            relations=[],
            cluster_labels=cluster_labels,
        )

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
# written for the legacy ``KnowledgeGraph`` schema. Rather than rewrite
# the UI contract, we wrap lineage DTOs in lightweight proxy objects
# that expose the attributes those helpers read. Keeps the frontend
# contract unchanged and the adapter isolated in one place.


def _adapt_lineage_entities(entities: list[Any]) -> List[SimpleNamespace]:
    """Wave 7 W7-10: project ``EntityWithLineage`` rows into the
    ``SimpleNamespace`` shape ``_to_ui_dict`` consumes.

    Mirrors the legacy ``_adapt_nodes`` projection so the UI dict
    builder's contract stays unchanged. ``description`` prefers
    ``compacted_description`` (W7-1 derived cache) and falls back to
    the joined per-doc parts so freshly-indexed entities (compactor
    not yet run) still show usable text. ``source_chunk_count``
    aggregates chunk ids across every lineage member.
    """
    out: List[SimpleNamespace] = []
    for entity in entities:
        compacted = getattr(entity, "compacted_description", None)
        if compacted:
            description = compacted
        else:
            parts = getattr(entity, "description_parts", ()) or ()
            description = "\n\n".join(p.text for p in parts if getattr(p, "text", ""))
        chunk_ids: set[str] = set()
        for member in getattr(entity, "source_lineage", ()) or ():
            for cid in getattr(member, "chunk_ids", ()) or ():
                chunk_ids.add(str(cid))
        props = {
            "entity_id": entity.name,
            "entity_name": entity.name,
            "entity_type": entity.entity_type,
            "description": description,
            "source_chunk_count": len(chunk_ids),
        }
        out.append(
            SimpleNamespace(
                id=entity.name,
                labels=[entity.entity_type] if entity.entity_type else [entity.name],
                properties=props,
                entity_id=entity.name,
                entity_name=entity.name,
                entity_type=entity.entity_type,
                description=description,
                source_chunk_count=len(chunk_ids),
            )
        )
    return out


def _adapt_lineage_relations(relations: list[Any]) -> List[SimpleNamespace]:
    """Wave 7 W7-10: project ``RelationWithLineage`` rows into the
    ``SimpleNamespace`` shape ``_to_ui_dict`` consumes.

    The lineage relation has no ``weight`` field (legacy
    artefact) — we surface ``1.0`` constant so the UI dict builder
    keeps a stable shape. Description prefers ``compacted_description``
    over the joined per-doc parts (mirror ``_adapt_lineage_entities``).
    """
    out: List[SimpleNamespace] = []
    for rel in relations:
        compacted = getattr(rel, "compacted_description", None)
        if compacted:
            description = compacted
        else:
            parts = getattr(rel, "description_parts", ()) or ()
            description = "\n\n".join(p.text for p in parts if getattr(p, "text", ""))
        chunk_ids: set[str] = set()
        for member in getattr(rel, "evidence_lineage", ()) or ():
            for cid in getattr(member, "chunk_ids", ()) or ():
                chunk_ids.add(str(cid))
        props = {
            "weight": 1.0,
            "description": description,
            "keywords": "",
            "source_chunk_count": len(chunk_ids),
        }
        out.append(
            SimpleNamespace(
                id=f"{rel.source}->{rel.target}",
                type="DIRECTED",
                source=rel.source,
                target=rel.target,
                properties=props,
                weight=1.0,
                description=description,
                keywords="",
                source_chunk_count=len(chunk_ids),
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
