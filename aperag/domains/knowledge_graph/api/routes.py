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

"""HTTP routes for the ``knowledge_graph`` domain.

Mounted under ``/api/v2`` by ``aperag/app.py`` (see the PR-2a pattern).
Route decorators keep relative paths so a single ``prefix="/api/v2"``
include does not result in ``/api/v2/api/v2/*``.

All handlers declare their auth dependency through the
``AuthenticatedUser`` ``Protocol`` (lesson 9a-ter) instead of importing
``aperag.db.models.User`` or falling back to ``Any``. The concrete
SQLAlchemy ``User`` row structurally satisfies the protocol, so
``Depends(required_user)`` wires up unchanged.
"""

import logging
from typing import Protocol

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from aperag.domains.identity.service.auth_dependencies import required_user
from aperag.domains.knowledge_graph.schemas import (
    GraphEmbeddingMapResponse,
    GraphEntitiesSearchResponse,
    GraphHybridResponse,
    GraphLabelsResponse,
    GraphSearchEntity,
    GraphSubgraphExpandRequest,
    GraphSubgraphExpandResponse,
    KnowledgeGraph,
    MergeSuggestionsRequest,
    SuggestionActionRequest,
)
from aperag.domains.knowledge_graph.service import graph_service
from aperag.exceptions import CollectionNotFoundException


class AuthenticatedUser(Protocol):
    """Minimal auth-context contract the ``knowledge_graph`` domain
    depends on.

    The domain only reads ``id`` from the authenticated caller (to
    validate collection ownership and attribute curation actions).
    Expressing the dependency as a narrow ``Protocol`` keeps the
    Phase 0 strict ban clean — the domain cannot drag in
    ``aperag.db.models.User`` — while still documenting the non-``Any``
    shape the handler depends on, per lesson 9a-ter. Phase 4 identity
    hard-cut will promote this local protocol into a canonical
    ``identity`` domain context.
    """

    id: object


logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------


@router.get(
    "/collections/{collection_id}/graphs/labels",
    tags=["graph"],
    response_model=GraphLabelsResponse,
)
async def get_graph_labels_view(
    request: Request,
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> GraphLabelsResponse:
    """Get all available node labels in the collection's knowledge graph.

    Empty list is a legitimate result — the collection simply hasn't
    been indexed yet — not a cue to 404 or fall back.
    """
    try:
        result = await graph_service.get_graph_labels(str(user.id), collection_id)
        if result is None:
            return GraphLabelsResponse(labels=[])
        return result
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/collections/{collection_id}/graphs",
    tags=["graph"],
    response_model=KnowledgeGraph,
)
async def get_knowledge_graph_view(
    request: Request,
    collection_id: str,
    label: str = "*",
    max_nodes: int = 1000,
    max_depth: int = 3,
    user: AuthenticatedUser = Depends(required_user),
) -> KnowledgeGraph:
    """Get knowledge graph — overview mode or subgraph mode."""
    # Validate parameters
    if not (1 <= max_nodes <= 10000):
        raise HTTPException(status_code=400, detail="max_nodes must be between 1 and 10000")
    if not (1 <= max_depth <= 10):
        raise HTTPException(status_code=400, detail="max_depth must be between 1 and 10")

    try:
        return await graph_service.get_knowledge_graph(str(user.id), collection_id, label, max_depth, max_nodes)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Write paths: node merge + curation run lifecycle
# ---------------------------------------------------------------------------


@router.post("/collections/{collection_id}/graphs/nodes/merge", tags=["graph"])
async def merge_nodes_view(
    request: Request,
    collection_id: str,
    payload: dict = Body(...),
    user: AuthenticatedUser = Depends(required_user),
) -> dict:
    """Merge N entities in a collection's knowledge graph into one.

    Request body:
    ``{"entity_ids": ["a", "b", "c"], "target_entity_id": "a" | null}``

    * ``entity_ids`` must contain at least two ids.
    * ``target_entity_id`` is the surviving entity; if omitted we pick
      the first id in ``entity_ids`` (callers that want "highest
      degree" auto-selection should do that on the client — the
      service layer intentionally does not embed a product policy).
    * The response echoes the merged description **after** LLM
      summarization, so the frontend can refresh the entity detail
      panel without a second fetch.
    * ``edges_redirected`` / ``edges_collapsed`` are always ``0`` by
      design (Wave 7 §K.12 invariant #9) — alias redirect happens at
      indexer write-time via the
      :class:`LineageGraphStoreWithAliasRedirect` decorator, not as
      part of the merge call, so the merge action has no "explicit
      edge re-anchor count" to surface. The fields are kept on the
      response shape for backward-compat with the existing frontend.
    """
    entity_ids = payload.get("entity_ids") or []
    if not isinstance(entity_ids, list) or len(entity_ids) < 2:
        raise HTTPException(status_code=400, detail="entity_ids must be a list with at least two entity ids")
    target = payload.get("target_entity_id") or entity_ids[0]
    sources = [eid for eid in entity_ids if eid != target]
    if not sources:
        raise HTTPException(status_code=400, detail="At least one source entity distinct from the target is required")

    try:
        return await graph_service.merge_entities(
            str(user.id),
            collection_id,
            target_entity_id=target,
            source_entity_ids=sources,
        )
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/collections/{collection_id}/graphs/merge-suggestions",
    tags=["graph"],
)
async def merge_suggestions_view(
    request: Request,
    collection_id: str,
    payload: MergeSuggestionsRequest = Body(default_factory=MergeSuggestionsRequest),
    user: AuthenticatedUser = Depends(required_user),
) -> dict:
    """Start a new graph-curation analysis run (idempotent)."""
    del payload  # body currently has no fields; kept to pin the request shape
    # Lazy import so the domain does not shackle its import graph to
    # ``aperag.graph_curation`` at module load time (it in turn imports
    # Celery tasks). This is not forbidden by the Phase 0 strict ban
    # (``aperag.graph_curation`` is infrastructure, not an aggregate).
    from aperag.graph_curation import graph_curation_service

    try:
        return await graph_curation_service.start_run(str(user.id), collection_id)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/collections/{collection_id}/graphs/merge-suggestions",
    tags=["graph"],
)
async def get_merge_suggestions_view(
    request: Request,
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> dict:
    """Return the latest run + its suggestions for this collection.

    When the collection has never started a curation run the response
    is the typed empty shape ``{"run": None, "suggestions": []}`` —
    never ``None`` and never ``{}``. Lesson 9a.
    """
    from aperag.graph_curation import graph_curation_service

    try:
        result = await graph_curation_service.get_latest(str(user.id), collection_id)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if result is None:
        return {"run": None, "suggestions": []}
    return result


@router.post(
    "/collections/{collection_id}/graphs/merge-suggestions/{suggestion_id}/action",
    tags=["graph"],
)
async def handle_suggestion_action_view(
    request: Request,
    collection_id: str,
    suggestion_id: str,
    payload: SuggestionActionRequest = Body(...),
    user: AuthenticatedUser = Depends(required_user),
) -> dict:
    """Accept or reject a merge suggestion."""
    from aperag.graph_curation import graph_curation_service

    try:
        return await graph_curation_service.handle_action(
            str(user.id),
            collection_id,
            suggestion_id,
            action=payload.action,
        )
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=exc.args[0] if exc.args else "Suggestion not found",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# Historical ``GET /api/v1/collections/{id}/graphs/export/kg-eval``
# endpoint was removed together with the legacy graph workflow.
# The ``aperag/views/graph.py`` 410-Gone shim that used to reply on the
# v1 path was deleted by the Phase 2 hard-cut; v1 KG paths now 404 at
# the FastAPI level, matching every other v1 KG URL. The canonical v2
# surface does not re-expose kg-eval. See
# ``docs/zh-CN/design/graphindex_rewrite.md``.


# ---------------------------------------------------------------------------
# Wave 7 §K.12.6/§K.12.7 — internal-only graph search endpoints (task #7)
#
# These three endpoints back the ``query_graph_entities`` /
# ``expand_graph_subgraph`` / ``get_entity_detail`` MCP tools defined in
# ``aperag/mcp/tools/graph_tools.py``. They are mounted under the same
# ``/api/v2`` prefix so the in-process MCP server can call them via
# httpx with ``API_BASE_URL`` (mirroring the existing
# ``aperag/mcp/tools/search_graph.py`` pattern).
#
# They are explicitly NOT user-facing: the frontend has no consumer
# (cuiwenbo task #9 verified backward-compat unaffected). They appear
# in OpenAPI regen so the SDK / agent runtime can bind to typed
# request / response models.
# ---------------------------------------------------------------------------


@router.get(
    "/collections/{collection_id}/graphs/entities/search",
    tags=["graph"],
    response_model=GraphEntitiesSearchResponse,
)
async def graph_entities_search_view(
    request: Request,
    collection_id: str,
    q: str,
    top_k: int = 10,
    user: AuthenticatedUser = Depends(required_user),
) -> GraphEntitiesSearchResponse:
    """Vector-recall the top-K entities matching ``q`` (Wave 7 §K.12.6).

    Empty / whitespace ``q`` returns ``entities=[]`` — same convention as
    the underlying :class:`GraphSearchService` so MCP tool callers see
    consistent empty-state behaviour.
    """
    if not (1 <= top_k <= 100):
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    try:
        return await graph_service.search_entities(str(user.id), collection_id, query=q, top_k=top_k)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post(
    "/collections/{collection_id}/graphs/subgraph/expand",
    tags=["graph"],
    response_model=GraphSubgraphExpandResponse,
)
async def graph_subgraph_expand_view(
    request: Request,
    collection_id: str,
    payload: GraphSubgraphExpandRequest = Body(...),
    user: AuthenticatedUser = Depends(required_user),
) -> GraphSubgraphExpandResponse:
    """Expand neighbours from anchor entities (Wave 7 §K.12.6).

    Backed by ``LineageGraphStore.expand_neighbors_n_hops`` via
    ``GraphSearchService.get_subgraph``. Backend implementations cap
    the result size — callers MUST not assume an unbounded subgraph.
    """
    try:
        return await graph_service.expand_subgraph(
            str(user.id),
            collection_id,
            entity_names=list(payload.entity_names),
            hops=payload.hops,
        )
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/collections/{collection_id}/graphs/entities/{name}",
    tags=["graph"],
    response_model=GraphSearchEntity,
)
async def graph_entity_detail_view(
    request: Request,
    collection_id: str,
    name: str,
    user: AuthenticatedUser = Depends(required_user),
) -> GraphSearchEntity:
    """Return the full record for a single entity (Wave 7 §K.12.6).

    Returns 404 when the entity does not exist in the lineage store
    (gc'd, never extracted, or wrong collection).
    """
    try:
        result = await graph_service.get_entity_detail(str(user.id), collection_id, entity_name=name)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return result


@router.get(
    "/collections/{collection_id}/graphs/embedding-map",
    tags=["graph"],
    response_model=GraphEmbeddingMapResponse,
)
async def graph_embedding_map_view(
    request: Request,
    collection_id: str,
    max_entities: int = 1000,
    user: AuthenticatedUser = Depends(required_user),
) -> GraphEmbeddingMapResponse:
    """Return projected entity coordinates for the hybrid graph view."""
    if not (1 <= max_entities <= 5000):
        raise HTTPException(status_code=400, detail="max_entities must be between 1 and 5000")
    try:
        return await graph_service.get_embedding_map(
            str(user.id),
            collection_id,
            max_entities=max_entities,
        )
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/collections/{collection_id}/graphs/hybrid",
    tags=["graph"],
    response_model=GraphHybridResponse,
)
async def graph_hybrid_view(
    request: Request,
    collection_id: str,
    max_entities: int = 1000,
    user: AuthenticatedUser = Depends(required_user),
) -> GraphHybridResponse:
    """Return positioned graph nodes and relation metadata for the hybrid view."""
    if not (1 <= max_entities <= 5000):
        raise HTTPException(status_code=400, detail="max_entities must be between 1 and 5000")
    try:
        return await graph_service.get_hybrid_graph(
            str(user.id),
            collection_id,
            max_entities=max_entities,
        )
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
