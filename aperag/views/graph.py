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

"""Graph HTTP routes.

``graphindex`` owns graph truth and merge primitives.
``graph_curation`` owns merge-suggestion discovery and review state.

The only removed route left in this file is the historical KG-eval
export endpoint.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from aperag.db.models import User
from aperag.exceptions import CollectionNotFoundException
from aperag.graph_curation import graph_curation_service
from aperag.schema import view_models
from aperag.service.graph_service import graph_service
from aperag.views.auth import required_user

logger = logging.getLogger(__name__)

router = APIRouter()

_KG_EVAL_REMOVAL_DETAIL = (
    "This legacy KG-eval export endpoint was removed together with the "
    "LightRAG-era graph workflow. See docs/zh-CN/design/graphindex_rewrite.md."
)


def _gone() -> HTTPException:
    """Uniform 410 response for the removed KG-eval route."""
    return HTTPException(status_code=410, detail=_KG_EVAL_REMOVAL_DETAIL)


@router.post("/collections/{collection_id}/graphs/nodes/merge", tags=["graph"])
async def merge_nodes_view(
    request: Request,
    collection_id: str,
    payload: dict = Body(...),
    user: User = Depends(required_user),
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
    "/collections/{collection_id}/graphs/merge-suggestions/{suggestion_id}/action",
    tags=["graph"],
)
async def handle_suggestion_action_view(
    request: Request,
    collection_id: str,
    suggestion_id: str,
    payload: view_models.SuggestionActionRequest = Body(...),
    user: User = Depends(required_user),
) -> dict:
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


@router.post("/collections/{collection_id}/graphs/merge-suggestions", tags=["graph"])
async def merge_suggestions_view(
    request: Request,
    collection_id: str,
    payload: view_models.MergeSuggestionsRequest = Body(default_factory=view_models.MergeSuggestionsRequest),
    user: User = Depends(required_user),
) -> dict:
    del payload
    try:
        return await graph_curation_service.start_run(str(user.id), collection_id)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/collections/{collection_id}/graphs/merge-suggestions", tags=["graph"])
async def get_merge_suggestions_view(
    request: Request,
    collection_id: str,
    user: User = Depends(required_user),
) -> dict:
    try:
        return await graph_curation_service.get_latest(str(user.id), collection_id)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/collections/{collection_id}/graphs/export/kg-eval", tags=["graph"])
async def export_kg_eval_view(request: Request, collection_id: str) -> dict:
    raise _gone()
