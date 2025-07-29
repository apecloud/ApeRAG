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

import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from aperag.db.models import User
from aperag.exceptions import CollectionNotFoundException
from aperag.schema import view_models
from aperag.utils.audit_decorator import audit

# Import authentication dependencies
from aperag.views.auth import current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/collections/{collection_id}/graphs/nodes/merge", tags=["graph"])
@audit(resource_type="index", api_name="MergeNodes")
async def merge_nodes_view(
    request: Request,
    collection_id: str,
    merge_request: view_models.NodeMergeRequest,
    user: User = Depends(current_user),
) -> view_models.NodeMergeResponse:
    """Merge multiple graph nodes into one"""
    from aperag.service.graph_service import graph_service

    logger.info(f"Merging nodes: entity_ids={merge_request.entity_ids} in collection {collection_id}")

    try:
        # Call graph service
        result = await graph_service.merge_nodes(
            user_id=str(user.id),
            collection_id=collection_id,
            entity_ids=merge_request.entity_ids,
            target_entity_data=merge_request.target_entity_data.model_dump(exclude_unset=True)
            if merge_request.target_entity_data
            else None,
        )
        return view_models.NodeMergeResponse(**result)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/collections/{collection_id}/graphs/merge-suggestions/{suggestion_id}/action", tags=["graph"])
@audit(resource_type="index", api_name="HandleSuggestionAction")
async def handle_suggestion_action_view(
    request: Request,
    collection_id: str,
    suggestion_id: str,
    action_request: view_models.SuggestionActionRequest,
    user: User = Depends(current_user),
) -> view_models.SuggestionActionResponse:
    """Accept or reject a merge suggestion"""
    from aperag.service.graph_service import graph_service

    logger.info(
        f"Handling suggestion action: {action_request.action} for suggestion {suggestion_id} in collection {collection_id}"
    )

    try:
        result = await graph_service.handle_suggestion_action(
            user_id=str(user.id),
            collection_id=collection_id,
            suggestion_id=suggestion_id,
            action=action_request.action,
            target_entity_data=action_request.target_entity_data.model_dump(exclude_unset=True)
            if action_request.target_entity_data
            else None,
        )
        return view_models.SuggestionActionResponse(**result)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/collections/{collection_id}/graphs/merge-suggestions", tags=["graph"])
@audit(resource_type="index", api_name="GenerateMergeSuggestions")
async def merge_suggestions_view(
    request: Request,
    collection_id: str,
    suggestions_request: Optional[view_models.MergeSuggestionsRequest] = Body(None),
    user: User = Depends(current_user),
) -> view_models.MergeSuggestionsResponse:
    """Get cached suggestions or generate new ones using LLM analysis"""
    from aperag.service.graph_service import graph_service

    # If no request body provided, create default request
    if suggestions_request is None:
        suggestions_request = view_models.MergeSuggestionsRequest()

    logger.info(
        f"Getting merge suggestions for collection {collection_id}, "
        f"max_suggestions={suggestions_request.max_suggestions}, "
        f"force_refresh={suggestions_request.force_refresh}"
    )

    try:
        # Call graph service
        result = await graph_service.get_or_generate_merge_suggestions(
            user_id=str(user.id),
            collection_id=collection_id,
            max_suggestions=suggestions_request.max_suggestions,
            max_concurrent_llm_calls=suggestions_request.max_concurrent_llm_calls,
            force_refresh=suggestions_request.force_refresh,
        )

        logger.info(
            f"Returned {len(result['suggestions'])} merge suggestions for collection {collection_id} "
            f"(from_cache={result['from_cache']}, {result['processing_time_seconds']:.2f}s)"
        )

        return view_models.MergeSuggestionsResponse(**result)
    except CollectionNotFoundException:
        raise HTTPException(status_code=404, detail="Collection not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
