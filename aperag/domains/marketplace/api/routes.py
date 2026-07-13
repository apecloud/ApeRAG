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

"""Read-only marketplace graph endpoints for embedding-map and entity-search.

These endpoints mirror workspace-only graph APIs but are safe for published
marketplace collections — they resolve to the collection owner's user_id so
that callers do not need to own the collection.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from aperag.db.models import User
from aperag.exceptions import CollectionMarketplaceAccessDeniedError, CollectionNotPublishedError
from aperag.service.marketplace_collection_service import marketplace_collection_service
from aperag.views.auth import optional_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["graph"])


@router.get("/marketplace/collections/{collection_id}/graph/embedding-map")
async def get_marketplace_graph_embedding_map(
    collection_id: str,
    max_nodes: int = Query(500, ge=1, le=5000),
    user: User = Depends(optional_user),
):
    """Get entity list for embedding map visualization (read-only, marketplace-safe)."""
    from aperag.service.graph_service import graph_service

    try:
        user_id = str(user.id) if user else ""
        marketplace_info = await marketplace_collection_service._check_marketplace_access(user_id, collection_id)
        owner_user_id = marketplace_info["owner_user_id"]
        return await graph_service.get_embedding_map(str(owner_user_id), collection_id, max_nodes)
    except CollectionNotPublishedError:
        raise HTTPException(status_code=404, detail="Collection not found or not published")
    except CollectionMarketplaceAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting marketplace graph embedding map {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/marketplace/collections/{collection_id}/graph/entity-search")
async def get_marketplace_graph_entity_search(
    collection_id: str,
    q: str = Query(..., min_length=1, description="Search query"),
    max_results: int = Query(50, ge=1, le=500),
    user: User = Depends(optional_user),
):
    """Search entities in the knowledge graph by name (read-only, marketplace-safe)."""
    from aperag.service.graph_service import graph_service

    try:
        user_id = str(user.id) if user else ""
        marketplace_info = await marketplace_collection_service._check_marketplace_access(user_id, collection_id)
        owner_user_id = marketplace_info["owner_user_id"]
        return await graph_service.search_entities(str(owner_user_id), collection_id, q, max_results)
    except CollectionNotPublishedError:
        raise HTTPException(status_code=404, detail="Collection not found or not published")
    except CollectionMarketplaceAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error searching marketplace graph entities {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
