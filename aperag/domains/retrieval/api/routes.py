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

"""HTTP routes for the ``retrieval`` domain.

Mounted under ``/api/v2`` by ``aperag/app.py``. Route decorators keep
relative paths so a single ``prefix="/api/v2"`` include does not
produce ``/api/v2/api/v2/*``.

Auth dependency uses a local ``AuthenticatedUser`` ``Protocol``
(lesson 9a-ter) instead of ``aperag.db.models.User`` or ``Any``. The
concrete SQLAlchemy ``User`` row structurally satisfies the protocol
so ``Depends(required_user)`` wires up unchanged.
"""

import logging
from typing import Protocol

from fastapi import APIRouter, Depends, Request

from aperag.domains.retrieval.schemas import SearchRequest, SearchResult, SearchResultList
from aperag.domains.retrieval.service import retrieval_service
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user


class AuthenticatedUser(Protocol):
    """Minimal auth-context contract the ``retrieval`` domain depends on.

    The domain only reads ``id`` from the authenticated caller; every
    other field of the legacy ``User`` row is out of scope. Declaring
    this narrow protocol (instead of importing the SQLAlchemy model or
    falling back to ``Any``) keeps the Phase 0 strict ban clean **and**
    pins the non-``Any`` type contract per lesson 9a-ter. Phase 4
    identity hard-cut will promote this protocol into a canonical
    ``identity`` domain context.
    """

    id: object


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/collections/{collection_id}/searches",
    tags=["search"],
    response_model=SearchResult,
)
@audit(resource_type="search", api_name="CreateSearch")
async def create_search_view(
    request: Request,
    collection_id: str,
    data: SearchRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> SearchResult:
    """Execute a hybrid search against a collection.

    When the caller does not own the collection but it is published in
    the marketplace, the service falls back to the owner's namespace
    for provider-key lookup. ``data.save_to_history`` controls whether
    the result is persisted as a ``SearchHistory`` row.
    """
    return await retrieval_service.create_search(str(user.id), collection_id, data)


@router.delete(
    "/collections/{collection_id}/searches/{search_id}",
    tags=["search"],
    name="DeleteSearch",
)
@audit(resource_type="search", api_name="DeleteSearch")
async def delete_search_view(
    request: Request,
    collection_id: str,
    search_id: str,
    user: AuthenticatedUser = Depends(required_user),
):
    """Idempotent delete of a persisted search history row."""
    return await retrieval_service.delete_search(str(user.id), collection_id, search_id)


@router.get(
    "/collections/{collection_id}/searches",
    tags=["search"],
    response_model=SearchResultList,
)
async def list_searches_view(
    request: Request,
    collection_id: str,
    user: AuthenticatedUser = Depends(required_user),
) -> SearchResultList:
    """List persisted search history for the collection.

    Returns the typed empty shape ``SearchResultList(items=[])`` when
    no history has been recorded yet (lesson 9a) — never ``None`` and
    never a raw ``{}``.
    """
    result = await retrieval_service.list_searches(str(user.id), collection_id)
    if result is None:
        return SearchResultList(items=[])
    return result
