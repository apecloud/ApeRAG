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

"""Persistence-aware retrieval service.

This is the thin wrapper that backs the three HTTP handlers mounted at
``/api/v2/collections/{id}/searches*``: ``create_search``,
``list_searches``, ``delete_search``. It delegates the actual hybrid
search orchestration to
``aperag.domains.retrieval.pipeline.search_pipeline_service`` and
handles the ``SearchHistory`` persistence path plus the marketplace-
backed collection lookup that makes a published collection searchable
by a non-owner user.

Method bodies were lifted from
``aperag.service.collection_service.CollectionService`` (methods
``create_search``, ``list_searches``, ``delete_search``). The
marketplace-access branch was inlined against ``async_db_ops``
directly so the domain does not import
``aperag.service.marketplace_collection_service``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from aperag.db.ops import async_db_ops
from aperag.domains.retrieval.pipeline import search_pipeline_service
from aperag.domains.retrieval.schemas import (
    SearchRequest,
    SearchResult,
    SearchResultItem,
    SearchResultList,
)
from aperag.exceptions import CollectionNotFoundException

logger = logging.getLogger(__name__)

# Literal value of ``aperag.db.models.CollectionMarketplaceStatusEnum.PUBLISHED``.
# Referenced as a string so the retrieval domain does not need to import the
# SQLAlchemy enum (forbidden by the Phase 0 strict ban). The upstream enum is
# a string enum — ``.value == "PUBLISHED"`` — and the comparison below holds
# byte-for-byte against the legacy marketplace access check.
_MARKETPLACE_STATUS_PUBLISHED = "PUBLISHED"


class RetrievalService:
    """HTTP-facing retrieval / search history service."""

    def __init__(self) -> None:
        self.db_ops = async_db_ops

    async def execute_search_flow(
        self,
        data: SearchRequest,
        collection_id: str,
        search_user_id: str,
        chat_id: Optional[str] = None,
    ) -> Tuple[List[SearchResultItem], str]:
        """Execute search using direct Python orchestration."""
        return await search_pipeline_service.execute_search(
            data=data,
            collection_id=collection_id,
            search_user_id=search_user_id,
            chat_id=chat_id,
        )

    async def create_search(self, user: str, collection_id: str, data: SearchRequest) -> SearchResult:
        """Owner-or-marketplace-subscriber search + optional history write.

        A collection is reachable if either:

        1. ``user`` owns it directly, or
        2. the collection is published in the marketplace; in which
           case the search uses the *owner*'s user id as the
           ``search_user_id`` so provider-key / storage lookups resolve
           through the owner's namespace.

        When ``data.save_to_history`` is false we short-circuit the
        history write and return a transient result with ``id=None`` /
        ``created=None`` — the caller's responsibility to treat the
        missing id as "ephemeral".
        """
        collection = await self.db_ops.query_collection(user, collection_id)
        search_user_id = user

        if not collection:
            marketplace_owner = await self._resolve_marketplace_owner(collection_id)
            if marketplace_owner is None:
                raise CollectionNotFoundException(collection_id)
            search_user_id = marketplace_owner
            collection = await self.db_ops.query_collection(search_user_id, collection_id)
            if not collection:
                raise CollectionNotFoundException(collection_id)

        items, _ = await self.execute_search_flow(
            data=data,
            collection_id=collection_id,
            search_user_id=search_user_id,
            chat_id=None,
        )

        if data.save_to_history:
            record = await self.db_ops.create_search(
                user=user,
                collection_id=collection_id,
                query=data.query,
                vector_search=data.vector_search.model_dump() if data.vector_search else None,
                fulltext_search=data.fulltext_search.model_dump() if data.fulltext_search else None,
                graph_search=data.graph_search.model_dump() if data.graph_search else None,
                summary_search=data.summary_search.model_dump() if data.summary_search else None,
                vision_search=data.vision_search.model_dump() if data.vision_search else None,
                items=[item.model_dump() for item in items],
            )
            return SearchResult(
                id=record.id,
                query=record.query,
                vector_search=record.vector_search,
                fulltext_search=record.fulltext_search,
                graph_search=record.graph_search,
                summary_search=record.summary_search,
                vision_search=record.vision_search,
                items=items,
                created=record.gmt_created.isoformat(),
            )

        return SearchResult(
            id=None,
            query=data.query,
            vector_search=data.vector_search,
            fulltext_search=data.fulltext_search,
            graph_search=data.graph_search,
            summary_search=data.summary_search,
            vision_search=data.vision_search,
            items=items,
            created=None,
        )

    async def list_searches(self, user: str, collection_id: str) -> SearchResultList:
        """List persisted search history for ``(user, collection)``.

        Returns the typed empty shape ``SearchResultList(items=[])``
        when the collection has no history yet — never ``None`` and
        never a raw ``{}`` (lesson 9a).
        """
        collection = await self.db_ops.query_collection(user, collection_id)
        if not collection:
            raise CollectionNotFoundException(collection_id)

        searches = await self.db_ops.query_searches(user, collection_id)

        items: List[SearchResult] = []
        for search in searches or []:
            search_result_items = []
            for item_data in search.items or []:
                search_result_items.append(SearchResultItem(**item_data))
            items.append(
                SearchResult(
                    id=search.id,
                    query=search.query,
                    vector_search=search.vector_search,
                    fulltext_search=search.fulltext_search,
                    graph_search=search.graph_search,
                    summary_search=search.summary_search,
                    items=search_result_items,
                    created=search.gmt_created.isoformat(),
                )
            )
        return SearchResultList(items=items)

    async def delete_search(self, user: str, collection_id: str, search_id: str) -> Optional[bool]:
        """Delete search by ID (idempotent operation).

        Returns ``True`` on a real deletion, ``False`` when the row was
        already gone. Collection-not-found is still surfaced as an
        explicit exception so the HTTP layer can distinguish ``404
        collection`` from ``404 search row`` (the latter is suppressed
        as idempotent).
        """
        collection = await self.db_ops.query_collection(user, collection_id)
        if not collection:
            raise CollectionNotFoundException(collection_id)

        return await self.db_ops.delete_search(user, collection_id, search_id)

    # ------------------------------------------------------------------ helpers
    async def _resolve_marketplace_owner(self, collection_id: str) -> Optional[str]:
        """Return the owner user id if the collection is published in
        the marketplace, else ``None``.

        The legacy code path routed through
        ``aperag.service.marketplace_collection_service._check_marketplace_access``
        which also does subscription / ownership bookkeeping. For the
        narrow search path we only need the **owner id** so that the
        subsequent search runs under the owner's provider-key
        namespace; subscription bookkeeping is not needed here. This
        inlined helper keeps the domain inside the strict ban while
        preserving the same observable behavior.
        """
        try:
            marketplace = await self.db_ops.get_collection_marketplace_by_collection_id(collection_id)
        except Exception:
            return None
        if marketplace is None or getattr(marketplace, "status", None) != _MARKETPLACE_STATUS_PUBLISHED:
            return None

        try:
            collection = await self.db_ops.query_collection_by_id(collection_id)
        except Exception:
            return None
        if collection is None:
            return None
        return getattr(collection, "user", None)


retrieval_service = RetrievalService()


__all__ = ["RetrievalService", "retrieval_service"]
