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

"""Knowledge-base ``CollectionService`` (Phase 3 Step 5b2b).

The service body moved verbatim from ``aperag/service/collection_service.py``;
the only restructuring is a G1 boundary pass:

* Imports of ``aperag.db.models`` / ``aperag.schema.view_models`` /
  ``aperag.service.*`` were rewritten to canonical paths under
  ``aperag/domains/knowledge_base`` and ``aperag/domains/retrieval``.
* Cross-service dependencies (``marketplace_service``,
  ``marketplace_collection_service``, ``search_pipeline_service``,
  ``quota_service``) are reached via the Phase 3 Step 5b2a
  consumer-owned Protocols in ``aperag.domains.knowledge_base.ports``
  and the module-level DI setters below. The legacy shim at
  ``aperag/service/collection_service.py`` wires the concrete instances
  at import time for Phase 3; Step 5b2c moves the wire-up into
  ``aperag/app.py`` startup as the canonical location.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, Tuple

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.domains.knowledge_base.db.models import (
    Collection as CollectionRow,
)
from aperag.domains.knowledge_base.db.models import (
    CollectionStatus,
    CollectionType,
)
from aperag.domains.knowledge_base.ports import (
    MarketplaceCollectionOps,
    MarketplaceOps,
    QuotaOps,
    SearchPipelineOps,
)
from aperag.domains.knowledge_base.schemas import (
    Collection,
    CollectionCreate,
    CollectionUpdate,
    CollectionView,
    CollectionViewList,
)
from aperag.domains.knowledge_base.tasks import collection_delete_task, collection_init_task
from aperag.domains.retrieval.schemas import (
    SearchRequest,
    SearchResult,
    SearchResultItem,
    SearchResultList,
)
from aperag.config import settings
from aperag.exceptions import ValidationException
from aperag.schema.common import PageResult, project_vector_backend_info
from aperag.schema.utils import dumpCollectionConfig, parseCollectionConfig
from aperag.utils.constant import QuotaType
from aperag.utils.utils import utc_now
from aperag.views.utils import validate_source_connect_config

logger = logging.getLogger(__name__)


# ---------- Consumer-owned Protocol DI setters (lesson 9a-quad) ---------- #
#
# Module-level singletons wired by ``aperag/app.py`` startup (Step 5b2c)
# with concrete instances that structurally satisfy the Protocols in
# ``aperag.domains.knowledge_base.ports``. Until the wire-up runs the
# getters raise ``RuntimeError`` so a test or request that reaches KB
# before the container is bootstrapped fails loudly instead of limping
# along with ``None``.

_marketplace_ops: MarketplaceOps | None = None
_marketplace_collection_ops: MarketplaceCollectionOps | None = None
_search_pipeline_ops: SearchPipelineOps | None = None
_quota_ops: QuotaOps | None = None


def set_marketplace_ops(ops: MarketplaceOps) -> None:
    global _marketplace_ops
    _marketplace_ops = ops


def set_marketplace_collection_ops(ops: MarketplaceCollectionOps) -> None:
    global _marketplace_collection_ops
    _marketplace_collection_ops = ops


def set_search_pipeline_ops(ops: SearchPipelineOps) -> None:
    global _search_pipeline_ops
    _search_pipeline_ops = ops


def set_quota_ops(ops: QuotaOps) -> None:
    global _quota_ops
    _quota_ops = ops


def _get_marketplace_ops() -> MarketplaceOps:
    if _marketplace_ops is None:
        raise RuntimeError(
            "knowledge_base.collection_service: marketplace_ops not wired. "
            "Call set_marketplace_ops() at app startup (Step 5b2c)."
        )
    return _marketplace_ops


def _get_marketplace_collection_ops() -> MarketplaceCollectionOps:
    if _marketplace_collection_ops is None:
        raise RuntimeError(
            "knowledge_base.collection_service: marketplace_collection_ops not wired. "
            "Call set_marketplace_collection_ops() at app startup (Step 5b2c)."
        )
    return _marketplace_collection_ops


def _get_search_pipeline_ops() -> SearchPipelineOps:
    if _search_pipeline_ops is None:
        raise RuntimeError(
            "knowledge_base.collection_service: search_pipeline_ops not wired. "
            "Call set_search_pipeline_ops() at app startup (Step 5b2c)."
        )
    return _search_pipeline_ops


def _get_quota_ops() -> QuotaOps:
    if _quota_ops is None:
        raise RuntimeError(
            "knowledge_base.collection_service: quota_ops not wired. Call set_quota_ops() at app startup (Step 5b2c)."
        )
    return _quota_ops


class CollectionService:
    """Collection service that handles business logic for collections"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    async def validate_collection_models(self, user: str, config) -> None:
        if config is None:
            return
        from aperag.domains.model_platform.schemas import ModelUseScenario
        from aperag.domains.model_platform.service.model_service import model_platform_service

        completion = getattr(config, "completion", None)
        if completion and completion.model_id:
            await model_platform_service.ensure_model_allowed_for_scenario(
                user, completion.model_id, ModelUseScenario.COLLECTION_COMPLETION
            )
        embedding = getattr(config, "embedding", None)
        if embedding and embedding.model_id:
            await model_platform_service.ensure_model_allowed_for_scenario(
                user, embedding.model_id, ModelUseScenario.COLLECTION_EMBEDDING
            )

    async def build_collection_response(self, instance: CollectionRow) -> Collection:
        """Build Collection response object for API return."""
        return Collection(
            id=instance.id,
            title=instance.title,
            description=instance.description,
            # Wave 10 §K.13: ``Collection.summary`` is auto-generated by
            # the regen reconciler. Include it on every read so the FE
            # settings page can render the textarea after a regen call —
            # without this the BE response is always ``summary=null``
            # even when the DB column has a fresh value, and the user
            # sees the placeholder "暂无摘要" despite a green toast
            # (per @earayu2 msg=e4120886 reproducer).
            summary=getattr(instance, "summary", None),
            type=instance.type,
            status=getattr(instance, "status", None),
            config=parseCollectionConfig(instance.config),
            # task #61 P1-D3 (PR for #87): static read-only projection of
            # the deployment vector backend identity + capability matrix.
            # The value is identical for every collection in the
            # deployment because ``settings.vector_db_type`` is a
            # deployment-wide env var (``aperag/config.py``); the
            # projection is intentionally not persisted per row. Returns
            # ``None`` when the configured backend is not in the static
            # capability matrix so the FE can render a placeholder
            # without a hard failure on misconfigured deployments.
            vector_backend=project_vector_backend_info(settings.vector_db_type),
            created=instance.gmt_created.isoformat(),
            updated=instance.gmt_updated.isoformat(),
        )

    async def create_collection(self, user: str, collection: CollectionCreate) -> Collection:
        collection_config = collection.config
        if collection.type != CollectionType.DOCUMENT:
            raise ValidationException("collection type is not supported")

        is_validate, error_msg = validate_source_connect_config(collection_config)
        if not is_validate:
            raise ValidationException(error_msg)
        await self.validate_collection_models(user, collection_config)

        # Create collection and consume quota in a single transaction
        async def _create_collection_with_quota(session):
            # Check and consume quota within the transaction
            await _get_quota_ops().check_and_consume_quota(user, "max_collection_count", 1, session)

            # Create collection within the same transaction
            config_str = dumpCollectionConfig(collection_config) if collection.config is not None else None

            instance = CollectionRow(
                user=user,
                title=collection.title,
                description=collection.description,
                type=collection.type,
                status=CollectionStatus.ACTIVE,
                config=config_str,
                gmt_created=utc_now(),
                gmt_updated=utc_now(),
            )
            session.add(instance)
            await session.flush()
            await session.refresh(instance)

            return instance

        instance = await self.db_ops.execute_with_transaction(_create_collection_with_quota)

        # Wave 10 §K.13: collection-level summary + description are now
        # auto-generated by the reconciler-driven Wave 10 regen pipeline
        # (Chunks C-E). The legacy ``collection_summary_service`` trigger
        # is gone; the new reconciler hook picks the collection up on its
        # next 30s sweep once docs land.

        # Initialize collection based on type. Pattern C (fire-and-forget)
        # per architect msg=3890c9d7 — wrap in asyncio.create_task so the
        # HTTP response returns immediately; failures log + are recovered
        # by the next reconciler scan (Wave 2 §I.3 + commit-5 follow-up
        # wires this lane into the periodic loop).
        document_user_quota = await self.db_ops.query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
        asyncio.create_task(asyncio.to_thread(collection_init_task, instance.id, document_user_quota))

        return await self.build_collection_response(instance)

    async def list_collections_view(
        self, user_id: str, include_subscribed: bool = True, page: int = 1, page_size: int = 20
    ) -> CollectionViewList:
        """
        Get user's collection list (lightweight view)

        Args:
            user_id: User ID
            include_subscribed: Whether to include subscribed collections, default True
            page: Page number
            page_size: Page size
        """
        items = []

        # 1. Get user's owned collections with marketplace info
        owned_collections_data = await self.db_ops.query_collections_with_marketplace_info(user_id)

        for row in owned_collections_data:
            is_published = row.marketplace_status == "PUBLISHED"
            items.append(
                CollectionView(
                    id=row.id,
                    title=row.title,
                    description=row.description,
                    type=row.type,
                    status=row.status,
                    created=row.gmt_created,
                    updated=row.gmt_updated,
                    is_published=is_published,
                    published_at=row.published_at if is_published else None,
                    owner_user_id=row.user,
                    owner_username=row.owner_username,
                    subscription_id=None,  # Own collection, subscription_id is None
                    subscribed_at=None,
                )
            )

        # 2. Get subscribed collections if needed (optimized - no N+1 queries)
        if include_subscribed:
            try:
                # Get subscribed collections data with all needed fields in one query
                subscribed_collections_data, _ = await self.db_ops.list_user_subscribed_collections(
                    user_id,
                    page=1,
                    page_size=1000,  # Get all subscriptions for now
                )

                for data in subscribed_collections_data:
                    is_published = data["marketplace_status"] == "PUBLISHED"
                    items.append(
                        CollectionView(
                            id=data["id"],
                            title=data["title"],
                            description=data["description"],
                            type=data["type"],
                            status=data["status"],
                            created=data["gmt_created"],
                            updated=data["gmt_updated"],
                            is_published=is_published,
                            published_at=data["published_at"] if is_published else None,
                            owner_user_id=data["owner_user_id"],
                            owner_username=data["owner_username"],
                            subscription_id=data["subscription_id"],
                            subscribed_at=data["gmt_subscribed"],
                        )
                    )
            except Exception as e:
                # If getting subscriptions fails, log and continue with owned collections
                logger.warning(f"Failed to get subscribed collections for user {user_id}: {e}")

        # 3. Sort by update time
        items.sort(key=lambda x: x.updated or x.created, reverse=True)

        # 4. Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_items = items[start_idx:end_idx]

        return CollectionViewList(
            items=paginated_items, pageResult=PageResult(total=len(items), page=page, page_size=page_size)
        )

    async def get_collection(self, user: str, collection_id: str) -> Collection:
        from aperag.exceptions import CollectionNotFoundException

        if not user:
            await _get_marketplace_ops().validate_marketplace_collection(collection_id)
            collection = await self.db_ops.query_collection_by_id(collection_id)
        else:
            collection = await self.db_ops.query_collection(user, collection_id)

        if collection is None:
            raise CollectionNotFoundException(collection_id)
        return await self.build_collection_response(collection)

    @staticmethod
    def _embedding_identity(cfg) -> Optional[tuple]:
        """Return a tuple that uniquely identifies an embedding model binding.

        Returns None when no embedding is configured yet (e.g. freshly created
        collection whose user has not filled it in), so first-time assignment is
        still allowed.
        """
        if cfg is None:
            return None
        emb = getattr(cfg, "embedding", None)
        if emb is None:
            return None
        model = getattr(emb, "model_id", None)
        if not model:
            return None
        return (model,)

    def _reject_embedding_change(self, instance: CollectionRow, update: CollectionUpdate) -> None:
        """Raise ValidationException if the update tries to change the embedding binding."""
        if update.config is None:
            return
        try:
            old_cfg = parseCollectionConfig(instance.config) if instance.config else None
        except Exception:
            old_cfg = None

        old_id = self._embedding_identity(old_cfg)
        new_id = self._embedding_identity(update.config)

        if old_id is None:
            # First-time binding is allowed.
            return
        if new_id is None:
            raise ValidationException(
                "Embedding model of an existing collection cannot be cleared. "
                "Keep the original embedding configuration or create a new collection."
            )
        if old_id != new_id:
            old_model = old_id[0]
            new_model = new_id[0]
            raise ValidationException(
                "Embedding model of an existing collection cannot be changed "
                f"(current: {old_model!r}, requested: {new_model!r}). "
                "Different embedding models produce vectors of different dimensions "
                "and/or incompatible semantics, which would leave existing vectors "
                "orphaned and break retrieval. Please create a new collection and "
                "re-ingest your documents if a different model is required."
            )

    async def update_collection(self, user: str, collection_id: str, collection: CollectionUpdate) -> Collection:
        from aperag.exceptions import CollectionNotFoundException

        # First check if collection exists
        instance = await self.db_ops.query_collection(user, collection_id)
        if instance is None:
            raise CollectionNotFoundException(collection_id)

        # Guardrail: embedding model/provider of an existing collection MUST NOT change.
        # Different embedding models produce vectors with different dimensions and/or
        # incompatible semantics. Silently switching will (a) leak orphan points in the
        # previous global collection (under multi-tenant mode), (b) break retrieval
        # recall, (c) in legacy mode even cause Qdrant to reject writes with a
        # dimension mismatch error. Rejecting here is the least surprising behavior;
        # users who truly need a different model should create a new collection and
        # re-ingest their data.
        self._reject_embedding_change(instance, collection)
        await self.validate_collection_models(user, collection.config)

        # Direct call to repository method, which handles its own transaction
        config_str = dumpCollectionConfig(collection.config)

        updated_instance = await self.db_ops.update_collection_by_id(
            user=user,
            collection_id=collection_id,
            title=collection.title,
            description=collection.description,
            config=config_str,
        )

        # Wave 10 §K.13: legacy ``collection_summary_service`` trigger
        # removed; reconciler hook picks up doc-change deltas on the
        # next sweep and regenerates summary + description automatically.

        if not updated_instance:
            raise CollectionNotFoundException(collection_id)

        return await self.build_collection_response(updated_instance)

    async def delete_collection(self, user: str, collection_id: str) -> Optional[Collection]:
        """Delete collection by ID (idempotent operation)

        Returns the deleted collection or None if already deleted/not found
        """
        # Check if collection exists - if not, silently succeed (idempotent)
        collection = await self.db_ops.query_collection(user, collection_id)
        if collection is None:
            return None

        # Delete collection and release quota in a single transaction
        async def _delete_collection_with_quota(session):
            from sqlalchemy import select

            # Get collection within transaction
            stmt = select(CollectionRow).where(CollectionRow.id == collection_id, CollectionRow.user == user)
            result = await session.execute(stmt)
            collection_to_delete = result.scalars().first()

            if not collection_to_delete:
                return None

            # Mark collection as deleted
            collection_to_delete.status = CollectionStatus.DELETED
            collection_to_delete.gmt_deleted = utc_now()

            # Release quota within the same transaction
            await _get_quota_ops().release_quota(user, "max_collection_count", 1, session)

            await session.flush()
            await session.refresh(collection_to_delete)

            return collection_to_delete

        deleted_instance = await self.db_ops.execute_with_transaction(_delete_collection_with_quota)

        if deleted_instance:
            # Pattern A (durability-required) per architect msg=3890c9d7:
            # synchronously cascade the cleanup so a failure surfaces as
            # an HTTP 500 (the user can retry, and the periodic cleanup
            # loop sweeps any orphaned rows path-C-style). NOT
            # asyncio.create_task — losing this work = orphan rows + DB
            # corruption.
            await asyncio.to_thread(collection_delete_task, collection_id)
            return await self.build_collection_response(deleted_instance)

        return None

    async def execute_search_flow(
        self,
        data: SearchRequest,
        collection_id: str,
        search_user_id: str,
        chat_id: Optional[str] = None,
        flow_name: str = "search",
        flow_title: str = "Search",
    ) -> Tuple[List[SearchResultItem], str]:
        """Execute search using direct Python orchestration."""
        _ = (flow_name, flow_title)
        return await _get_search_pipeline_ops().execute_search(
            data=data,
            collection_id=collection_id,
            search_user_id=search_user_id,
            chat_id=chat_id,
        )

    async def create_search(self, user: str, collection_id: str, data: SearchRequest) -> SearchResult:
        from aperag.exceptions import CollectionNotFoundException

        # Try to find collection as owner first
        collection = await self.db_ops.query_collection(user, collection_id)
        search_user_id = user  # Default to current user for search operations

        if not collection:
            # If not found as owner, check if it's a marketplace collection
            try:
                marketplace_info = await _get_marketplace_collection_ops().check_marketplace_access(user, collection_id)
                # Use owner's user_id for search operations in marketplace collections
                search_user_id = marketplace_info["owner_user_id"]
                collection = await self.db_ops.query_collection(search_user_id, collection_id)
                if not collection:
                    raise CollectionNotFoundException(collection_id)
            except Exception:
                # If marketplace access also fails, raise original not found error
                raise CollectionNotFoundException(collection_id)

        # Execute search flow using helper method
        items, _ = await self.execute_search_flow(
            data=data,
            collection_id=collection_id,
            search_user_id=search_user_id,
            chat_id=None,  # No chat filtering for regular collection searches
            flow_name="search",
            flow_title="Search",
        )

        # Save to database only if save_to_history is True
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
        else:
            # Return search result without saving to database
            return SearchResult(
                id=None,  # No ID since not saved
                query=data.query,
                vector_search=data.vector_search,
                fulltext_search=data.fulltext_search,
                graph_search=data.graph_search,
                summary_search=data.summary_search,
                vision_search=data.vision_search,
                items=items,
                created=None,  # No creation time since not saved
            )

    async def list_searches(self, user: str, collection_id: str) -> SearchResultList:
        from aperag.exceptions import CollectionNotFoundException

        collection = await self.db_ops.query_collection(user, collection_id)
        if not collection:
            raise CollectionNotFoundException(collection_id)

        # Use DatabaseOps to query searches
        searches = await self.db_ops.query_searches(user, collection_id)

        items = []
        for search in searches:
            search_result_items = []
            for item_data in search.items:
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
        """Delete search by ID (idempotent operation)

        Returns True if deleted, None if already deleted/not found
        """
        from aperag.exceptions import CollectionNotFoundException

        collection = await self.db_ops.query_collection(user, collection_id)
        if not collection:
            raise CollectionNotFoundException(collection_id)

        return await self.db_ops.delete_search(user, collection_id, search_id)

    async def validate_collections_batch(self, user: str, collections: list[Any]) -> tuple[bool, str]:
        """
        Validate multiple collections in a single database call.

        Args:
            user: User identifier
            collections: List of collection objects to validate

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is empty.
        """
        if not collections:
            return True, ""

        # Extract collection IDs and validate they exist
        collection_ids = []
        for collection in collections:
            if not collection.id:
                return False, "Collection object missing 'id' field"
            collection_ids.append(collection.id)

        # Remove duplicates while preserving order
        unique_collection_ids = list(dict.fromkeys(collection_ids))

        try:
            # Single database call to get all collections
            db_collections = await self.db_ops.query_collections_by_ids(user, unique_collection_ids)

            # Create a set of found collection IDs for fast lookup
            found_collection_ids = {str(col.id) for col in db_collections}

            # Check if all requested collections were found
            for collection_id in unique_collection_ids:
                if collection_id not in found_collection_ids:
                    return False, f"Collection {collection_id} not found"

            return True, ""

        except Exception as e:
            return False, f"Failed to validate collections: {str(e)}"

    async def test_mineru_token(self, token: str) -> dict:
        """Test the MinerU API token."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "https://mineru.net/api/v4/extract-results/batch/test-token",
                    headers={"Authorization": f"Bearer {token}"},
                )
                return {"status_code": response.status_code, "data": response.json()}
            except httpx.RequestError as e:
                return {"status_code": 500, "data": {"msg": f"Request failed: {e}"}}


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
collection_service = CollectionService()


__all__ = [
    "CollectionService",
    "collection_service",
    "set_marketplace_ops",
    "set_marketplace_collection_ops",
    "set_search_pipeline_ops",
    "set_quota_ops",
]
