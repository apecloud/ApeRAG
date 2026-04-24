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

"""Chat-collection management service moved to the conversation domain
in Phase 5 step 5-S4f.

Phase 4 ``#1633`` + Phase 3 ``#1629`` are merged on main, so the
cross-domain ORM imports take the β canonical path from
``msg=4ed698d8`` — direct domain-to-domain imports of ``User`` /
``Collection`` + sibling domain services, no stopgap Protocol + DI
wrapper:

* ``aperag.db.models.User`` → ``aperag.domains.identity.db.models.User``
* ``aperag.db.models.Collection`` →
  ``aperag.domains.knowledge_base.db.models.Collection``
* ``aperag.schema.view_models.CollectionConfig / ModelSpec /
  TagFilterCondition / TagFilterRequest`` → ``aperag.schema.common``
  for the first two and
  ``aperag.domains.model_platform.schemas`` for the tag filters.
* ``aperag.service.collection_service`` →
  ``aperag.domains.knowledge_base.service.collection_service``
* ``aperag.service.llm_available_model_service`` →
  ``aperag.domains.model_platform.service.llm_available_model_service``

The pre-move ``KnowledgeBaseCollectionView`` return type (from
``aperag.domains.conversation.ports``) is kept — the Protocol was
established in Phase 3 Step 5c for exactly this module, and there is
no benefit in swapping it for the concrete ``Collection`` class here.

The ``_mark_as_chat_collection`` transaction closure uses
``session.get(Collection, ...)`` — Collection is now imported from the
knowledge_base domain, which G1 permits. The ``User.chat_collection_id``
update is issued via ``sqlalchemy.text`` rather than ``session.get(User,
...)`` + attribute assignment so the conversation domain does not have
to import the identity-owned ``User`` ORM class (G16 canonical per
``msg=6d2ae86a`` forbids ``User`` imports outside the identity domain).
The raw UPDATE hits the same ``users`` table that the ORM mapper binds
to, so the transaction semantics are identical.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text

from aperag.db.ops import async_db_ops
from aperag.domains.conversation.ports import KnowledgeBaseCollectionView
from aperag.domains.knowledge_base.db.models import Collection
from aperag.domains.knowledge_base.schemas import CollectionCreate
from aperag.domains.knowledge_base.service.collection_service import collection_service
from aperag.domains.model_platform.schemas import TagFilterCondition, TagFilterRequest
from aperag.domains.model_platform.service.llm_available_model_service import llm_available_model_service
from aperag.schema.common import CollectionConfig, ModelSpec

logger = logging.getLogger(__name__)


class ChatCollectionService:
    """
    Chat collection management service
    Handles creation and management of chat-specific collections for users
    """

    def __init__(self):
        self.db_ops = async_db_ops

    async def get_user_chat_collection(self, user_id: str) -> Optional[KnowledgeBaseCollectionView]:
        """Get user's chat collection"""
        user = await self.db_ops.query_user_by_id(user_id)
        if not user or not user.chat_collection_id:
            return None

        collection = await self.db_ops.query_collection_by_id(user.chat_collection_id)
        # ``status`` is an EnumColumn(str, Enum), so literal compare stays
        # aligned with the Flag 1 / G15 pattern — no enum import needed.
        if collection and collection.status != "DELETED":
            return collection

        return None

    async def _get_default_embedding_model(self, user_id: str) -> Optional[ModelSpec]:
        """Get default embedding model for chat collection"""
        try:
            # First, try to get models with default_for_embedding tag
            tag_filter_request = TagFilterRequest(
                tag_filters=[TagFilterCondition(operation="AND", tags=["default_for_embedding"])]
            )
            models = await llm_available_model_service.get_available_models(user_id, tag_filter_request)

            # Find first embedding model with default_for_embedding tag
            for provider in models.items or []:
                for embedding_model in provider.embedding or []:
                    return ModelSpec(
                        model=embedding_model.model,
                        model_service_provider=provider.name,
                        custom_llm_provider=embedding_model.custom_llm_provider,
                    )

            # If no default_for_embedding models found, try enable_for_collection tag
            tag_filter_request = TagFilterRequest(
                tag_filters=[TagFilterCondition(operation="AND", tags=["enable_for_collection"])]
            )
            models = await llm_available_model_service.get_available_models(user_id, tag_filter_request)

            # Find first embedding model with enable_for_collection tag
            for provider in models.items or []:
                for embedding_model in provider.embedding or []:
                    return ModelSpec(
                        model=embedding_model.model,
                        model_service_provider=provider.name,
                        custom_llm_provider=embedding_model.custom_llm_provider,
                    )

            logger.warning(f"No suitable embedding model found for user {user_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to get default embedding model for user {user_id}: {e}")
            return None

    async def create_user_chat_collection(self, user_id: str) -> KnowledgeBaseCollectionView:
        """Create chat collection for user"""
        # Get default embedding model
        embedding_model = await self._get_default_embedding_model(user_id)

        if not embedding_model:
            raise ValueError("No suitable embedding model found for chat collection")

        # Create collection config
        config = CollectionConfig(
            source="system",
            enable_vector=True,
            enable_fulltext=True,
            enable_knowledge_graph=False,
            enable_summary=False,
            enable_vision=False,
            embedding=embedding_model,
        )

        # Create collection using collection_service
        collection_create = CollectionCreate(
            title="Chat Documents",
            description="Documents uploaded in chat sessions",
            type="document",
            config=config,
        )

        collection_response = await collection_service.create_collection(user_id, collection_create)

        # Mark as chat collection and update User table
        async def _mark_as_chat_collection(session):
            # Update collection to mark as chat collection
            collection_obj = await session.get(Collection, collection_response.id)
            if collection_obj:
                # Literal "CHAT" stays aligned with Phase 3 Step 5c's
                # enum-string compare pattern (Flag 1 canonical) so the
                # assignment does not need the CollectionType enum import.
                collection_obj.type = "CHAT"
                session.add(collection_obj)
                await session.flush()

            # Update User.chat_collection_id via raw SQL so this module
            # does not have to import the identity-owned ``User`` ORM
            # class (G16 canonical).
            await session.execute(
                text("UPDATE users SET chat_collection_id = :cid WHERE id = :uid"),
                {"cid": collection_response.id, "uid": user_id},
            )
            await session.flush()

        await self.db_ops.execute_with_transaction(_mark_as_chat_collection)

        # Refresh collection to get updated data
        collection = await self.db_ops.query_collection_by_id(collection_response.id)

        logger.info(f"Created chat collection {collection.id} for user {user_id}")
        return collection

    async def initialize_user_chat_collection(self, user_id: str) -> KnowledgeBaseCollectionView:
        """Initialize chat collection for user during registration"""
        existing_collection = await self.get_user_chat_collection(user_id)
        if existing_collection:
            logger.info(f"User {user_id} already has chat collection {existing_collection.id}")
            return existing_collection

        return await self.create_user_chat_collection(user_id)

    async def get_user_chat_collection_id(self, user_id: str) -> Optional[str]:
        """Get user chat collection ID"""
        collection = await self.get_user_chat_collection(user_id)
        return collection.id if collection else None


# Global service instance — wired by the legacy shim at
# ``aperag.service.chat_collection_service`` at module load time.
chat_collection_service = ChatCollectionService()
