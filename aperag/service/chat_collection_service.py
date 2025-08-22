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

import json
import logging
from typing import Optional

from aperag.db.models import Collection, CollectionStatus, CollectionType, User
from aperag.db.ops import async_db_ops
from aperag.schema.view_models import CollectionConfig
from aperag.service.collection_service import dumpCollectionConfig
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)

# Chat collection default configuration
CHAT_COLLECTION_DEFAULT_CONFIG = {
    "vector_index": True,
    "fulltext_index": True,
    "graph_index": True,
    "summary_index": True,
    "enable_summary": False,
    "enable_ai_summary": False,
    "sources": [],
}


class ChatCollectionService:
    """
    Chat collection management service
    Handles creation and management of chat-specific collections for users
    """

    def __init__(self):
        self.db_ops = async_db_ops

    async def get_user_chat_collection(self, user_id: str) -> Optional[Collection]:
        """Get user's chat collection"""
        user = await self.db_ops.query_user_by_id(user_id)
        if not user or not user.chat_collection_id:
            return None

        collection = await self.db_ops.query_collection_by_id(user.chat_collection_id)
        if collection and collection.status != CollectionStatus.DELETED:
            return collection

        return None

    async def create_user_chat_collection(self, user_id: str) -> Collection:
        """Create chat collection for user"""
        
        async def _create_chat_collection(session):
            # Create chat collection config
            config = CollectionConfig(**CHAT_COLLECTION_DEFAULT_CONFIG)
            config_str = dumpCollectionConfig(config)

            # Create collection
            collection = Collection(
                user=user_id,
                title="Chat Documents",
                description="Documents uploaded in chat sessions",
                type=CollectionType.DOCUMENT,
                status=CollectionStatus.ACTIVE,
                config=config_str,
                is_chat_collection=True,
                gmt_created=utc_now(),
                gmt_updated=utc_now(),
            )
            session.add(collection)
            await session.flush()
            await session.refresh(collection)

            # Update User table to link chat collection
            user = await session.get(User, user_id)
            if user:
                user.chat_collection_id = collection.id
                session.add(user)
                await session.flush()

            return collection

        collection = await self.db_ops.execute_with_transaction(_create_chat_collection)
        
        logger.info(f"Created chat collection {collection.id} for user {user_id}")
        return collection

    async def initialize_user_chat_collection(self, user_id: str) -> Collection:
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


# Create a global service instance
chat_collection_service = ChatCollectionService()
