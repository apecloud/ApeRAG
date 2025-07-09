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
from typing import Any, Dict

from aperag.exceptions import CollectionNotFoundException
from aperag.graph import lightrag_manager
from aperag.schema import view_models

logger = logging.getLogger(__name__)


class GraphIndexService:
    """Service for handling knowledge graph index operations"""

    def __init__(self):
        # Import here to avoid circular imports
        from aperag.service.collection_service import collection_service

        self.collection_service = collection_service

    async def merge_nodes(
        self,
        user_id: str,
        collection_id: str,
        entity_id1: str,
        entity_id2: str,
    ) -> Dict[str, Any]:
        """
        Merge two graph nodes into one

        Args:
            user_id: User ID
            collection_id: Collection ID
            entity_id1: First entity ID to merge
            entity_id2: Second entity ID to merge

        Returns:
            Dict with merge results

        Raises:
            CollectionNotFoundException: If collection is not found
            ValueError: If knowledge graph is not enabled for the collection
        """
        logger.info(f"Starting node merge for collection {collection_id}: {entity_id1} <-> {entity_id2}")

        # Get and validate collection
        collection = await self._get_and_validate_collection(user_id, collection_id)

        # Create LightRAG instance and perform merge
        rag = await lightrag_manager.create_lightrag_instance(collection)
        try:
            result = await rag.amerge_nodes(
                entity_id1=entity_id1,
                entity_id2=entity_id2,
                collection_id=collection_id,
            )

            logger.info(
                f"Node merge completed for collection {collection_id}: "
                f"{result.get('source_entity')} -> {result.get('target_entity')}, "
                f"redirected {result.get('redirected_edges', 0)} edges"
            )

            return result
        finally:
            await rag.finalize_storages()

    async def _get_and_validate_collection(self, user_id: str, collection_id: str):
        """
        Get collection database model and validate that knowledge graph is enabled

        Args:
            user_id: User ID
            collection_id: Collection ID

        Returns:
            Collection database model (needed for lightrag_manager)

        Raises:
            CollectionNotFoundException: If collection is not found
            ValueError: If knowledge graph is not enabled
        """
        # Validate that user has access to the collection
        view_collection: view_models.Collection = await self.collection_service.get_collection(user_id, collection_id)

        # Check if knowledge graph is enabled
        if not view_collection.config or not view_collection.config.enable_knowledge_graph:
            raise ValueError(f"Knowledge graph is not enabled for collection {collection_id}")

        # Get the database model (needed for lightrag_manager)
        db_collection = await self.collection_service.db_ops.query_collection(user_id, collection_id)
        if not db_collection:
            raise CollectionNotFoundException(collection_id)

        return db_collection


# Global service instance
graphindex_service = GraphIndexService()
