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

from asgiref.sync import async_to_sync

from aperag.context.full_text import create_index, delete_index
from aperag.db.models import CollectionStatus
from aperag.db.ops import db_ops
from aperag.embed.base_embedding import get_collection_embedding_service_sync
from aperag.graph import lightrag_holder
from aperag.tasks.index import get_collection_config_settings
from aperag.utils.utils import (
    generate_fulltext_index_name,
    generate_qa_vector_db_collection_name,
    generate_vector_db_collection_name,
)
from config.celery import app
from config.vector_db import get_vector_db_connector

logger = logging.getLogger(__name__)


def _init_collection_logic(collection_id: str, document_user_quota: int):
    """Internal function for collection initialization logic"""
    # Get collection from database using db_ops
    collection = db_ops.query_collection_by_id(collection_id)

    if not collection or collection.status == CollectionStatus.DELETED:
        return

    # Get embedding service using sync version
    vector_db_conn = get_vector_db_connector(collection=generate_vector_db_collection_name(collection_id=collection_id))
    _, vector_size = get_collection_embedding_service_sync(collection)
    # pre-create collection in vector db
    vector_db_conn.connector.create_collection(vector_size=vector_size)

    qa_vector_db_conn = get_vector_db_connector(
        collection=generate_qa_vector_db_collection_name(collection=collection_id)
    )
    qa_vector_db_conn.connector.create_collection(vector_size=vector_size)

    index_name = generate_fulltext_index_name(collection_id)
    create_index(index_name)

    # Update collection status using db_ops
    collection.status = CollectionStatus.ACTIVE
    db_ops.update_collection(collection)

    logger.info(f"Successfully initialized collection {collection_id}")


def _delete_collection_logic(collection_id: str):
    """Internal function for collection deletion logic"""
    # Get collection from database using db_ops
    collection = db_ops.query_collection_by_id(collection_id)

    if not collection:
        return

    _, enable_knowledge_graph = get_collection_config_settings(collection)

    # Delete lightrag documents for this collection
    if enable_knowledge_graph:

        async def _delete_lightrag():
            # Create new LightRAG instance without using cache for Celery tasks
            rag_holder = await lightrag_holder.get_lightrag_holder(collection, use_cache=False)
            await rag_holder.adelete_by_collection(collection_id)

        # Execute async deletion
        async_to_sync(_delete_lightrag)()

    # TODO remove the related collection in the vector db
    index_name = generate_fulltext_index_name(collection.id)
    delete_index(index_name)

    vector_db_conn = get_vector_db_connector(collection=generate_vector_db_collection_name(collection_id=collection_id))
    vector_db_conn.connector.delete_collection()

    qa_vector_db_conn = get_vector_db_connector(
        collection=generate_qa_vector_db_collection_name(collection=collection_id)
    )
    qa_vector_db_conn.connector.delete_collection()

    logger.info(f"Successfully deleted collection {collection_id}")


@app.task
def init_collection_task(collection_id, document_user_quota):
    """Celery task for collection initialization"""
    try:
        return _init_collection_logic(collection_id, document_user_quota)
    except Exception as e:
        logger.error(f"Failed to initialize collection {collection_id}: {e}")
        raise


@app.task
def delete_collection_task(collection_id):
    """Celery task for collection deletion"""
    try:
        return _delete_collection_logic(collection_id)
    except Exception as e:
        logger.error(f"Failed to delete collection {collection_id}: {e}")
        raise
