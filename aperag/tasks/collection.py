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
from aperag.db.models import Collection
from aperag.embed.base_embedding import get_collection_embedding_service
from aperag.graph import lightrag_holder
from aperag.schema.utils import parseCollectionConfig
from aperag.source.base import get_source
from aperag.tasks.index import get_collection_config_settings
from aperag.tasks.sync_documents_task import sync_documents
from aperag.utils.utils import (
    generate_fulltext_index_name,
    generate_qa_vector_db_collection_name,
    generate_vector_db_collection_name,
)
from config.celery import app
from config.vector_db import get_vector_db_connector

logger = logging.getLogger(__name__)


@app.task
def init_collection_task(collection_id, document_user_quota):
    collection = Collection.objects.get(id=collection_id)
    if collection.status == Collection.Status.DELETED:
        return

    vector_db_conn = get_vector_db_connector(collection=generate_vector_db_collection_name(collection_id=collection_id))
    _, vector_size = async_to_sync(get_collection_embedding_service)(collection)
    # pre-create collection in vector db
    vector_db_conn.connector.create_collection(vector_size=vector_size)

    qa_vector_db_conn = get_vector_db_connector(
        collection=generate_qa_vector_db_collection_name(collection=collection_id)
    )
    qa_vector_db_conn.connector.create_collection(vector_size=vector_size)

    index_name = generate_fulltext_index_name(collection_id)
    create_index(index_name)

    collection.status = Collection.Status.ACTIVE
    collection.save()

    source = get_source(parseCollectionConfig(collection.config))
    if source.sync_enabled():
        sync_documents.delay(collection_id=collection_id, document_user_quota=document_user_quota)


@app.task
def delete_collection_task(collection_id):
    collection = Collection.objects.get(id=collection_id)

    _, enable_knowledge_graph = get_collection_config_settings(collection)

    # Delete lightrag documents for this collection
    async def _async_delete_lightrag():
        # Create new LightRAG instance without using cache for Celery tasks
        rag_holder = await lightrag_holder.get_lightrag_holder(collection, use_cache=False)
        await rag_holder.adelete_by_collection(collection_id)

    # Execute the async deletion
    if enable_knowledge_graph:
        async_to_sync(_async_delete_lightrag)()

    # TODO remove the related collection in the vector db
    index_name = generate_fulltext_index_name(collection.id)
    delete_index(index_name)

    vector_db_conn = get_vector_db_connector(collection=generate_vector_db_collection_name(collection_id=collection_id))
    vector_db_conn.connector.delete_collection()

    qa_vector_db_conn = get_vector_db_connector(
        collection=generate_qa_vector_db_collection_name(collection=collection_id)
    )
    qa_vector_db_conn.connector.delete_collection()
