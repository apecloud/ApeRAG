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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import select

from aperag.config import settings
from aperag.context.full_text import create_index, delete_index
from aperag.db.models import Collection, CollectionStatus, ModelServiceProvider, ModelServiceProviderStatus
from aperag.embed.base_embedding import get_embedding_model
from aperag.graph import lightrag_holder
from aperag.schema.utils import parseCollectionConfig
from aperag.tasks.index import get_collection_config_settings
from aperag.utils.utils import (
    generate_fulltext_index_name,
    generate_qa_vector_db_collection_name,
    generate_vector_db_collection_name,
)
from config.celery import app
from config.vector_db import get_vector_db_connector

logger = logging.getLogger(__name__)


# Create a separate synchronous database engine for celery tasks
def get_sync_database_url():
    """Convert async database URL to sync version for celery"""
    url = settings.database_url
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://")
    elif url.startswith("sqlite+aiosqlite://"):
        return url.replace("sqlite+aiosqlite://", "sqlite:///")
    elif url.startswith("sqlite://"):
        return url  # sqlite:// is already sync
    else:
        return url


# Create sync engine and session factory for celery tasks
sync_engine = create_engine(get_sync_database_url())
SyncSessionLocal = sessionmaker(bind=sync_engine)


def get_sync_session():
    """Get a synchronous database session for celery tasks"""
    return SyncSessionLocal()


def get_collection_embedding_service_sync(collection):
    """Synchronous version of get_collection_embedding_service for celery tasks"""
    config = parseCollectionConfig(collection.config)
    embedding_msp = config.embedding.model_service_provider
    embedding_model_name = config.embedding.model
    custom_llm_provider = config.embedding.custom_llm_provider
    logging.info("get_collection_embedding_model %s %s", embedding_msp, embedding_model_name)

    # Query MSP using sync session
    with get_sync_session() as session:
        result = session.execute(
            select(ModelServiceProvider).where(
                ModelServiceProvider.user == collection.user,
                ModelServiceProvider.status != ModelServiceProviderStatus.DELETED,
            )
        )
        msps = result.scalars().all()
        msp_dict = {msp.name: msp for msp in msps}

    if embedding_msp in msp_dict:
        msp = msp_dict[embedding_msp]
        embedding_service_url = msp.base_url
        embedding_service_api_key = msp.api_key
        logging.info("get_collection_embedding_model %s %s", embedding_service_url, embedding_service_api_key)

        return get_embedding_model(
            embedding_provider=custom_llm_provider,
            embedding_model=embedding_model_name,
            embedding_service_url=embedding_service_url,
            embedding_service_api_key=embedding_service_api_key,
            embedding_max_chunks_in_batch=settings.embedding_max_chunks_in_batch,
        )

    logging.warning("get_collection_embedding_model cannot find model service provider %s", embedding_msp)
    return None, 0


@app.task
def init_collection_task(collection_id, document_user_quota):
    """Celery task for collection initialization"""
    with get_sync_session() as session:
        try:
            # Get collection from database
            collection = session.get(Collection, collection_id)

            if not collection or collection.status == CollectionStatus.DELETED:
                return

            # Get embedding service using sync version
            embedding_svc, vector_size = get_collection_embedding_service_sync(collection)

            if vector_size == 0:
                logger.error(f"Failed to get embedding service for collection {collection_id}")
                return

            vector_db_conn = get_vector_db_connector(
                collection=generate_vector_db_collection_name(collection_id=collection_id)
            )
            # pre-create collection in vector db
            vector_db_conn.connector.create_collection(vector_size=vector_size)

            qa_vector_db_conn = get_vector_db_connector(
                collection=generate_qa_vector_db_collection_name(collection=collection_id)
            )
            qa_vector_db_conn.connector.create_collection(vector_size=vector_size)

            index_name = generate_fulltext_index_name(collection_id)
            create_index(index_name)

            # Update collection status
            collection.status = CollectionStatus.ACTIVE
            session.add(collection)
            session.commit()

            logger.info(f"Successfully initialized collection {collection_id}")

        except Exception as e:
            logger.error(f"Failed to initialize collection {collection_id}: {e}")
            session.rollback()
            raise


@app.task
def delete_collection_task(collection_id):
    """Celery task for collection deletion"""
    with get_sync_session() as session:
        try:
            # Get collection from database
            collection = session.get(Collection, collection_id)

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

            vector_db_conn = get_vector_db_connector(
                collection=generate_vector_db_collection_name(collection_id=collection_id)
            )
            vector_db_conn.connector.delete_collection()

            qa_vector_db_conn = get_vector_db_connector(
                collection=generate_qa_vector_db_collection_name(collection=collection_id)
            )
            qa_vector_db_conn.connector.delete_collection()

            logger.info(f"Successfully deleted collection {collection_id}")

        except Exception as e:
            logger.error(f"Failed to delete collection {collection_id}: {e}")
            raise
