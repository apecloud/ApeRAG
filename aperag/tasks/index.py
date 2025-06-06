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
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from asgiref.sync import async_to_sync
from celery import Task
from sqlmodel import select

from aperag.config import SyncSessionDep, get_sync_session, settings, with_sync_session
from aperag.context.full_text import insert_document, remove_document
from aperag.db.models import (
    Collection,
    Document,
    DocumentIndexStatus,
    DocumentStatus,
)
from aperag.docparser.doc_parser import DocParser
from aperag.embed.base_embedding import get_collection_embedding_service_sync
from aperag.embed.local_path_embedding import LocalPathEmbedding
from aperag.graph import lightrag_holder
from aperag.objectstore.base import get_object_store
from aperag.schema.utils import parseCollectionConfig
from aperag.source.base import get_source
from aperag.source.feishu.client import FeishuNoPermission, FeishuPermissionDenied
from aperag.utils.tokenizer import get_default_tokenizer
from aperag.utils.uncompress import SUPPORTED_COMPRESSED_EXTENSIONS, uncompress
from aperag.utils.utils import (
    generate_fulltext_index_name,
    generate_vector_db_collection_name,
)
from config.celery import app
from config.vector_db import get_vector_db_connector

logger = logging.getLogger(__name__)


# Configuration constants
class IndexTaskConfig:
    MAX_EXTRACTED_SIZE = 5000 * 1024 * 1024  # 5 GB
    RETRY_COUNTDOWN_LIGHTRAG = 60
    RETRY_MAX_RETRIES_LIGHTRAG = 2
    RETRY_COUNTDOWN_ADD_INDEX = 5
    RETRY_MAX_RETRIES_ADD_INDEX = 1
    RETRY_COUNTDOWN_UPDATE_INDEX = 5
    RETRY_MAX_RETRIES_UPDATE_INDEX = 1


class CustomLoadDocumentTask(Task):
    @staticmethod
    @with_sync_session
    def _on_success(session: SyncSessionDep, document_id: str):
        document = session.get(Document, document_id)
        if not document:
            return
        # Update overall status
        document.update_overall_status()
        session.add(document)
        session.commit()
        logger.info(f"index for document {document.name} success")

    @staticmethod
    @with_sync_session
    def _on_failure(session: SyncSessionDep, document_id: str, exc: Exception):
        document = session.get(Document, document_id)
        if not document:
            return
        # Set all index statuses to failed
        document.vector_index_status = DocumentIndexStatus.FAILED
        document.fulltext_index_status = DocumentIndexStatus.FAILED
        document.graph_index_status = DocumentIndexStatus.FAILED
        document.update_overall_status()
        session.add(document)
        session.commit()
        logger.error(f"index for document {document.name} error:{exc}")

    def on_success(self, retval, task_id, args, kwargs):
        document_id = args[0]
        self._on_success(document_id)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        document_id = args[0]
        self._on_failure(document_id, exc)


class CustomDeleteDocumentTask(Task):
    @staticmethod
    @with_sync_session
    def _on_success(session: SyncSessionDep, document_id: str):
        document = session.get(Document, document_id)
        if not document:
            return
        logger.info(f"remove qdrant points for document {document.name} success")
        document.status = DocumentStatus.DELETED
        document.gmt_deleted = datetime.utcnow()
        document.name = document.name + "-" + str(uuid.uuid4())
        session.add(document)
        session.commit()

    @staticmethod
    @with_sync_session
    def _on_failure(session: SyncSessionDep, document_id: str, exc: Exception):
        document = session.get(Document, document_id)
        if not document:
            return
        logger.error(f"remove_index(): index delete from vector db failed:{exc}")
        document.status = DocumentStatus.FAILED
        session.add(document)
        session.commit()

    def on_success(self, retval, task_id, args, kwargs):
        document_id = args[0]
        self._on_success(document_id)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        document_id = args[0]
        self._on_failure(document_id, exc)


# Utility functions: Extract repeated code without changing main flow
def create_local_path_embedding_loader(local_doc, document_id, collection_id, user_id, embedding_model, vector_size):
    """Create LocalPathEmbedding instance - extracted from repeated code"""
    # Generate object store base path directly without using document object
    user_safe = user_id.replace("|", "-")
    object_store_base_path = f"user-{user_safe}/{collection_id}/{document_id}"

    return LocalPathEmbedding(
        filepath=local_doc.path,
        file_metadata=local_doc.metadata,
        object_store_base_path=object_store_base_path,
        embedding_model=embedding_model,
        vector_size=vector_size,
        vector_store_adaptor=get_vector_db_connector(
            collection=generate_vector_db_collection_name(collection_id=collection_id)
        ),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap_size,
        tokenizer=get_default_tokenizer(),
    )


def get_collection_config_settings(collection):
    """Extract collection configuration settings - extracted from repeated code"""
    config = parseCollectionConfig(collection.config)
    return config, config.enable_knowledge_graph or False


def uncompress_file(document: Document, supported_file_extensions: list[str]):
    obj_store = get_object_store()
    supported_file_extensions = supported_file_extensions or []

    with tempfile.TemporaryDirectory(prefix=f"aperag_unzip_{document.id}_") as temp_dir_path_str:
        tmp_dir = Path(temp_dir_path_str)
        obj = obj_store.get(document.object_path)
        if obj is None:
            raise Exception(f"object '{document.object_path}' is not found")
        suffix = Path(document.object_path).suffix
        with obj:
            uncompress(obj, suffix, tmp_dir)

        extracted_files = []
        total_size = 0
        for root, dirs, file_names in os.walk(tmp_dir):
            for name in file_names:
                path = Path(os.path.join(root, name))
                if path.suffix.lower() in SUPPORTED_COMPRESSED_EXTENSIONS:
                    continue
                if path.suffix.lower() not in supported_file_extensions:
                    continue
                extracted_files.append(path)
                total_size += path.stat().st_size

                if total_size > IndexTaskConfig.MAX_EXTRACTED_SIZE:
                    raise Exception("Extracted size exceeded limit")

        for extracted_file_path in extracted_files:
            with extracted_file_path.open(mode="rb") as extracted_file:  # open in binary
                document_instance = Document(
                    user=document.user,
                    name=document.name + "/" + extracted_file_path.name,
                    status=DocumentStatus.PENDING,
                    size=extracted_file_path.stat().st_size,
                    collection_id=document.collection_id,
                )
                # Upload to object store
                upload_path = f"{document_instance.object_store_base_path()}/original{suffix}"
                obj_store.put(upload_path, extracted_file)

                document_instance.object_path = upload_path
                document_instance.doc_metadata = json.dumps({"object_path": upload_path, "uncompressed": "true"})

                # Save document using sync session
                for session in get_sync_session():
                    session.add(document_instance)
                    session.commit()
                    session.refresh(document_instance)  # Refresh to get the generated ID

                add_index_for_local_document.delay(document_instance.id)

    return


@app.task(base=CustomLoadDocumentTask, bind=True, ignore_result=True)
def add_index_for_local_document(self, document_id):
    try:
        add_index_for_document(document_id)
    except Exception as e:
        raise self.retry(
            exc=e,
            countdown=IndexTaskConfig.RETRY_COUNTDOWN_ADD_INDEX,
            max_retries=IndexTaskConfig.RETRY_MAX_RETRIES_ADD_INDEX,
        )


@app.task(base=CustomLoadDocumentTask, bind=True, track_started=True)
def add_index_for_document(self, document_id):
    """
    Main task function for creating document indexes

    Handles the creation of vector index, fulltext index and knowledge graph index

    Args:
        document_id: ID of the Document model

    Raises:
        Exception: Various document processing exceptions (permissions, etc.)
    """
    document: Document = None
    collection: Collection = None
    # Get initial document and collection info
    for session in get_sync_session():
        document = session.get(Document, document_id)
        if not document:
            raise Exception(f"Document {document_id} not found")

        # Get collection synchronously
        collection = session.get(Collection, document.collection_id)
        if not collection:
            raise Exception(f"Collection {document.collection_id} not found")

    # Set all index statuses to running
    document.vector_index_status = DocumentIndexStatus.RUNNING
    document.fulltext_index_status = DocumentIndexStatus.RUNNING
    document.graph_index_status = DocumentIndexStatus.RUNNING
    document.status = DocumentStatus.RUNNING

    # Get document metadata while we have the session
    source = None
    local_doc = None
    metadata = json.loads(document.doc_metadata or "{}")
    metadata["doc_id"] = document_id
    supported_file_extensions = DocParser().supported_extensions()  # TODO: apply collection config
    supported_file_extensions += SUPPORTED_COMPRESSED_EXTENSIONS

    try:
        if document.object_path and Path(document.object_path).suffix in SUPPORTED_COMPRESSED_EXTENSIONS:
            config = parseCollectionConfig(collection.config)
            if config.source != "system":
                return
            # Need to get document again for uncompress_file since it might update it
            for session in get_sync_session():
                document = session.get(Document, document_id)
                uncompress_file(document, supported_file_extensions)
            return
        else:
            source = get_source(parseCollectionConfig(collection.config))
            local_doc = source.prepare_document(name=document.name, metadata=metadata)

            # Update document size if needed
            if document.size == 0:
                new_size = os.path.getsize(local_doc.path)
                document.size = new_size

            config, enable_knowledge_graph = get_collection_config_settings(collection)

            # Process vector index
            try:
                embedding_model, vector_size = get_collection_embedding_service_sync(collection)
                loader = create_local_path_embedding_loader(
                    local_doc, document_id, collection.id, document.user, embedding_model, vector_size
                )

                ctx_ids, content = loader.load_data()

                relate_ids = {
                    "ctx": ctx_ids,
                }
                document.relate_ids = json.dumps(relate_ids)
                document.vector_index_status = DocumentIndexStatus.COMPLETE
                logger.info(f"Vector index completed for document {local_doc.path}: {ctx_ids}")

            except Exception as e:
                document.vector_index_status = DocumentIndexStatus.FAILED
                logger.error(f"Vector index failed for document {local_doc.path}: {str(e)}")
                raise e

            # Process fulltext index
            try:
                if ctx_ids:  # Only create fulltext index when vector data exists
                    index = generate_fulltext_index_name(collection.id)
                    insert_document(index, document_id, local_doc.name, content)
                    document.fulltext_index_status = DocumentIndexStatus.COMPLETE
                    logger.info(f"Fulltext index completed for document {local_doc.path}")
                else:
                    document.fulltext_index_status = DocumentIndexStatus.SKIPPED
                    logger.info(f"Fulltext index skipped for document {local_doc.path} (no content)")
            except Exception as e:
                document.fulltext_index_status = DocumentIndexStatus.FAILED
                logger.error(f"Fulltext index failed for document {local_doc.path}: {str(e)}")

            # Process knowledge graph index
            try:
                if enable_knowledge_graph:
                    # Start asynchronous LightRAG indexing task
                    add_lightrag_index_task.delay(content, document_id, local_doc.path)
                    document.graph_index_status = DocumentIndexStatus.RUNNING
                    logger.info(f"Graph index task scheduled for document {local_doc.path}")
                else:
                    document.graph_index_status = DocumentIndexStatus.SKIPPED
                    logger.info(f"Graph index skipped for document {local_doc.path} (not enabled)")
            except Exception as e:
                document.graph_index_status = DocumentIndexStatus.FAILED
                logger.error(f"Graph index failed for document {local_doc.path}: {str(e)}")

    except FeishuNoPermission:
        raise Exception("no permission to access document %s" % document.name)
    except FeishuPermissionDenied:
        raise Exception("permission denied to access document %s" % document.name)
    except Exception as e:
        raise e
    finally:
        # Update overall status
        document.update_overall_status()
        for session in get_sync_session():
            session.add(document)
            session.commit()
        if local_doc and source:
            source.cleanup_document(local_doc.path)


@app.task(base=CustomDeleteDocumentTask, bind=True, track_started=True)
def remove_index(self, document_id):
    """
    Remove the document embedding index from vector store database

    Args:
        document_id: ID of the Document model

    Raises:
        Exception: Various database operation exceptions
    """
    for session in get_sync_session():
        document = session.get(Document, document_id)
        if not document:
            raise Exception(f"Document {document_id} not found")

        # Get collection synchronously
        collection = session.get(Collection, document.collection_id)
        if not collection:
            raise Exception(f"Collection {document.collection_id} not found")

        try:
            index = generate_fulltext_index_name(collection.id)
            remove_document(index, document.id)

            if document.relate_ids == "":
                return

            relate_ids = json.loads(document.relate_ids)
            vector_db = get_vector_db_connector(
                collection=generate_vector_db_collection_name(collection_id=collection.id)
            )
            ctx_relate_ids = relate_ids.get("ctx", [])
            vector_db.connector.delete(ids=ctx_relate_ids)
            logger.info(f"remove ctx qdrant points: {ctx_relate_ids} for document {document.name}")

            # Only call LightRAG deletion task if knowledge graph is enabled
            if collection.config:
                config = parseCollectionConfig(collection.config)
                enable_knowledge_graph = config.enable_knowledge_graph or False
                if enable_knowledge_graph:
                    remove_lightrag_index_task.delay(document_id, collection.id)

        except Exception as e:
            raise e


@app.task(base=CustomLoadDocumentTask, bind=True, track_started=True)
def update_index_for_local_document(self, document_id):
    try:
        update_index_for_document(document_id)
    except Exception as e:
        raise self.retry(
            exc=e,
            countdown=IndexTaskConfig.RETRY_COUNTDOWN_UPDATE_INDEX,
            max_retries=IndexTaskConfig.RETRY_MAX_RETRIES_UPDATE_INDEX,
        )


@app.task(base=CustomLoadDocumentTask, bind=True, track_started=True)
def update_index_for_document(self, document_id):
    """
    Task function for updating document indexes

    Deletes old index data and creates new indexes

    Args:
        document_id: ID of the Document model

    Raises:
        Exception: Various document processing exceptions (permissions, etc.)
    """
    for session in get_sync_session():
        document = session.get(Document, document_id)
        if not document:
            raise Exception(f"Document {document_id} not found")

        # Get collection synchronously
        collection = session.get(Collection, document.collection_id)
        if not collection:
            raise Exception(f"Collection {document.collection_id} not found")

        # Get document info while we have the session
        doc_user = document.user
        collection_id = collection.id
        doc_name = document.name
        doc_metadata = document.doc_metadata or "{}"
        relate_ids_str = document.relate_ids

        document.status = DocumentStatus.RUNNING
        session.add(document)
        session.commit()

        try:
            relate_ids = json.loads(relate_ids_str) if relate_ids_str and relate_ids_str.strip() else {}
            source = get_source(parseCollectionConfig(collection.config))
            metadata = json.loads(doc_metadata)
            metadata["doc_id"] = document_id
            local_doc = source.prepare_document(name=doc_name, metadata=metadata)

            embedding_model, vector_size = get_collection_embedding_service_sync(collection)
            loader = create_local_path_embedding_loader(
                local_doc, document_id, collection_id, doc_user, embedding_model, vector_size
            )
            loader.connector.delete(ids=relate_ids.get("ctx", []))

            config, enable_knowledge_graph = get_collection_config_settings(collection)
            ctx_ids, content = loader.load_data()
            logger.info(f"add ctx qdrant points: {ctx_ids} for document {local_doc.path}")

            # only index the document that have points in the vector database
            if ctx_ids:
                index = generate_fulltext_index_name(collection_id)
                insert_document(index, document_id, local_doc.name, content)

            relate_ids = {
                "ctx": ctx_ids,
            }
            # Update relate_ids in a new session
            with get_sync_session() as update_session:
                document = update_session.get(Document, document_id)
                document.relate_ids = json.dumps(relate_ids)
                update_session.add(document)
                update_session.commit()
            logger.info(f"update qdrant points: {json.dumps(relate_ids)} for document {local_doc.path}")

            if enable_knowledge_graph:
                add_lightrag_index_task.delay(content, document_id, local_doc.path)

        except FeishuNoPermission:
            raise Exception("no permission to access document %s" % doc_name)
        except FeishuPermissionDenied:
            raise Exception("permission denied to access document %s" % doc_name)
        except Exception as e:
            logger.error(e)
            raise Exception("an error occur %s" % e)
        finally:
            # Final status update in a new session
            with get_sync_session() as final_session:
                document = final_session.get(Document, document_id)
                if document:
                    final_session.add(document)
                    final_session.commit()

        source.cleanup_document(local_doc.path)


@app.task(bind=True, track_started=True)
def add_lightrag_index_task(self, content, document_id, file_path):
    """
    Dedicated Celery task for LightRAG indexing
    Create new LightRAG instance each time to avoid event loop conflicts in this task
    """
    logger.info(f"Begin LightRAG indexing task for document (ID: {document_id})")

    # Get document object and check if it's deleted
    for session in get_sync_session():
        document = session.get(Document, document_id)
        if not document:
            logger.info(f"Document {document_id} not found, skipping LightRAG indexing")
            return

        if document.status == DocumentStatus.DELETED:
            logger.info(f"Document {document_id} is deleted, skipping LightRAG indexing")
            return

        # Check if collection is deleted
        try:
            collection = session.get(Collection, document.collection_id)
            if not collection:
                logger.info(
                    f"Collection {document.collection_id} not found for document {document_id}, skipping LightRAG indexing"
                )
                return

            if collection.status == Collection.Status.DELETED:
                logger.info(
                    f"Collection {collection.id} is deleted, skipping LightRAG indexing for document {document_id}"
                )
                document.graph_index_status = DocumentStatus.SKIPPED
                session.add(document)
                session.commit()
                return
        except Exception:
            logger.info(f"Collection not found for document {document_id}, skipping LightRAG indexing")
            return

        document.graph_index_status = DocumentStatus.RUNNING
        session.add(document)
        session.commit()

    async def _async_add_lightrag_index():
        from aperag.config import get_async_session

        async for async_session in get_async_session():
            # Get document and collection using async session
            document_stmt = select(Document).where(Document.id == document_id)
            document_result = await async_session.execute(document_stmt)
            document = document_result.scalars().first()

            if not document:
                raise Exception(f"Document {document_id} not found")

            collection_stmt = select(Collection).where(Collection.id == document.collection_id)
            collection_result = await async_session.execute(collection_stmt)
            collection = collection_result.scalars().first()

            if not collection:
                raise Exception(f"Collection {document.collection_id} not found")

            # Create new LightRAG instance without using cache for Celery tasks
            rag_holder = await lightrag_holder.get_lightrag_holder(collection, use_cache=False)

            await rag_holder.ainsert(input=content, ids=document_id, file_paths=file_path)

            lightrag_docs = await rag_holder.get_processed_docs()
            if not lightrag_docs or str(document_id) not in lightrag_docs:
                error_msg = f"Error indexing document for LightRAG (ID: {document_id}). No processed document found."
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info(f"Successfully completed LightRAG indexing for document (ID: {document_id})")
            break

    try:
        async_to_sync(_async_add_lightrag_index)()
        # Update graph index status to complete
        for session in get_sync_session():
            document = session.get(Document, document_id)
            if document:
                document.graph_index_status = DocumentStatus.COMPLETE
                document.update_overall_status()
                session.add(document)
                session.commit()
        logger.info(f"Graph index completed for document (ID: {document_id})")
    except Exception as e:
        logger.error(f"LightRAG indexing failed for document (ID: {document_id}): {str(e)}")
        # Update graph index status to failed
        for session in get_sync_session():
            document = session.get(Document, document_id)
            if document:
                document.graph_index_status = DocumentStatus.FAILED
                document.update_overall_status()
                session.add(document)
                session.commit()
        raise self.retry(
            exc=e,
            countdown=IndexTaskConfig.RETRY_COUNTDOWN_LIGHTRAG,
            max_retries=IndexTaskConfig.RETRY_MAX_RETRIES_LIGHTRAG,
        )


@app.task(bind=True, track_started=True)
def remove_lightrag_index_task(self, document_id, collection_id):
    """
    Dedicated Celery task for LightRAG deletion
    Create new LightRAG instance without using cache for Celery tasks
    """
    logger.info(f"Begin LightRAG deletion task for document (ID: {document_id})")

    async def _async_delete_lightrag():
        from aperag.config import get_async_session

        async for async_session in get_async_session():
            collection_stmt = select(Collection).where(Collection.id == collection_id)
            collection_result = await async_session.execute(collection_stmt)
            collection = collection_result.scalars().first()

            if not collection:
                raise Exception(f"Collection {collection_id} not found")

            # Create new LightRAG instance without using cache for Celery tasks
            rag_holder = await lightrag_holder.get_lightrag_holder(collection, use_cache=False)
            await rag_holder.adelete_by_doc_id(document_id)
            logger.info(f"Successfully completed LightRAG deletion for document (ID: {document_id})")
            break

    try:
        async_to_sync(_async_delete_lightrag)()
    except Exception as e:
        logger.error(f"LightRAG deletion failed for document (ID: {document_id}): {str(e)}")
        raise self.retry(
            exc=e,
            countdown=IndexTaskConfig.RETRY_COUNTDOWN_LIGHTRAG,
            max_retries=IndexTaskConfig.RETRY_MAX_RETRIES_LIGHTRAG,
        )
