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
from typing import Any, Dict, List

from aperag.config import get_vector_db_connector
from aperag.db.models import CollectionStatus, Document, DocumentStatus
from datetime import timedelta
from typing import Any

from asgiref.sync import Dict, async_to_sync
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from aperag.config import get_vector_db_connector
from aperag.db import models as db_models
from aperag.db.models import CollectionStatus
from aperag.db.ops import db_ops
from aperag.graph import lightrag_manager
from aperag.index.fulltext_index import create_index, delete_index
from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
from aperag.objectstore.base import get_object_store
from aperag.schema.utils import parseCollectionConfig
from aperag.service.document_service_sync import document_service_sync, SyncUploadFile
from aperag.source.base import get_source
from aperag.tasks.models import TaskResult
from asgiref.sync import async_to_sync
from aperag.utils.utils import (
    generate_fulltext_index_name,
    generate_vector_db_collection_name,
    utc_now,
)

logger = logging.getLogger(__name__)


class CollectionTask:
    """Collection workflow orchestrator"""

    def initialize_collection(self, collection_id: str, document_user_quota: int) -> TaskResult:
        """
        Initialize a new collection with all required components

        Args:
            collection_id: Collection ID to initialize
            document_user_quota: User quota for documents

        Returns:
            TaskResult: Result of the initialization
        """
        try:
            # Get collection from database
            collection = db_ops.query_collection_by_id(collection_id)

            if not collection or collection.status == CollectionStatus.DELETED:
                return TaskResult(success=False, error=f"Collection {collection_id} not found or deleted")

            # Initialize vector database connections
            self._initialize_vector_databases(collection_id, collection)

            # Initialize fulltext index
            self._initialize_fulltext_index(collection_id)

            # Update collection status
            collection.status = CollectionStatus.ACTIVE
            db_ops.update_collection(collection)

            logger.info(f"Successfully initialized collection {collection_id}")

            return TaskResult(
                success=True,
                data={"collection_id": collection_id, "status": "initialized"},
                metadata={"document_user_quota": document_user_quota},
            )

        except Exception as e:
            logger.error(f"Failed to initialize collection {collection_id}: {str(e)}")
            return TaskResult(success=False, error=f"Collection initialization failed: {str(e)}")

    def delete_collection(self, collection_id: str) -> TaskResult:
        """
        Delete a collection and all its associated data

        Args:
            collection_id: Collection ID to delete

        Returns:
            TaskResult: Result of the deletion
        """
        try:
            # Get collection from database
            collection = db_ops.query_collection_by_id(collection_id, ignore_deleted=False)

            if not collection:
                return TaskResult(success=False, error=f"Collection {collection_id} not found")

            # Delete knowledge graph data if enabled
            deletion_stats = self._delete_knowledge_graph_data(collection)

            # Delete vector databases
            self._delete_vector_databases(collection_id)

            # Delete fulltext index
            self._delete_fulltext_index(collection_id)

            logger.info(f"Successfully deleted collection {collection_id}")

            return TaskResult(
                success=True, data={"collection_id": collection_id, "status": "deleted"}, metadata=deletion_stats
            )

        except Exception as e:
            logger.error(f"Failed to delete collection {collection_id}: {str(e)}")
            return TaskResult(success=False, error=f"Collection deletion failed: {str(e)}")

    def _initialize_vector_databases(self, collection_id: str, collection) -> None:
        """Initialize vector database collections"""
        # Get embedding service
        _, vector_size = get_collection_embedding_service_sync(collection)

        # Create main vector database collection
        vector_db_conn = get_vector_db_connector(
            collection=generate_vector_db_collection_name(collection_id=collection_id)
        )
        vector_db_conn.connector.create_collection(vector_size=vector_size)

        logger.debug(f"Initialized vector databases for collection {collection_id}")

    def _initialize_fulltext_index(self, collection_id: str) -> None:
        """Initialize fulltext search index"""
        index_name = generate_fulltext_index_name(collection_id)
        create_index(index_name)
        logger.debug(f"Initialized fulltext index {index_name}")

    def _delete_knowledge_graph_data(self, collection) -> Dict[str, Any]:
        """Delete knowledge graph data for the collection"""
        config = parseCollectionConfig(collection.config)
        enable_knowledge_graph = config.enable_knowledge_graph or False

        deletion_stats = {"knowledge_graph_enabled": enable_knowledge_graph}

        if not enable_knowledge_graph:
            return deletion_stats

        async def _delete_lightrag():
            # Create new LightRAG instance
            rag = await lightrag_manager.create_lightrag_instance(collection)

            # Get all document IDs in this collection
            documents = db_ops.query_documents([collection.user], collection.id)
            document_ids = [doc.id for doc in documents]

            if document_ids:
                deleted_count = 0
                failed_count = 0

                for document_id in document_ids:
                    try:
                        await rag.adelete_by_doc_id(str(document_id))
                        deleted_count += 1
                        logger.debug(f"Deleted lightrag document for document ID: {document_id}")
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Failed to delete lightrag document for document ID {document_id}: {str(e)}")

                logger.info(
                    f"Completed lightrag document deletion for collection {collection.id}: "
                    f"{deleted_count} deleted, {failed_count} failed"
                )

                deletion_stats.update({"documents_deleted": deleted_count, "documents_failed": failed_count})
            else:
                logger.info(f"No documents found for collection {collection.id}")
                deletion_stats["documents_deleted"] = 0

            # Clean up resources
            await rag.finalize_storages()

        # Execute async deletion
        async_to_sync(_delete_lightrag)()

        return deletion_stats

    def _delete_vector_databases(self, collection_id: str) -> None:
        """Delete vector database collections"""
        # Delete main vector database collection
        vector_db_conn = get_vector_db_connector(
            collection=generate_vector_db_collection_name(collection_id=collection_id)
        )
        vector_db_conn.connector.delete_collection()

        logger.debug(f"Deleted vector database collections for collection {collection_id}")

    def _delete_fulltext_index(self, collection_id: str) -> None:
        """Delete fulltext search index"""
        index_name = generate_fulltext_index_name(collection_id)
        delete_index(index_name)
        logger.debug(f"Deleted fulltext index {index_name}")

    def cleanup_expired_documents(self, collection_id: str):
        """
        Clean up documents that have been in UPLOADED status for more than 1 day.
        This function runs asynchronously and handles all database operations.
        Uses soft delete by marking documents as EXPIRED instead of deleting them.
        """
        logger.info("Starting cleanup of expired uploaded documents")

        def _cleanup_expired_documents(session: Session):
            # Calculate expiration time (1 day ago)
            current_time = utc_now()
            expiration_threshold = current_time - timedelta(days=1)

            # Query for expired documents
            stmt = select(db_models.Document).where(
                and_(
                    db_models.Document.collection_id == collection_id,
                    db_models.Document.status == db_models.DocumentStatus.UPLOADED,
                    db_models.Document.gmt_created < expiration_threshold,
                )
            )

            result = session.execute(stmt)
            expired_documents = result.scalars().all()

            if not expired_documents:
                logger.info("No expired documents found")
                return {"total_found": 0, "expired_count": 0, "failed_count": 0}

            logger.info(f"Found {len(expired_documents)} expired documents to clean up")

            expired_count = 0
            failed_count = 0
            obj_store = get_object_store()

            for document in expired_documents:
                try:
                    # Delete from object store
                    try:
                        obj_store.delete_objects_by_prefix(document.object_store_base_path())
                        logger.info(
                            f"Deleted objects from object store for expired document {document.id}: {document.object_store_base_path()}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete objects for expired document {document.id} from object store: {e}"
                        )

                    # Soft delete: Mark document as EXPIRED instead of deleting
                    document.status = db_models.DocumentStatus.EXPIRED
                    document.gmt_updated = current_time
                    session.add(document)
                    expired_count += 1
                    logger.info(
                        f"Marked document {document.id} as expired (name: {document.name}, created: {document.gmt_created})"
                    )

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to cleanup expired document {document.id}: {e}")

            session.commit()

            return {"expired_count": expired_count, "failed_count": failed_count, "total_found": len(expired_documents)}

        try:
            # Execute the cleanup with transaction
            result = db_ops._execute_transaction(_cleanup_expired_documents)

            logger.info(
                f"Cleanup completed - Expired: {result.get('expired_count', 0)}, "
                f"Failed: {result['failed_count']}, Total found: {result['total_found']}"
            )

            return result

        except Exception as e:
            logger.error(f"Error during expired documents cleanup: {e}", exc_info=True)
            return {"expired_count": 0, "failed_count": 0, "error": str(e)}


    def sync_object_storage_collection(self, collection_id: str, trigger_type: str = "manual") -> TaskResult:
        """
        Synchronize object storage collection with documents
        
        Args:
            collection_id: Collection ID to sync
            trigger_type: Type of trigger ('manual', 'scheduled', 'collection_update')
            
        Returns:
            TaskResult: Result of the synchronization
        """
        try:
            # Wait for collection to be initialized if needed
            collection = self._wait_for_collection_initialization(collection_id)
            if not collection:
                return TaskResult(success=False, error=f"Collection {collection_id} not found or deleted")

            # Parse collection config to check if it's object storage or anybase type
            config = parseCollectionConfig(collection.config)
            is_object_storage = hasattr(config, 'object_storage') and config.object_storage
            is_anybase = hasattr(config, 'anybase') and config.anybase
            
            if not (is_object_storage or is_anybase):
                return TaskResult(success=False, error=f"Collection {collection_id} is not an object storage or anybase collection")

            # Perform the actual sync
            sync_result = self._perform_object_storage_sync(collection)

            logger.info(f"Successfully completed sync for collection {collection_id}")
            
            return TaskResult(
                success=sync_result["success"],
                data={
                    "collection_id": collection_id,
                    "stats": sync_result
                },
                error=sync_result.get("error")
            )

        except Exception as e:
            logger.error(f"Failed to sync collection {collection_id}: {str(e)}")
            return TaskResult(success=False, error=f"Collection sync failed: {str(e)}")

    def _perform_object_storage_sync(self, collection) -> Dict[str, Any]:
        """
        Perform the actual object storage synchronization using Source interface and document_service
        
        Args:
            collection: Collection object
            
        Returns:
            Dict containing sync results
        """
        try:
            config = parseCollectionConfig(collection.config)
            
            # Get source connector
            source = get_source(config)
            
            # Scan documents from object storage
            remote_documents = list(source.scan_documents())
            total_objects = len(remote_documents)
            
            # Get existing documents in collection
            existing_docs = db_ops.query_documents([collection.user], collection.id)
            existing_doc_map = {doc.name: doc for doc in existing_docs if doc.status != DocumentStatus.DELETED}
            
            # Track sync statistics
            stats = {
                "success": True,
                "total_objects": total_objects,
                "new_documents": 0,
                "updated_documents": 0,
                "deleted_documents": 0,
                "failed_documents": 0,
                "error_details": []
            }
            
            # Prepare documents to be created/updated
            documents_to_create = []
            documents_to_update = []
            
            for remote_doc in remote_documents:
                try:
                    if remote_doc.name in existing_doc_map:
                        # Check if document needs update
                        existing_doc = existing_doc_map[remote_doc.name]
                        if self._should_update_document(existing_doc, remote_doc):
                            documents_to_update.append(remote_doc)
                    else:
                        # New document to be created
                        documents_to_create.append(remote_doc)
                        
                except Exception as obj_error:
                    stats["failed_documents"] += 1
                    error_detail = {
                        "object": remote_doc.name,
                        "error": str(obj_error)
                    }
                    stats["error_details"].append(error_detail)
                    logger.warning(f"Failed to process object {remote_doc.name}: {str(obj_error)}")
            
            # Process new documents in batches
            if documents_to_create:
                stats["new_documents"] = self._create_documents_from_source(
                    source, collection, documents_to_create, stats
                )
            
            # Process updated documents
            if documents_to_update:
                stats["updated_documents"] = self._update_documents_from_source(
                    source, collection, documents_to_update, existing_doc_map, stats
                )
            
            # Handle deleted objects (objects that exist in DB but not in storage)
            # storage_object_names = {doc.name for doc in remote_documents}
            # documents_to_delete = []
            # for doc_name, doc in existing_doc_map.items():
            #     if doc_name not in storage_object_names:
            #         documents_to_delete.append(doc.id)
            
            # if documents_to_delete:
            #     stats["deleted_documents"] = self._delete_documents(
            #         collection, documents_to_delete, stats
            #     )
            
            # Close source connection
            source.close()
            
            logger.info(f"Sync completed for collection {collection.id}: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Object storage sync failed for collection {collection.id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_objects": 0,
                "new_documents": 0,
                "updated_documents": 0,
                "deleted_documents": 0,
                "failed_documents": 0
            }

    def _should_update_document(self, document: Document, remote_doc) -> bool:
        """
        Check if a document should be updated based on remote document metadata
        
        Args:
            document: Existing document
            remote_doc: RemoteDocument from storage
            
        Returns:
            bool: True if document should be updated
        """
        # Check size difference
        if document.size != remote_doc.size:
            return True
            
        # Check modification time if available
        storage_modified = remote_doc.metadata.get("modified_time")
        if storage_modified and document.gmt_updated:
            # Convert to comparable format if needed
            if storage_modified > document.gmt_updated:
                return True
                
        return False

    def _create_documents_from_source(self, source, collection, remote_documents, stats) -> int:
        """
        Create documents from remote documents using document_service
        
        Args:
            source: Source connector
            collection: Collection object
            remote_documents: List of RemoteDocument objects
            stats: Statistics dictionary to update
            
        Returns:
            int: Number of documents created successfully
        """
        created_count = 0
        batch_size = 10  # Process in batches to avoid memory issues
        
        for i in range(0, len(remote_documents), batch_size):
            batch = remote_documents[i:i + batch_size]
            upload_files = []
            local_documents = []
            
            try:
                # Prepare documents from source
                for remote_doc in batch:
                    try:
                        local_doc = source.prepare_document(remote_doc.name, remote_doc.metadata)
                        local_documents.append(local_doc)
                        
                        # Read file content and create UploadFile
                        with open(local_doc.path, 'rb') as f:
                            content = f.read()
                        
                        upload_file = SyncUploadFile(
                            filename=remote_doc.name,
                            content=content,
                            size=len(content)
                        )
                        upload_files.append(upload_file)
                        
                    except Exception as e:
                        stats["failed_documents"] += 1
                        error_detail = {
                            "object": remote_doc.name,
                            "error": f"Failed to prepare document: {str(e)}"
                        }
                        stats["error_details"].append(error_detail)
                        logger.warning(f"Failed to prepare document {remote_doc.name}: {str(e)}")
                
                # Create documents using document_service
                if upload_files:
                    try:
                        result = document_service_sync.create_documents(
                            collection.user, collection.id, upload_files
                        )
                        created_count += len(result.items)
                        logger.info(f"Created {len(result.items)} documents in batch")
                        
                    except Exception as e:
                        stats["failed_documents"] += len(upload_files)
                        for upload_file in upload_files:
                            error_detail = {
                                "object": upload_file.filename,
                                "error": f"Failed to create document: {str(e)}"
                            }
                            stats["error_details"].append(error_detail)
                        logger.error(f"Failed to create documents batch: {str(e)}")
                
            finally:
                # Clean up local files
                for local_doc in local_documents:
                    try:
                        source.cleanup_document(local_doc.path)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup local file {local_doc.path}: {str(e)}")
        
        return created_count

    def _update_documents_from_source(self, source, collection, remote_documents, existing_doc_map, stats) -> int:
        """
        Update documents from remote documents using document_service
        
        Args:
            source: Source connector
            collection: Collection object
            remote_documents: List of RemoteDocument objects to update
            existing_doc_map: Map of existing documents
            stats: Statistics dictionary to update
            
        Returns:
            int: Number of documents updated successfully
        """
        updated_count = 0
        
        for remote_doc in remote_documents:
            try:
                existing_doc = existing_doc_map[remote_doc.name]
                
                # Delete existing document
                document_service_sync.delete_documents(
                    collection.user, collection.id, [existing_doc.id]
                )
                
                # Prepare and create new document
                local_doc = source.prepare_document(remote_doc.name, remote_doc.metadata)
                
                try:
                    # Read file content and create UploadFile
                    with open(local_doc.path, 'rb') as f:
                        content = f.read()
                    
                    upload_file = SyncUploadFile(
                        filename=remote_doc.name,
                        content=content,
                        size=len(content)
                    )
                    
                    # Create new document
                    result = document_service_sync.create_documents(
                        collection.user, collection.id, [upload_file]
                    )
                    
                    if result.items:
                        updated_count += 1
                        logger.debug(f"Updated document: {remote_doc.name}")
                    
                finally:
                    # Clean up local file
                    source.cleanup_document(local_doc.path)
                    
            except Exception as e:
                stats["failed_documents"] += 1
                error_detail = {
                    "object": remote_doc.name,
                    "error": f"Failed to update document: {str(e)}"
                }
                stats["error_details"].append(error_detail)
                logger.warning(f"Failed to update document {remote_doc.name}: {str(e)}")
        
        return updated_count

    def _delete_documents(self, collection, document_ids, stats) -> int:
        """
        Delete documents using document_service
        
        Args:
            collection: Collection object
            document_ids: List of document IDs to delete
            stats: Statistics dictionary to update
            
        Returns:
            int: Number of documents deleted successfully
        """
        deleted_count = 0
        
        try:
            # Delete documents in batch
            result = document_service_sync.delete_documents(
                collection.user, collection.id, document_ids
            )
            
            if result.get("status") == "success":
                deleted_count = len(result.get("deleted_ids", []))
                logger.info(f"Deleted {deleted_count} documents")
            
        except Exception as e:
            stats["failed_documents"] += len(document_ids)
            for doc_id in document_ids:
                error_detail = {
                    "object": doc_id,
                    "error": f"Failed to delete document: {str(e)}"
                }
                stats["error_details"].append(error_detail)
            logger.error(f"Failed to delete documents: {str(e)}")
        
        return deleted_count

    def _wait_for_collection_initialization(self, collection_id: str, max_wait_seconds: int = 300, check_interval: int = 5):
        """
        Wait for collection to be initialized (status becomes ACTIVE)
        
        Args:
            collection_id: Collection ID to wait for
            max_wait_seconds: Maximum time to wait in seconds (default: 5 minutes)
            check_interval: Check interval in seconds (default: 5 seconds)
            
        Returns:
            Collection object if initialized successfully, None if failed or timeout
        """
        import time
        
        start_time = time.time()
        logger.info(f"Waiting for collection {collection_id} to be initialized...")
        
        while time.time() - start_time < max_wait_seconds:
            # Get collection from database
            collection = db_ops.query_collection_by_id(collection_id)
            
            if not collection:
                logger.error(f"Collection {collection_id} not found")
                return None
                
            if collection.status == CollectionStatus.DELETED:
                logger.error(f"Collection {collection_id} has been deleted")
                return None
                
            if collection.status == CollectionStatus.ACTIVE:
                logger.info(f"Collection {collection_id} is now active and ready for sync")
                return collection
                
            # Log current status and wait
            logger.debug(f"Collection {collection_id} status: {collection.status}, waiting...")
            time.sleep(check_interval)
        
        # Timeout reached
        logger.warning(f"Timeout waiting for collection {collection_id} to be initialized after {max_wait_seconds} seconds")
        
        # Return the collection anyway, let the caller decide what to do
        collection = db_ops.query_collection_by_id(collection_id)
        if collection and collection.status != CollectionStatus.DELETED:
            logger.warning(f"Proceeding with collection {collection_id} in status {collection.status}")
            return collection
            
        return None


collection_task = CollectionTask()
