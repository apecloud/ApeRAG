"""
Index Tasks - Actual index operations

Simple Celery tasks that perform the actual index creation/update/deletion operations.
These tasks call back to IndexTaskCallbacks when complete.
"""

import logging
import asyncio
from celery import current_app

from aperag.db.models import DocumentIndexType
from aperag.index_reconciler import index_task_callbacks

logger = logging.getLogger(__name__)


@current_app.task(bind=True)
def create_index_task(self, document_id: str, index_type: str):
    """Create an index for a document"""
    try:
        logger.info(f"Creating {index_type} index for document {document_id}")
        
        # TODO: Implement actual index creation logic
        # This would call the appropriate indexer based on index_type
        index_data = None
        
        if index_type == DocumentIndexType.VECTOR.value:
            index_data = _create_vector_index(document_id)
        elif index_type == DocumentIndexType.FULLTEXT.value:
            index_data = _create_fulltext_index(document_id)
        elif index_type == DocumentIndexType.GRAPH.value:
            index_data = _create_graph_index(document_id)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Notify completion
        asyncio.run(index_task_callbacks.on_index_created(document_id, index_type, index_data))
        
        logger.info(f"Successfully created {index_type} index for document {document_id}")
        
    except Exception as e:
        error_msg = f"Failed to create {index_type} index for document {document_id}: {str(e)}"
        logger.error(error_msg)
        
        # Notify failure
        asyncio.run(index_task_callbacks.on_index_failed(document_id, index_type, error_msg))
        
        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60, max_retries=3)


@current_app.task(bind=True)
def update_index_task(self, document_id: str, index_type: str):
    """Update an existing index for a document"""
    try:
        logger.info(f"Updating {index_type} index for document {document_id}")
        
        # TODO: Implement actual index update logic
        # For now, treat update same as create
        index_data = None
        
        if index_type == DocumentIndexType.VECTOR.value:
            index_data = _update_vector_index(document_id)
        elif index_type == DocumentIndexType.FULLTEXT.value:
            index_data = _update_fulltext_index(document_id)
        elif index_type == DocumentIndexType.GRAPH.value:
            index_data = _update_graph_index(document_id)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Notify completion
        asyncio.run(index_task_callbacks.on_index_created(document_id, index_type, index_data))
        
        logger.info(f"Successfully updated {index_type} index for document {document_id}")
        
    except Exception as e:
        error_msg = f"Failed to update {index_type} index for document {document_id}: {str(e)}"
        logger.error(error_msg)
        
        # Notify failure
        asyncio.run(index_task_callbacks.on_index_failed(document_id, index_type, error_msg))
        
        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60, max_retries=3)


@current_app.task(bind=True)
def delete_index_task(self, document_id: str, index_type: str, index_data: str = None):
    """Delete an index for a document"""
    try:
        logger.info(f"Deleting {index_type} index for document {document_id}")
        
        # TODO: Implement actual index deletion logic
        if index_type == DocumentIndexType.VECTOR.value:
            _delete_vector_index(document_id, index_data)
        elif index_type == DocumentIndexType.FULLTEXT.value:
            _delete_fulltext_index(document_id, index_data)
        elif index_type == DocumentIndexType.GRAPH.value:
            _delete_graph_index(document_id, index_data)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Notify completion
        asyncio.run(index_task_callbacks.on_index_deleted(document_id, index_type))
        
        logger.info(f"Successfully deleted {index_type} index for document {document_id}")
        
    except Exception as e:
        error_msg = f"Failed to delete {index_type} index for document {document_id}: {str(e)}"
        logger.error(error_msg)
        
        # Notify failure
        asyncio.run(index_task_callbacks.on_index_failed(document_id, index_type, error_msg))
        
        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60, max_retries=3)


@current_app.task
def reconcile_indexes_task():
    """Periodic task to reconcile index specs with statuses"""
    try:
        logger.info("Starting index reconciliation")
        
        # Import here to avoid circular dependencies
        from aperag.index_reconciler import index_reconciler
        
        # Run reconciliation
        asyncio.run(index_reconciler.reconcile_all())
        
        logger.info("Index reconciliation completed")
        
    except Exception as e:
        logger.error(f"Index reconciliation failed: {e}")
        raise


# ============ Index Operation Implementations ============

def _get_document_and_collection(document_id: str):
    """Helper function to get document and collection objects"""
    from aperag.db.ops import db_ops
    
    document = db_ops.query_document_by_id(int(document_id))
    if not document:
        raise ValueError(f"Document {document_id} not found")
    
    collection = db_ops.query_collection_by_id(document.collection_id)
    if not collection:
        raise ValueError(f"Collection {document.collection_id} not found")
    
    return document, collection


def _prepare_document_content(document, collection):
    """Helper function to prepare document content for indexing"""
    from aperag.core.processors.document_parser import document_parser
    from aperag.source.base import get_source
    from aperag.schema.utils import parseCollectionConfig
    import json
    
    # Get document source and prepare local file
    source = get_source(parseCollectionConfig(collection.config))
    metadata = json.loads(document.doc_metadata or "{}")
    metadata["doc_id"] = document.id
    local_doc = source.prepare_document(name=document.name, metadata=metadata)
    
    try:
        # Parse document to get content and parts
        parsing_result = document_parser.process_document_parsing(
            local_doc.path,
            local_doc.metadata,
            document.object_store_base_path()
        )
        
        return parsing_result.content, parsing_result.doc_parts, local_doc
    except Exception as e:
        # Cleanup on error
        source.cleanup_document(local_doc.path)
        raise e


def _create_vector_index(document_id: str) -> str:
    """Create vector index for document"""
    from aperag.index.vector.indexer import vector_indexer
    
    try:
        document, collection = _get_document_and_collection(document_id)
        content, doc_parts, local_doc = _prepare_document_content(document, collection)
        
        try:
            # Create vector index
            result = vector_indexer.create_index(
                document_id=int(document_id),
                content=content,
                doc_parts=doc_parts,
                collection=collection,
                file_path=local_doc.path
            )
            
            if not result.success:
                raise Exception(result.error)
            
            # Return index data as JSON string
            import json
            return json.dumps(result.data) if result.data else None
            
        finally:
            # Cleanup local document
            from aperag.source.base import get_source
            from aperag.schema.utils import parseCollectionConfig
            source = get_source(parseCollectionConfig(collection.config))
            source.cleanup_document(local_doc.path)
    
    except Exception as e:
        logger.error(f"Vector index creation failed for document {document_id}: {str(e)}")
        raise


def _create_fulltext_index(document_id: str) -> str:
    """Create fulltext index for document"""
    from aperag.index.fulltext.indexer import fulltext_indexer
    
    try:
        document, collection = _get_document_and_collection(document_id)
        content, doc_parts, local_doc = _prepare_document_content(document, collection)
        
        try:
            # Create fulltext index
            result = fulltext_indexer.create_index(
                document_id=int(document_id),
                content=content,
                doc_parts=doc_parts,
                collection=collection,
                file_path=local_doc.path
            )
            
            if not result.success:
                raise Exception(result.error)
            
            # Return index data as JSON string
            import json
            return json.dumps(result.data) if result.data else None
            
        finally:
            # Cleanup local document
            from aperag.source.base import get_source
            from aperag.schema.utils import parseCollectionConfig
            source = get_source(parseCollectionConfig(collection.config))
            source.cleanup_document(local_doc.path)
    
    except Exception as e:
        logger.error(f"Fulltext index creation failed for document {document_id}: {str(e)}")
        raise


def _create_graph_index(document_id: str) -> str:
    """Create graph index for document"""
    from aperag.index.graph.indexer import graph_indexer
    
    try:
        document, collection = _get_document_and_collection(document_id)
        
        # Check if graph indexing is enabled
        if not graph_indexer.is_enabled(collection):
            logger.info(f"Graph indexing disabled for document {document_id}")
            return None
        
        content, doc_parts, local_doc = _prepare_document_content(document, collection)
        
        try:
            # For graph indexing, we use the async approach via LightRAG
            # This will schedule the actual graph processing task
            from aperag.graph.lightrag_manager import process_document_for_celery
            
            result = process_document_for_celery(
                collection=collection,
                content=content,
                doc_id=document_id,
                file_path=local_doc.path
            )
            
            if result.get("status") != "success":
                error_msg = result.get("message", "Unknown error")
                raise Exception(f"Graph indexing failed: {error_msg}")
            
            # Return result data as JSON string
            import json
            return json.dumps(result)
            
        finally:
            # Cleanup local document
            from aperag.source.base import get_source
            from aperag.schema.utils import parseCollectionConfig
            source = get_source(parseCollectionConfig(collection.config))
            source.cleanup_document(local_doc.path)
    
    except Exception as e:
        logger.error(f"Graph index creation failed for document {document_id}: {str(e)}")
        raise


def _update_vector_index(document_id: str) -> str:
    """Update vector index for document"""
    from aperag.index.vector.indexer import vector_indexer
    
    try:
        document, collection = _get_document_and_collection(document_id)
        content, doc_parts, local_doc = _prepare_document_content(document, collection)
        
        try:
            # Update vector index
            result = vector_indexer.update_index(
                document_id=int(document_id),
                content=content,
                doc_parts=doc_parts,
                collection=collection,
                file_path=local_doc.path
            )
            
            if not result.success:
                raise Exception(result.error)
            
            # Return index data as JSON string
            import json
            return json.dumps(result.data) if result.data else None
            
        finally:
            # Cleanup local document
            from aperag.source.base import get_source
            from aperag.schema.utils import parseCollectionConfig
            source = get_source(parseCollectionConfig(collection.config))
            source.cleanup_document(local_doc.path)
    
    except Exception as e:
        logger.error(f"Vector index update failed for document {document_id}: {str(e)}")
        raise


def _update_fulltext_index(document_id: str) -> str:
    """Update fulltext index for document"""
    from aperag.index.fulltext.indexer import fulltext_indexer
    
    try:
        document, collection = _get_document_and_collection(document_id)
        content, doc_parts, local_doc = _prepare_document_content(document, collection)
        
        try:
            # Update fulltext index
            result = fulltext_indexer.update_index(
                document_id=int(document_id),
                content=content,
                doc_parts=doc_parts,
                collection=collection,
                file_path=local_doc.path
            )
            
            if not result.success:
                raise Exception(result.error)
            
            # Return index data as JSON string
            import json
            return json.dumps(result.data) if result.data else None
            
        finally:
            # Cleanup local document
            from aperag.source.base import get_source
            from aperag.schema.utils import parseCollectionConfig
            source = get_source(parseCollectionConfig(collection.config))
            source.cleanup_document(local_doc.path)
    
    except Exception as e:
        logger.error(f"Fulltext index update failed for document {document_id}: {str(e)}")
        raise


def _update_graph_index(document_id: str) -> str:
    """Update graph index for document"""
    from aperag.index.graph.indexer import graph_indexer
    
    try:
        document, collection = _get_document_and_collection(document_id)
        
        # Check if graph indexing is enabled
        if not graph_indexer.is_enabled(collection):
            logger.info(f"Graph indexing disabled for document {document_id}")
            return None
        
        content, doc_parts, local_doc = _prepare_document_content(document, collection)
        
        try:
            # For graph indexing update, we use the same process as creation
            # LightRAG will handle incremental updates internally
            from aperag.graph.lightrag_manager import process_document_for_celery
            
            result = process_document_for_celery(
                collection=collection,
                content=content,
                doc_id=document_id,
                file_path=local_doc.path
            )
            
            if result.get("status") != "success":
                error_msg = result.get("message", "Unknown error")
                raise Exception(f"Graph indexing update failed: {error_msg}")
            
            # Return result data as JSON string
            import json
            return json.dumps(result)
            
        finally:
            # Cleanup local document
            from aperag.source.base import get_source
            from aperag.schema.utils import parseCollectionConfig
            source = get_source(parseCollectionConfig(collection.config))
            source.cleanup_document(local_doc.path)
    
    except Exception as e:
        logger.error(f"Graph index update failed for document {document_id}: {str(e)}")
        raise


def _delete_vector_index(document_id: str, index_data: str = None):
    """Delete vector index for document"""
    from aperag.index.vector.indexer import vector_indexer
    
    try:
        document, collection = _get_document_and_collection(document_id)
        
        # Delete vector index
        result = vector_indexer.delete_index(
            document_id=int(document_id),
            collection=collection
        )
        
        if not result.success:
            raise Exception(result.error)
        
        logger.info(f"Vector index deleted for document {document_id}")
    
    except Exception as e:
        logger.error(f"Vector index deletion failed for document {document_id}: {str(e)}")
        raise


def _delete_fulltext_index(document_id: str, index_data: str = None):
    """Delete fulltext index for document"""
    from aperag.index.fulltext.indexer import fulltext_indexer
    
    try:
        document, collection = _get_document_and_collection(document_id)
        
        # Delete fulltext index
        result = fulltext_indexer.delete_index(
            document_id=int(document_id),
            collection=collection
        )
        
        if not result.success:
            raise Exception(result.error)
        
        logger.info(f"Fulltext index deleted for document {document_id}")
    
    except Exception as e:
        logger.error(f"Fulltext index deletion failed for document {document_id}: {str(e)}")
        raise


def _delete_graph_index(document_id: str, index_data: str = None):
    """Delete graph index for document"""
    from aperag.index.graph.indexer import graph_indexer
    
    try:
        document, collection = _get_document_and_collection(document_id)
        
        # Check if graph indexing is enabled
        if not graph_indexer.is_enabled(collection):
            logger.info(f"Graph indexing disabled for document {document_id}")
            return
        
        # Delete graph index using LightRAG
        from aperag.graph.lightrag_manager import delete_document_for_celery
        
        result = delete_document_for_celery(
            collection=collection,
            doc_id=document_id
        )
        
        if result.get("status") != "success":
            error_msg = result.get("message", "Unknown error")
            raise Exception(f"Graph index deletion failed: {error_msg}")
        
        logger.info(f"Graph index deleted for document {document_id}")
    
    except Exception as e:
        logger.error(f"Graph index deletion failed for document {document_id}: {str(e)}")
        raise 