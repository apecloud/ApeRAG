import logging
from celery import current_app
from aperag.db.models import DocumentIndexType
from aperag.index.reconciler import index_task_callbacks


import asyncio
from typing import List

logger = logging.getLogger(__name__)


@current_app.task(bind=True)
def delete_index_task(self, document_id: str, index_types: List[str], index_data: str = None):
    """
    Delete index task entry point

    Args:
        document_id: Document ID to process
        index_types: List of index types to process ['vector', 'fulltext', 'graph']
    """
    try:
        return _delete_index_task(document_id, index_types, index_data)
    except Exception as e:
        error_msg = f"Failed to delete {index_types} index for document {document_id}: {str(e)}"
        logger.error(error_msg)

        # Notify failure
        asyncio.run(index_task_callbacks.on_index_failed(document_id, index_types, error_msg))

        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60, max_retries=3)


def _delete_index_task(document_id: str, index_types: List[str], index_data: str = None):
    """
    Delete an index for a document

    Args:
        document_id: Document ID to process
        index_types: List of index types to process ['vector', 'fulltext', 'graph']
    """
    logger.info(f"Deleting {index_types} index for document {document_id}")

    # Call indexers directly for deletion (no document parsing needed)
    from aperag.index.fulltext_index import fulltext_indexer
    from aperag.index.graph_index import graph_indexer
    from aperag.index.operations import get_document_and_collection
    from aperag.index.vector_index import vector_indexer

    # Get document and collection info
    document, collection = get_document_and_collection(document_id)

    for index_type in index_types:
        if index_type == DocumentIndexType.VECTOR.value:
            result = vector_indexer.delete_index(int(document_id), collection)
            if not result.success:
                raise Exception(result.error)
        elif index_type == DocumentIndexType.FULLTEXT.value:
            result = fulltext_indexer.delete_index(int(document_id), collection)
            if not result.success:
                raise Exception(result.error)
        elif index_type == DocumentIndexType.GRAPH.value:
            if graph_indexer.is_enabled(collection):
                from aperag.graph.lightrag_manager import delete_document_for_celery
                result = delete_document_for_celery(collection=collection, doc_id=document_id)
                if result.get("status") != "success":
                    error_msg = result.get("message", "Unknown error")
                    raise Exception(f"Graph index deletion failed: {error_msg}")
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    # Notify completion
    asyncio.run(index_task_callbacks.on_index_deleted(document_id, index_types))

    logger.info(f"Successfully deleted {index_types} index for document {document_id}")
