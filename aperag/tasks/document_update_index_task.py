import logging
from celery import current_app
from aperag.db.models import DocumentIndexType
from aperag.index.reconciler import index_task_callbacks


import asyncio
import json
from typing import List

logger = logging.getLogger(__name__)


@current_app.task(bind=True)
def update_index_task(self, document_id: str, index_types: List[str]):
    """Update an existing index for a document (legacy support)"""
    try:
        return _update_index_task(document_id, index_types)
    except Exception as e:
        error_msg = f"Failed to update {index_types} index for document {document_id}: {str(e)}"
        logger.error(error_msg)

        # Notify failure
        asyncio.run(index_task_callbacks.on_index_failed(document_id, index_types, error_msg))

        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60, max_retries=3)

def _update_index_task(document_id: str, index_types: List[str]):
    """
    Update an existing index for a document

    Args:
        document_id: Document ID to process
        index_types: List of index types to process ['vector', 'fulltext', 'graph']
    """
    logger.info(f"Updating {index_types} index for document {document_id}")

    # Call indexers directly (same as create for most indexers)
    from aperag.index.fulltext_index import fulltext_indexer
    from aperag.index.graph_index import graph_indexer
    from aperag.index.operations import cleanup_local_document, get_document_and_collection, parse_document_content
    from aperag.index.vector_index import vector_indexer

    # Parse document once
    document, collection = get_document_and_collection(document_id)
    content, doc_parts, local_doc = parse_document_content(document, collection)
    file_path = local_doc.path

    try:
        for index_type in index_types:
            if index_type == DocumentIndexType.VECTOR.value:
                result = vector_indexer.update_index(
                    document_id=int(document_id),
                    content=content,
                    doc_parts=doc_parts,
                    collection=collection,
                    file_path=file_path
                )
                if not result.success:
                    raise Exception(result.error)
                import json
                index_data = json.dumps(result.data) if result.data else None

            elif index_type == DocumentIndexType.FULLTEXT.value:
                result = fulltext_indexer.update_index(
                    document_id=int(document_id),
                    content=content,
                    doc_parts=doc_parts,
                    collection=collection,
                    file_path=file_path
                )
                if not result.success:
                    raise Exception(result.error)
                import json
                index_data = json.dumps(result.data) if result.data else None

            elif index_type == DocumentIndexType.GRAPH.value:
                if graph_indexer.is_enabled(collection):
                    from aperag.graph.lightrag_manager import process_document_for_celery
                    result = process_document_for_celery(
                        collection=collection,
                        content=content,
                        doc_id=document_id,
                        file_path=file_path
                    )
                    if result.get("status") != "success":
                        error_msg = result.get("message", "Unknown error")
                        raise Exception(f"Graph indexing update failed: {error_msg}")
                    import json
                    index_data = json.dumps(result)
                else:
                    index_data = None
            else:
                raise ValueError(f"Unknown index type: {index_type}")
    finally:
        cleanup_local_document(local_doc, collection)

    # Notify completion
    asyncio.run(index_task_callbacks.on_index_created(document_id, index_type, index_data))

    logger.info(f"Successfully updated {index_type} index for document {document_id}")
