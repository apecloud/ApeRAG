import logging
from typing import List

from celery import current_app

from aperag.db.models import DocumentIndexType
from aperag.index.reconciler import index_task_callbacks
from aperag.tasks.utils import cleanup_local_document, parse_document_content

logger = logging.getLogger(__name__)


def _create_index_task(document_id: str, index_types: list):
    """
    Process multiple indexes for a document in batch to avoid duplicate parsing

    Args:
        document_id: Document ID to process
        index_types: List of index types to process ['vector', 'fulltext', 'graph']
    """
    logger.info(f"Batch processing {index_types} indexes for document {document_id}")

    # Import here to avoid circular dependencies
    from aperag.index.fulltext_index import fulltext_indexer
    from aperag.index.graph_index import graph_indexer
    from aperag.index.vector_index import vector_indexer
    from aperag.tasks.utils import get_document_and_collection

    # Parse document once
    document, collection = get_document_and_collection(document_id)
    content, doc_parts, local_doc = parse_document_content(document, collection)
    file_path = local_doc.path

    results = {}

    try:
        # Process each requested index type
        for index_type in index_types:
            try:
                result = None
                if index_type == DocumentIndexType.VECTOR.value:
                    result = vector_indexer.create_index(
                        document_id=document_id,
                        content=content,
                        doc_parts=doc_parts,
                        collection=collection,
                        file_path=file_path,
                    )
                elif index_type == DocumentIndexType.FULLTEXT.value:
                    result = fulltext_indexer.create_index(
                        document_id=document_id,
                        content=content,
                        doc_parts=doc_parts,
                        collection=collection,
                        file_path=file_path,
                    )
                elif index_type == DocumentIndexType.GRAPH.value:
                    if graph_indexer.is_enabled(collection):
                        from aperag.graph.lightrag_manager import process_document_for_celery

                        result = process_document_for_celery(
                            collection=collection, content=content, doc_id=document_id, file_path=file_path
                        )
                    else:
                        logger.info(f"Graph indexing disabled for document {document_id}")
                        results["graph"] = {"success": True, "data": None, "message": "disabled"}
                else:
                    raise ValueError(f"Unknown index type: {index_type}")
                if not result:
                    continue
                if result.success:
                    import json

                    index_data = json.dumps(result.data) if result.data else None
                    index_task_callbacks.on_index_created(document_id, index_type, index_data)
                    results[index_type] = {"success": True, "data": index_data}
                else:
                    raise Exception(result.error)

            except Exception as e:
                error_msg = f"Failed to create index {index_type}: {str(e)}"
                logger.error(f"Document {document_id}: {error_msg}")
                index_task_callbacks.on_index_failed(document_id, [index_type], error_msg)
                results[index_type] = {"success": False, "error": error_msg}
                raise e

    finally:
        # Cleanup local document
        cleanup_local_document(local_doc, collection)

    logger.info(f"Batch processing completed for document {document_id}: {results}")
    return results


@current_app.task(bind=True)
def create_index_task(self, document_id: str, index_types: list):
    """
    Create index task entry point

    Args:
        document_id: Document ID to process
        index_types: List of index types to process ['vector', 'fulltext', 'graph']
    """
    try:
        return _create_index_task(document_id, index_types)
    except Exception as e:
        error_msg = f"Batch processing failed for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)


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
    from aperag.index.vector_index import vector_indexer
    from aperag.tasks.utils import get_document_and_collection

    # Get document and collection info
    document, collection = get_document_and_collection(document_id)

    for index_type in index_types:
        try:
            if index_type == DocumentIndexType.VECTOR.value:
                result = vector_indexer.delete_index(document_id, collection)
                if not result.success:
                    raise Exception(result.error)
            elif index_type == DocumentIndexType.FULLTEXT.value:
                result = fulltext_indexer.delete_index(document_id, collection)
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
            index_task_callbacks.on_index_deleted(document_id, index_type)
        except Exception as e:
            error_msg = f"Failed to delete index {index_type}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            index_task_callbacks.on_index_failed(document_id, [index_type], error_msg)
            raise e

    logger.info(f"Successfully deleted {index_types} index for document {document_id}")


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
        logger.error(error_msg, exc_info=True)


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
    from aperag.index.vector_index import vector_indexer
    from aperag.tasks.utils import get_document_and_collection

    # Parse document once
    document, collection = get_document_and_collection(document_id)
    content, doc_parts, local_doc = parse_document_content(document, collection)
    file_path = local_doc.path

    try:
        for index_type in index_types:
            try:
                if index_type == DocumentIndexType.VECTOR.value:
                    result = vector_indexer.update_index(
                        document_id=document_id,
                        content=content,
                        doc_parts=doc_parts,
                        collection=collection,
                        file_path=file_path,
                    )
                    if not result.success:
                        raise Exception(result.error)
                    import json

                    index_data = json.dumps(result.data) if result.data else None

                elif index_type == DocumentIndexType.FULLTEXT.value:
                    result = fulltext_indexer.update_index(
                        document_id=document_id,
                        content=content,
                        doc_parts=doc_parts,
                        collection=collection,
                        file_path=file_path,
                    )
                    if not result.success:
                        raise Exception(result.error)
                    import json

                    index_data = json.dumps(result.data) if result.data else None

                elif index_type == DocumentIndexType.GRAPH.value:
                    if graph_indexer.is_enabled(collection):
                        from aperag.graph.lightrag_manager import process_document_for_celery

                        result = process_document_for_celery(
                            collection=collection, content=content, doc_id=document_id, file_path=file_path
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

                # Notify completion
                index_task_callbacks.on_index_created(document_id, index_type, index_data)
            except Exception as e:
                error_msg = f"Failed to update index {index_type}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                index_task_callbacks.on_index_failed(document_id, [index_type], error_msg)
                raise e

    finally:
        cleanup_local_document(local_doc, collection)

    logger.info(f"Successfully updated {index_type} index for document {document_id}")


@current_app.task(bind=True)
def update_index_task(self, document_id: str, index_types: List[str]):
    """Update an existing index for a document (legacy support)"""
    try:
        return _update_index_task(document_id, index_types)
    except Exception as e:
        error_msg = f"Failed to update {index_types} index for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)


@current_app.task
def reconcile_indexes_task():
    """Periodic task to reconcile index specs with statuses"""
    try:
        logger.info("Starting index reconciliation")

        # Import here to avoid circular dependencies
        from aperag.index.reconciler import index_reconciler

        # Run reconciliation
        index_reconciler.reconcile_all()

        logger.info("Index reconciliation completed")

    except Exception as e:
        logger.error(f"Index reconciliation failed: {e}", exc_info=True)
        raise
