import logging
from celery import current_app
from aperag.index.reconciler import index_task_callbacks

import asyncio
import json

logger = logging.getLogger(__name__)


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
        logger.error(error_msg)

        # Notify failure for all requested index types
        asyncio.run(index_task_callbacks.on_index_failed(document_id, index_types, error_msg))

        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60, max_retries=3)
    

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
    from aperag.index.operations import cleanup_local_document, get_document_and_collection, parse_document_content
    from aperag.index.vector_index import vector_indexer

    # Parse document once
    document, collection = get_document_and_collection(document_id)
    content, doc_parts, local_doc = parse_document_content(document, collection)
    file_path = local_doc.path

    results = {}

    try:
        # Process each requested index type
        for index_type in index_types:
            try:
                if index_type == 'vector':
                    result = vector_indexer.create_index(
                        document_id=int(document_id),
                        content=content,
                        doc_parts=doc_parts,
                        collection=collection,
                        file_path=file_path
                    )
                    if result.success:
                        import json
                        index_data = json.dumps(result.data) if result.data else None
                        asyncio.run(index_task_callbacks.on_index_created(
                            document_id, 'vector', index_data
                        ))
                        results['vector'] = {'success': True, 'data': index_data}
                    else:
                        raise Exception(result.error)

                elif index_type == 'fulltext':
                    result = fulltext_indexer.create_index(
                        document_id=int(document_id),
                        content=content,
                        doc_parts=doc_parts,
                        collection=collection,
                        file_path=file_path
                    )
                    if result.success:
                        import json
                        index_data = json.dumps(result.data) if result.data else None
                        asyncio.run(index_task_callbacks.on_index_created(
                            document_id, 'fulltext', index_data
                        ))
                        results['fulltext'] = {'success': True, 'data': index_data}
                    else:
                        raise Exception(result.error)

                elif index_type == 'graph':
                    if graph_indexer.is_enabled(collection):
                        from aperag.graph.lightrag_manager import process_document_for_celery
                        result = process_document_for_celery(
                            collection=collection,
                            content=content,
                            doc_id=document_id,
                            file_path=file_path
                        )
                        if result.get("status") == "success":
                            import json
                            index_data = json.dumps(result)
                            asyncio.run(index_task_callbacks.on_index_created(
                                document_id, 'graph', index_data
                            ))
                            results['graph'] = {'success': True, 'data': index_data}
                        else:
                            error_msg = result.get("message", "Unknown error")
                            raise Exception(f"Graph indexing failed: {error_msg}")
                    else:
                        logger.info(f"Graph indexing disabled for document {document_id}")
                        results['graph'] = {'success': True, 'data': None, 'message': 'disabled'}

                else:
                    raise ValueError(f"Unknown index type: {index_type}")

            except Exception as e:
                error_msg = f"Failed to create {index_type} index: {str(e)}"
                logger.error(f"Document {document_id}: {error_msg}")
                asyncio.run(index_task_callbacks.on_index_failed(
                    document_id, index_type, error_msg
                ))
                results[index_type] = {'success': False, 'error': error_msg}

    finally:
        # Cleanup local document
        cleanup_local_document(local_doc, collection)

    logger.info(f"Batch processing completed for document {document_id}: {results}")
    return results