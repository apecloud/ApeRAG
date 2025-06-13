import logging
from typing import Any, List

from celery import current_app
from aperag.tasks.collection import collection_task
from aperag.tasks.document import document_index_task
from aperag.tasks.utils import TaskConfig
from config.celery import app


logger = logging.getLogger()

@current_app.task(bind=True)
def create_index_task(self, document_id: str, index_types: list):
    """
    Create index task entry point

    Args:
        document_id: Document ID to process
        index_types: List of index types to process ['vector', 'fulltext', 'graph']
    """
    try:
        return document_index_task.create_index_task(document_id, index_types)
    except Exception as e:
        error_msg = f"Batch processing failed for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)


@current_app.task(bind=True)
def delete_index_task(self, document_id: str, index_types: List[str], index_data: str = None):
    """
    Delete index task entry point

    Args:
        document_id: Document ID to process
        index_types: List of index types to process ['vector', 'fulltext', 'graph']
    """
    try:
        return document_index_task.delete_index_task(document_id, index_types, index_data)
    except Exception as e:
        error_msg = f"Failed to delete {index_types} index for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)


@current_app.task(bind=True)
def update_index_task(self, document_id: str, index_types: List[str]):
    """Update an existing index for a document (legacy support)"""
    try:
        return document_index_task.update_index_task(document_id, index_types)
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


@app.task(bind=True)
def collection_delete_task(self, collection_id: str) -> Any:
    """
    Delete collection task entry point

    Args:
        collection_id: Collection ID to delete
    """
    try:
        result = collection_task.delete_collection(collection_id)

        if not result.success:
            raise Exception(result.error)

        logger.info(f"Collection {collection_id} deleted successfully")
        return result.to_dict()

    except Exception as e:
        logger.error(f"Collection deletion failed for {collection_id}: {str(e)}")
        raise self.retry(
            exc=e,
            countdown=TaskConfig.RETRY_COUNTDOWN_COLLECTION,
            max_retries=TaskConfig.RETRY_MAX_RETRIES_COLLECTION,
        )


@app.task(bind=True)
def collection_init_task(self, collection_id: str, document_user_quota: int) -> Any:
    """
    Initialize collection task entry point

    Args:
        collection_id: Collection ID to initialize
        document_user_quota: User quota for documents
    """
    try:
        result = collection_task.initialize_collection(collection_id, document_user_quota)

        if not result.success:
            raise Exception(result.error)

        logger.info(f"Collection {collection_id} initialized successfully")
        return result.to_dict()

    except Exception as e:
        logger.error(f"Collection initialization failed for {collection_id}: {str(e)}")
        raise self.retry(
            exc=e,
            countdown=TaskConfig.RETRY_COUNTDOWN_COLLECTION,
            max_retries=TaskConfig.RETRY_MAX_RETRIES_COLLECTION,
        )