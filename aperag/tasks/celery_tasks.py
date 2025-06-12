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

"""
Celery tasks entry points
This module only handles task orchestration and error handling
All business logic is delegated to service layer
"""

import logging
from datetime import datetime
from typing import Any, List

from celery import Task

from aperag.db.models import DocumentIndexStatusOld, DocumentStatus
from aperag.db.ops import db_ops
from aperag.workflow.collection_workflow import collection_workflow
from aperag.workflow.document_workflow import document_workflow
from config.celery import app
from aperag.tasks.reconciliation_tasks import run_reconciliation_once

logger = logging.getLogger(__name__)


# Configuration constants
class TaskConfig:
    RETRY_COUNTDOWN_COLLECTION = 60
    RETRY_MAX_RETRIES_COLLECTION = 2
    RETRY_COUNTDOWN_DOCUMENT = 60
    RETRY_MAX_RETRIES_DOCUMENT = 2


class DocumentIndexTask(Task):
    """Base task for document indexing operations"""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle successful task completion"""
        if args:
            document_id = args[0]
            document = db_ops.query_document_by_id(document_id)
            if document:
                document.update_overall_status()
                db_ops.update_document(document)
                logger.info(f"Document indexing completed successfully for document {document.name}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        if args:
            document_id = args[0]
            document = db_ops.query_document_by_id(document_id)
            if document:
                # Set all index statuses to failed
                document.vector_index_status = DocumentIndexStatusOld.FAILED
                document.fulltext_index_status = DocumentIndexStatusOld.FAILED
                document.graph_index_status = DocumentIndexStatusOld.FAILED
                document.update_overall_status()
                db_ops.update_document(document)
                logger.error(f"Document indexing failed for document {document.name}: {exc}")


class DocumentDeleteTask(Task):
    """Base task for document deletion operations"""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle successful task completion"""
        if args:
            document_id = args[0]
            document = db_ops.query_document_by_id(document_id)
            if document:
                document.status = DocumentStatus.DELETED
                document.gmt_deleted = datetime.utcnow()
                document.name = f"{document.name}-{document_id}"
                db_ops.update_document(document)
                logger.info(f"Document deletion completed successfully for document {document.name}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        if args:
            document_id = args[0]
            document = db_ops.query_document_by_id(document_id)
            if document:
                document.status = DocumentStatus.FAILED
                db_ops.update_document(document)
                logger.error(f"Document deletion failed for document {document.name}: {exc}")


# Collection Tasks
@app.task(bind=True)
def init_collection_task(self, collection_id: str, document_user_quota: int) -> Any:
    """
    Initialize collection task entry point
    
    Args:
        collection_id: Collection ID to initialize
        document_user_quota: User quota for documents
    """
    try:
        result = collection_workflow.initialize_collection(collection_id, document_user_quota)
        
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


@app.task(bind=True)
def delete_collection_task(self, collection_id: str) -> Any:
    """
    Delete collection task entry point
    
    Args:
        collection_id: Collection ID to delete
    """
    try:
        result = collection_workflow.delete_collection(collection_id)
        
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


# Document Tasks
@app.task(base=DocumentIndexTask, bind=True, track_started=True)
def add_index_for_document_task(self, document_id: int) -> Any:
    """
    Add document index task entry point
    
    Args:
        document_id: Document ID to index
    """
    try:
        result = document_workflow.add_document_index(document_id)
        
        if not result.success:
            raise Exception(result.error)
        
        logger.info(f"Document {document_id} indexed successfully")
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Document indexing failed for {document_id}: {str(e)}")
        raise self.retry(
            exc=e,
            countdown=TaskConfig.RETRY_COUNTDOWN_DOCUMENT,
            max_retries=TaskConfig.RETRY_MAX_RETRIES_DOCUMENT,
        )


@app.task(base=DocumentDeleteTask, bind=True, track_started=True)
def remove_index_task(self, document_id: int) -> Any:
    """
    Remove document index task entry point
    
    Args:
        document_id: Document ID to remove
    """
    try:
        result = document_workflow.remove_document_index(document_id)
        
        if not result.success:
            raise Exception(result.error)
        
        logger.info(f"Document {document_id} removed successfully")
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Document removal failed for {document_id}: {str(e)}")
        raise self.retry(
            exc=e,
            countdown=TaskConfig.RETRY_COUNTDOWN_DOCUMENT,
            max_retries=TaskConfig.RETRY_MAX_RETRIES_DOCUMENT,
        )


@app.task(base=DocumentIndexTask, bind=True, track_started=True)
def update_index_for_document_task(self, document_id: int) -> Any:
    """
    Update document index task entry point
    
    Args:
        document_id: Document ID to update
    """
    try:
        result = document_workflow.update_document_index(document_id)
        
        if not result.success:
            raise Exception(result.error)
        
        logger.info(f"Document {document_id} updated successfully")
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Document update failed for {document_id}: {str(e)}")
        raise self.retry(
            exc=e,
            countdown=TaskConfig.RETRY_COUNTDOWN_DOCUMENT,
            max_retries=TaskConfig.RETRY_MAX_RETRIES_DOCUMENT,
        )


# Legacy compatibility - keep same task names as before
@app.task(base=DocumentIndexTask, bind=True, ignore_result=True)
def add_index_for_local_document(self, document_id: int) -> Any:
    """Legacy compatibility for local document indexing"""
    return add_index_for_document_task.delay(document_id)


# Knowledge Graph Tasks
@app.task(bind=True, track_started=True)
def add_lightrag_index_task(self, content: str, document_id: int, file_path: str) -> Any:
    """
    Add LightRAG index task entry point
    
    Args:
        content: Document content
        document_id: Document ID
        file_path: File path for logging
    """
    try:
        result = document_workflow.add_lightrag_index(content, document_id, file_path)
        
        if not result.success:
            raise Exception(result.error)
        
        logger.info(f"LightRAG index added successfully for document {document_id}")
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"LightRAG indexing failed for document {document_id}: {str(e)}")
        raise self.retry(
            exc=e,
            countdown=TaskConfig.RETRY_COUNTDOWN_DOCUMENT,
            max_retries=TaskConfig.RETRY_MAX_RETRIES_DOCUMENT,
        )


@app.task(bind=True, track_started=True)
def remove_lightrag_index_task(self, document_id: int, collection_id: str) -> Any:
    """
    Remove LightRAG index task entry point
    
    Args:
        document_id: Document ID
        collection_id: Collection ID
    """
    try:
        result = document_workflow.remove_lightrag_index(document_id, collection_id)
        
        if not result.success:
            raise Exception(result.error)
        
        logger.info(f"LightRAG index removed successfully for document {document_id}")
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"LightRAG removal failed for document {document_id}: {str(e)}")
        raise self.retry(
            exc=e,
            countdown=TaskConfig.RETRY_COUNTDOWN_DOCUMENT,
            max_retries=TaskConfig.RETRY_MAX_RETRIES_DOCUMENT,
        )


# New K8s-inspired reconciliation tasks

@shared_task(bind=True, soft_time_limit=60, time_limit=120)
def reconcile_document_indexes(self):
    """
    Run reconciliation once to align desired and actual index states
    
    This is the main task that runs periodically to ensure all document indexes
    are in their desired state.
    
    Returns:
        Dict: Reconciliation result
    """
    try:
        run_reconciliation_once()
        
        logger.debug("Document index reconciliation completed successfully")
        
        return {
            "status": "success",
            "message": "Reconciliation completed"
        }
        
    except Exception as e:
        logger.error(f"Error in reconcile_document_indexes: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=3) 