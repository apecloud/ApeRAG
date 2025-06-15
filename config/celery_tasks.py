"""
Celery Task System for Document Indexing - Dynamic Workflow Architecture

This module implements a dynamic task system for document indexing with runtime workflow orchestration.
All tasks use structured data classes for parameter passing and result handling.

## Architecture Overview

The new task system is designed with the following principles:
1. **Fine-grained tasks**: Each operation (parse, create index, delete index, update index) is a separate task
2. **Dynamic workflow orchestration**: Tasks are composed at runtime using trigger tasks
3. **Parallel execution**: Index creation/update/deletion tasks run in parallel for better performance
4. **Individual retries**: Each task has its own retry mechanism with configurable parameters
5. **Runtime decision making**: Workflows can adapt based on document content and parsing results

## Task Flow Architecture

### Sequential Phase (Chain):
```
parse_document_task -> trigger_indexing_workflow
```

### Parallel Phase (Group + Chord):
```
[create_index_task(vector), create_index_task(fulltext), create_index_task(graph)] -> notify_workflow_complete
```

### Key Innovation: Dynamic Fan-out
The `trigger_indexing_workflow` task receives parsed document data and dynamically creates
the parallel index tasks, solving the static parameter passing limitation.

## Task Hierarchy

### Core Tasks:
- `parse_document_task`: Parse document content and extract metadata
- `create_index_task`: Create a single type of index (vector/fulltext/graph)
- `delete_index_task`: Delete a single type of index
- `update_index_task`: Update a single type of index

### Workflow Orchestration Tasks:
- `trigger_create_indexes_workflow`: Dynamic fan-out for index creation
- `trigger_delete_indexes_workflow`: Dynamic fan-out for index deletion
- `trigger_update_indexes_workflow`: Dynamic fan-out for index updates
- `notify_workflow_complete`: Aggregation task for workflow completion

### Workflow Entry Points:
- `create_document_indexes_workflow()`: Chain composition function
- `delete_document_indexes_workflow()`: Chain composition function
- `update_document_indexes_workflow()`: Chain composition function

## Usage Examples

### Direct Workflow Execution:
```python
from config.celery_tasks import create_document_indexes_workflow

# Execute workflow with dynamic orchestration
workflow_result = create_document_indexes_workflow(
    document_id="doc_123",
    index_types=["vector", "fulltext", "graph"]
)

print(f"Workflow ID: {workflow_result.id}")
```

### Via TaskScheduler:
```python
from aperag.tasks.scheduler import create_task_scheduler

scheduler = create_task_scheduler("celery")

# Execute workflow via scheduler
workflow_id = scheduler.schedule_create_index(
    document_id="doc_123", 
    index_types=["vector", "fulltext"]
)

# Check status
status = scheduler.get_task_status(workflow_id)
print(f"Success: {status.success}")
```

## Benefits of Dynamic Orchestration

1. **Runtime Parameter Passing**: Index tasks receive actual parsed document data
2. **Adaptive Workflows**: Can decide which indexes to create based on document content
3. **Better Error Isolation**: Parse failures don't create orphaned index tasks
4. **Clear Data Flow**: Each task knows exactly what data it will receive
5. **Extensible**: Easy to add conditional logic for different document types

## Error Handling and Retries

Each task has built-in retry mechanisms:
- **Max retries**: 3 attempts for most tasks
- **Retry countdown**: 60 seconds between retries
- **Exception handling**: Detailed logging and error callbacks
- **Failure notifications**: Integration with index_task_callbacks for status updates
"""

import json
import logging
from typing import Any, List

from celery import Task, current_app, group, chord, chain
from celery.exceptions import Retry
from aperag.tasks.collection import collection_task
from aperag.tasks.document import document_index_task
from aperag.tasks.utils import TaskConfig
from aperag.tasks.models import (
    ParsedDocumentData,
    IndexTaskResult, 
    WorkflowResult,
    WorkflowStatusInfo,
    TaskStatus
)
from config.celery import app


logger = logging.getLogger()

class BaseIndexTask(Task):
    abstract = True

    def _handle_index_success(self, document_id: str, index_type: str, index_data: dict = None):
        try:
            from aperag.index.reconciler import index_task_callbacks
            index_data_json = json.dumps(index_data) if index_data else None
            index_task_callbacks.on_index_created(document_id, index_type, index_data_json)
            logger.info(f"Index success callback executed for {index_type} index of document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to execute index success callback for {index_type} of {document_id}: {e}", exc_info=True)

    def _handle_index_deletion_success(self, document_id: str, index_type: str):
        try:
            from aperag.index.reconciler import index_task_callbacks
            index_task_callbacks.on_index_deleted(document_id, index_type)
            logger.info(f"Index deletion callback executed for {index_type} index of document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to execute index deletion callback for {index_type} of {document_id}: {e}", exc_info=True)

    def _handle_index_failure(self, document_id: str, index_types: List[str], error_msg: str):
        try:
            from aperag.index.reconciler import index_task_callbacks
            index_task_callbacks.on_index_failed(document_id, index_types, error_msg)
            logger.info(f"Index failure callback executed for {index_types} indexes of document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to execute index failure callback for {document_id}: {e}", exc_info=True)


# ========== Core Document Processing Tasks ==========

@current_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def parse_document_task(self, document_id: str) -> dict:
    """
    Parse document content task
    
    Args:
        document_id: Document ID to parse
        
    Returns:
        Serialized ParsedDocumentData
    """
    try:
        logger.info(f"Starting to parse document {document_id}")
        parsed_data = document_index_task.parse_document(document_id)
        logger.info(f"Successfully parsed document {document_id}")
        return parsed_data.to_dict()
    except Exception as e:
        error_msg = f"Failed to parse document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@current_app.task(bind=True, base=BaseIndexTask, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def create_index_task(self, document_id: str, index_type: str, parsed_data_dict: dict) -> dict:
    """
    Create a single index for a document
    
    Args:
        document_id: Document ID to process
        index_type: Type of index to create ('vector', 'fulltext', 'graph')
        parsed_data_dict: Serialized ParsedDocumentData from parse_document_task
        
    Returns:
        Serialized IndexTaskResult
    """
    try:
        logger.info(f"Starting to create {index_type} index for document {document_id}")
        
        # Convert dict back to structured data
        parsed_data = ParsedDocumentData.from_dict(parsed_data_dict)
        
        # Execute index creation
        result = document_index_task.create_index(document_id, index_type, parsed_data)
        
        # Handle success/failure callbacks
        if result.success:
            logger.info(f"Successfully created {index_type} index for document {document_id}")
            self._handle_index_success(document_id, index_type, result.data)
        else:
            logger.error(f"Failed to create {index_type} index for document {document_id}: {result.error}")
            # Only mark as failed if all retries are exhausted
            if self.request.retries >= self.max_retries:
                self._handle_index_failure(document_id, [index_type], result.error)
        
        return result.to_dict()
        
    except Exception as e:
        error_msg = f"Failed to create {index_type} index for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Only mark as failed if all retries are exhausted
        if self.request.retries >= self.max_retries:
            self._handle_index_failure(document_id, [index_type], error_msg)
        
        raise


@current_app.task(bind=True, base=BaseIndexTask, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def delete_index_task(self, document_id: str, index_type: str) -> dict:
    """
    Delete a single index for a document
    
    Args:
        document_id: Document ID to process
        index_type: Type of index to delete ('vector', 'fulltext', 'graph')
        
    Returns:
        Serialized IndexTaskResult
    """
    try:
        logger.info(f"Starting to delete {index_type} index for document {document_id}")
        
        # Execute index deletion
        result = document_index_task.delete_index(document_id, index_type)
        
        # Handle success/failure callbacks
        if result.success:
            logger.info(f"Successfully deleted {index_type} index for document {document_id}")
            self._handle_index_deletion_success(document_id, index_type)
        else:
            logger.error(f"Failed to delete {index_type} index for document {document_id}: {result.error}")
            # Only mark as failed if all retries are exhausted
            if self.request.retries >= self.max_retries:
                self._handle_index_failure(document_id, [index_type], result.error)
        
        return result.to_dict()
        
    except Exception as e:
        error_msg = f"Failed to delete {index_type} index for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Only mark as failed if all retries are exhausted
        if self.request.retries >= self.max_retries:
            self._handle_index_failure(document_id, [index_type], error_msg)
        
        raise


@current_app.task(bind=True, base=BaseIndexTask, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def update_index_task(self, document_id: str, index_type: str, parsed_data_dict: dict) -> dict:
    """
    Update a single index for a document
    
    Args:
        document_id: Document ID to process
        index_type: Type of index to update ('vector', 'fulltext', 'graph')
        parsed_data_dict: Serialized ParsedDocumentData from parse_document_task
        
    Returns:
        Serialized IndexTaskResult
    """
    try:
        logger.info(f"Starting to update {index_type} index for document {document_id}")
        
        # Convert dict back to structured data
        parsed_data = ParsedDocumentData.from_dict(parsed_data_dict)
        
        # Execute index update
        result = document_index_task.update_index(document_id, index_type, parsed_data)
        
        # Handle success/failure callbacks
        if result.success:
            logger.info(f"Successfully updated {index_type} index for document {document_id}")
            self._handle_index_success(document_id, index_type, result.data)
        else:
            logger.error(f"Failed to update {index_type} index for document {document_id}: {result.error}")
            # Only mark as failed if all retries are exhausted
            if self.request.retries >= self.max_retries:
                self._handle_index_failure(document_id, [index_type], result.error)
        
        return result.to_dict()
        
    except Exception as e:
        error_msg = f"Failed to update {index_type} index for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Only mark as failed if all retries are exhausted
        if self.request.retries >= self.max_retries:
            self._handle_index_failure(document_id, [index_type], error_msg)
        
        raise


# ========== Dynamic Workflow Orchestration Tasks ==========

@current_app.task(bind=True)
def trigger_create_indexes_workflow(self, parsed_data_dict: dict, document_id: str, index_types: List[str]) -> Any:
    """
    Dynamic orchestration task for index creation workflow.
    
    This task acts as a fan-out point, receiving parsed document data and dynamically
    creating parallel index creation tasks based on the actual parsed content.
    
    Args:
        parsed_data_dict: Serialized ParsedDocumentData from parse_document_task
        document_id: Document ID to process
        index_types: List of index types to create
        
    Returns:
        Chord signature for parallel index creation + completion notification
    """
    try:
        logger.info(f"Triggering parallel index creation for document {document_id} with types: {index_types}")
        
        # Dynamically create parallel index creation tasks
        parallel_index_tasks = group([
            create_index_task.s(document_id, index_type, parsed_data_dict)
            for index_type in index_types
        ])
        
        # Create chord: parallel tasks + completion notification
        workflow_chord = chord(
            parallel_index_tasks,
            notify_workflow_complete.s(document_id, "create", index_types)
        )

        chord_async_result = workflow_chord.apply_async()
        
        return chord_async_result
        
    except Exception as e:
        error_msg = f"Failed to trigger create indexes workflow: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@current_app.task(bind=True)
def trigger_delete_indexes_workflow(self, document_id: str, index_types: List[str]) -> Any:
    """
    Dynamic orchestration task for index deletion workflow.
    
    Args:
        document_id: Document ID to process
        index_types: List of index types to delete
        
    Returns:
        Chord signature for parallel index deletion + completion notification
    """
    try:
        logger.info(f"Triggering parallel index deletion for document {document_id} with types: {index_types}")
        
        # Create parallel index deletion tasks
        parallel_delete_tasks = group([
            delete_index_task.s(document_id, index_type)
            for index_type in index_types
        ])
        
        # Create chord: parallel tasks + completion notification
        workflow_chord = chord(
            parallel_delete_tasks,
            notify_workflow_complete.s(document_id, "delete", index_types)
        )

        chord_async_result = workflow_chord.apply_async()

        return chord_async_result
        
    except Exception as e:
        error_msg = f"Failed to trigger delete indexes workflow: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@current_app.task(bind=True)
def trigger_update_indexes_workflow(self, parsed_data_dict: dict, document_id: str, index_types: List[str]) -> Any:
    """
    Dynamic orchestration task for index update workflow.
    
    Args:
        parsed_data_dict: Serialized ParsedDocumentData from parse_document_task
        document_id: Document ID to process
        index_types: List of index types to update
        
    Returns:
        Chord signature for parallel index update + completion notification
    """
    try:
        logger.info(f"Triggering parallel index update for document {document_id} with types: {index_types}")
        
        # Create parallel index update tasks
        parallel_update_tasks = group([
            update_index_task.s(document_id, index_type, parsed_data_dict)
            for index_type in index_types
        ])
        
        # Create chord: parallel tasks + completion notification
        workflow_chord = chord(
            parallel_update_tasks,
            notify_workflow_complete.s(document_id, "update", index_types)
        )
        
        chord_async_result = workflow_chord.apply_async()

        return chord_async_result
        
    except Exception as e:
        error_msg = f"Failed to trigger update indexes workflow: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@current_app.task
def notify_workflow_complete(index_results: List[dict], document_id: str, operation: str, index_types: List[str]) -> dict:
    """
    Workflow completion notification task.
    
    This task is called after all parallel index operations complete,
    aggregating results and providing final workflow status.
    
    Args:
        index_results: List of IndexTaskResult dicts from parallel tasks
        document_id: Document ID that was processed
        operation: Operation type ('create', 'delete', 'update')
        index_types: List of index types that were processed
        
    Returns:
        Serialized WorkflowResult
    """
    try:
        logger.info(f"Workflow {operation} completed for document {document_id}")
        logger.info(f"Index results: {index_results}")
        
        # Analyze results
        successful_tasks = []
        failed_tasks = []
        
        for result_dict in index_results:
            try:
                result = IndexTaskResult.from_dict(result_dict)
                if result.success:
                    successful_tasks.append(result.index_type)
                else:
                    failed_tasks.append(f"{result.index_type}: {result.error}")
            except Exception as e:
                failed_tasks.append(f"unknown: {str(e)}")
        
        # Determine overall status
        if not failed_tasks:
            status = TaskStatus.SUCCESS
            status_message = f"Document {document_id} {operation} COMPLETED SUCCESSFULLY! All indexes processed: {', '.join(successful_tasks)}"
            logger.info(status_message)
        elif successful_tasks:
            status = TaskStatus.PARTIAL_SUCCESS
            status_message = f"Document {document_id} {operation} COMPLETED with WARNINGS. Success: {', '.join(successful_tasks)}. Failures: {'; '.join(failed_tasks)}"
            logger.warning(status_message)
        else:
            status = TaskStatus.FAILED
            status_message = f"Document {document_id} {operation} FAILED. All tasks failed: {'; '.join(failed_tasks)}"
            logger.error(status_message)
        
        # Create workflow result
        workflow_result = WorkflowResult(
            workflow_id=f"{document_id}_{operation}",
            document_id=document_id,
            operation=operation,
            status=status,
            message=status_message,
            successful_indexes=successful_tasks,
            failed_indexes=[f.split(':')[0] for f in failed_tasks],
            total_indexes=len(index_types),
            index_results=[IndexTaskResult.from_dict(r) for r in index_results]
        )
        
        return workflow_result.to_dict()
        
    except Exception as e:
        error_msg = f"Failed to process workflow completion for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Return failure result
        workflow_result = WorkflowResult(
            workflow_id=f"{document_id}_{operation}",
            document_id=document_id,
            operation=operation,
            status=TaskStatus.FAILED,
            message=error_msg,
            successful_indexes=[],
            failed_indexes=index_types,
            total_indexes=len(index_types),
            index_results=[]
        )
        
        return workflow_result.to_dict()


# ========== Workflow Entry Point Functions ==========

def create_document_indexes_workflow(document_id: str, index_types: List[str]):
    """
    Create indexes for a document using dynamic workflow orchestration.
    
    This function composes a chain that:
    1. Parses the document
    2. Dynamically triggers parallel index creation based on parsed content
    3. Aggregates results and notifies completion
    
    Args:
        document_id: Document ID to process
        index_types: List of index types to create
        
    Returns:
        AsyncResult for the workflow chain
    """
    logger.info(f"Starting create indexes workflow for document {document_id} with types: {index_types}")
    # Create the workflow chain: parse -> dynamic trigger
    workflow_chain = chain(
        parse_document_task.s(document_id),
        trigger_create_indexes_workflow.s(document_id, index_types)
    )
    
    # Submit the workflow
    workflow_result = workflow_chain.delay()
    logger.info(f"Create indexes workflow submitted for document {document_id}, workflow ID: {workflow_result.id}")
    
    return workflow_result


def delete_document_indexes_workflow(document_id: str, index_types: List[str]):
    """
    Delete indexes for a document using dynamic workflow orchestration.
    
    Args:
        document_id: Document ID to process
        index_types: List of index types to delete
        
    Returns:
        AsyncResult for the workflow
    """
    logger.info(f"Starting delete indexes workflow for document {document_id} with types: {index_types}")
    
    # For deletion, we don't need parsing, so we directly trigger the delete workflow
    workflow_result = trigger_delete_indexes_workflow.delay(document_id, index_types)
    logger.info(f"Delete indexes workflow submitted for document {document_id}, workflow ID: {workflow_result.id}")
    
    return workflow_result


def update_document_indexes_workflow(document_id: str, index_types: List[str]):
    """
    Update indexes for a document using dynamic workflow orchestration.
    
    This function composes a chain that:
    1. Re-parses the document to get updated content
    2. Dynamically triggers parallel index updates based on parsed content
    3. Aggregates results and notifies completion
    
    Args:
        document_id: Document ID to process
        index_types: List of index types to update
        
    Returns:
        AsyncResult for the workflow chain
    """
    logger.info(f"Starting update indexes workflow for document {document_id} with types: {index_types}")
    
    # Create the workflow chain: parse -> dynamic trigger
    workflow_chain = chain(
        parse_document_task.s(document_id),
        trigger_update_indexes_workflow.s(document_id, index_types)
    )
    
    # Submit the workflow
    workflow_result = workflow_chain.delay()
    logger.info(f"Update indexes workflow submitted for document {document_id}, workflow ID: {workflow_result.id}")
    
    return workflow_result


# ========== Workflow Status Monitoring ==========

@current_app.task(bind=True)
def get_workflow_status(self, workflow_id: str) -> dict:
    """
    Get the status of a workflow by ID
    
    Args:
        workflow_id: Workflow ID to check
        
    Returns:
        Serialized WorkflowStatusInfo
    """
    try:
        from celery.result import AsyncResult
        
        # Get the workflow result
        workflow_result = AsyncResult(workflow_id, app=current_app)
        
        # Determine status based on Celery task state
        if workflow_result.state == 'PENDING':
            status = TaskStatus.RUNNING
            message = "Workflow is pending execution"
            progress = 0
        elif workflow_result.state == 'STARTED':
            status = TaskStatus.RUNNING  
            message = "Workflow has started"
            progress = 25
        elif workflow_result.state == 'SUCCESS':
            status = TaskStatus.SUCCESS
            message = "Workflow completed successfully"
            progress = 100
        elif workflow_result.state == 'FAILURE':
            status = TaskStatus.FAILED
            message = f"Workflow failed: {str(workflow_result.info)}"
            progress = 0
        else:
            status = TaskStatus.RUNNING
            message = f"Workflow state: {workflow_result.state}"
            progress = 50
        
        # Create status info
        status_info = WorkflowStatusInfo(
            workflow_id=workflow_id,
            status=status,
            message=message,
            progress=progress,
            result=workflow_result.result if workflow_result.state == 'SUCCESS' else None
        )
        
        return status_info.to_dict()
        
    except Exception as e:
        error_msg = f"Failed to get workflow status for {workflow_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Return error status
        status_info = WorkflowStatusInfo(
            workflow_id=workflow_id,
            status=TaskStatus.FAILED,
            message=error_msg,
            progress=0,
            result=None
        )
        
        return status_info.to_dict()


# ========== Collection Tasks ==========

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


