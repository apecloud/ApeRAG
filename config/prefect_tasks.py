"""
Prefect Flow System for Document Indexing - Dynamic Workflow Architecture

This module implements a dynamic task system for document indexing with runtime workflow orchestration using Prefect.
All tasks use structured data classes for parameter passing and result handling.

## Architecture Overview

The Prefect task system is designed with the following principles:
1. **Fine-grained tasks**: Each operation (parse, create index, delete index, update index) is a separate task
2. **Dynamic workflow orchestration**: Tasks are composed at runtime using flows
3. **Parallel execution**: Index creation/update/deletion tasks run in parallel for better performance
4. **Individual retries**: Each task has its own retry mechanism with configurable parameters
5. **Runtime decision making**: Workflows can adapt based on document content and parsing results

## Task Flow Architecture

### Sequential Phase:
```
parse_document_task -> trigger_indexing_workflow
```

### Parallel Phase:
```
[create_index_task(vector), create_index_task(fulltext), create_index_task(graph)] -> notify_workflow_complete
```

### Key Innovation: Dynamic Fan-out
The indexing flows receive parsed document data and dynamically create
the parallel index tasks, solving the static parameter passing limitation.

## Task Hierarchy

### Core Tasks:
- `parse_document_task`: Parse document content and extract metadata
- `create_index_task`: Create a single type of index (vector/fulltext/graph)
- `delete_index_task`: Delete a single type of index
- `update_index_task`: Update a single type of index

### Workflow Flows:
- `create_document_indexes_flow`: Dynamic flow for index creation
- `delete_document_indexes_flow`: Dynamic flow for index deletion
- `update_document_indexes_flow`: Dynamic flow for index updates
- `collection_init_flow`: Flow for collection initialization
- `collection_delete_flow`: Flow for collection deletion
- `reconcile_indexes_flow`: Flow for index reconciliation

## Usage Examples

### Direct Flow Execution:
```python
from config.prefect_tasks import create_document_indexes_flow

# Execute flow with dynamic orchestration
flow_run = create_document_indexes_flow(
    document_id="doc_123",
    index_types=["vector", "fulltext", "graph"]
)

print(f"Flow Run ID: {flow_run.id}")
```

### Via TaskScheduler:
```python
from aperag.tasks.scheduler import create_task_scheduler

scheduler = create_task_scheduler("prefect")

# Execute workflow via scheduler
flow_id = scheduler.schedule_create_index(
    document_id="doc_123", 
    index_types=["vector", "fulltext"]
)

# Check status
status = scheduler.get_task_status(flow_id)
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

import logging
from typing import Any, List, Optional

from prefect import task, flow
from prefect.context import get_run_context
from prefect.runtime import flow_run
from aperag.tasks.collection import collection_task
from aperag.tasks.document import document_index_task
from aperag.tasks.models import (
    ParsedDocumentData,
    IndexTaskResult, 
    WorkflowResult,
    TaskStatus
)

logger = logging.getLogger(__name__)

# ========== Core Document Processing Tasks ==========

@task(retries=3, retry_delay_seconds=60)
def parse_document_task(document_id: str) -> dict:
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


@task(retries=3, retry_delay_seconds=60)
def create_index_task(document_id: str, index_type: str, parsed_data_dict: dict) -> dict:
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
            _handle_index_success(document_id, index_type, result.data)
        else:
            logger.error(f"Failed to create {index_type} index for document {document_id}: {result.error}")
            _handle_index_failure(document_id, [index_type], result.error)
        
        return result.to_dict()
        
    except Exception as e:
        error_msg = f"Failed to create {index_type} index for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        _handle_index_failure(document_id, [index_type], error_msg)
        raise


@task(retries=3, retry_delay_seconds=60)
def delete_index_task(document_id: str, index_type: str) -> dict:
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
            _handle_index_deletion_success(document_id, index_type)
        else:
            logger.error(f"Failed to delete {index_type} index for document {document_id}: {result.error}")
            _handle_index_failure(document_id, [index_type], result.error)
        
        return result.to_dict()
        
    except Exception as e:
        error_msg = f"Failed to delete {index_type} index for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        _handle_index_failure(document_id, [index_type], error_msg)
        raise


@task(retries=3, retry_delay_seconds=60)
def update_index_task(document_id: str, index_type: str, parsed_data_dict: dict) -> dict:
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
            _handle_index_success(document_id, index_type, result.data)
        else:
            logger.error(f"Failed to update {index_type} index for document {document_id}: {result.error}")
            _handle_index_failure(document_id, [index_type], result.error)
        
        return result.to_dict()
        
    except Exception as e:
        error_msg = f"Failed to update {index_type} index for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        _handle_index_failure(document_id, [index_type], error_msg)
        raise


# ========== Collection Tasks ==========

@task(retries=3, retry_delay_seconds=60)
def collection_init_task(collection_id: str, document_user_quota: int) -> dict:
    """
    Initialize collection task
    
    Args:
        collection_id: Collection ID to initialize
        document_user_quota: User quota for documents
        
    Returns:
        Serialized task result
    """
    try:
        logger.info(f"Starting to initialize collection {collection_id}")
        result = collection_task.initialize_collection(collection_id, document_user_quota)
        
        if not result.success:
            raise Exception(result.error)
        
        logger.info(f"Collection {collection_id} initialized successfully")
        return result.to_dict()
        
    except Exception as e:
        error_msg = f"Collection initialization failed for {collection_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@task(retries=3, retry_delay_seconds=60)
def collection_delete_task(collection_id: str) -> dict:
    """
    Delete collection task
    
    Args:
        collection_id: Collection ID to delete
        
    Returns:
        Serialized task result
    """
    try:
        logger.info(f"Starting to delete collection {collection_id}")
        result = collection_task.delete_collection(collection_id)
        
        if not result.success:
            raise Exception(result.error)
        
        logger.info(f"Collection {collection_id} deleted successfully")
        return result.to_dict()
        
    except Exception as e:
        error_msg = f"Collection deletion failed for {collection_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@task
def reconcile_indexes_task() -> str:
    """Periodic task to reconcile index specs with statuses"""
    try:
        logger.info("Starting index reconciliation")

        # Import here to avoid circular dependencies
        from aperag.index.reconciler import index_reconciler

        # Run reconciliation
        index_reconciler.reconcile_all()

        logger.info("Index reconciliation completed")
        return "Index reconciliation completed successfully"

    except Exception as e:
        error_msg = f"Index reconciliation failed: {e}"
        logger.error(error_msg, exc_info=True)
        raise


# ========== Workflow Flows ==========

@flow(name="create-document-indexes")
def create_document_indexes_flow(document_id: str, index_types: List[str]) -> dict:
    """
    Create indexes for a document using dynamic workflow orchestration.
    
    This flow:
    1. Parses the document
    2. Dynamically triggers parallel index creation based on parsed content
    3. Aggregates results and provides completion status
    
    Args:
        document_id: Document ID to process
        index_types: List of index types to create
        
    Returns:
        Serialized WorkflowResult
    """
    logger.info(f"Starting create indexes flow for document {document_id} with types: {index_types}")
    
    try:
        # Parse document first
        parsed_data_dict = parse_document_task(document_id)
        
        # Create parallel index creation tasks
        index_task_futures = []
        for index_type in index_types:
            future = create_index_task.submit(document_id, index_type, parsed_data_dict)
            index_task_futures.append(future)
        
        # Wait for all index tasks to complete
        index_results = []
        for future in index_task_futures:
            try:
                result = future.result()
                index_results.append(result)
            except Exception as e:
                # Create failure result for this index type
                error_result = IndexTaskResult(
                    status=TaskStatus.FAILED,
                    index_type="unknown",
                    document_id=document_id,
                    success=False,
                    error=str(e)
                )
                index_results.append(error_result.to_dict())
        
        # Aggregate results
        workflow_result = _create_workflow_result(
            document_id, "create", index_types, index_results
        )
        
        logger.info(f"Create indexes flow completed for document {document_id}")
        return workflow_result.to_dict()
        
    except Exception as e:
        error_msg = f"Create indexes flow failed for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Return failure result
        workflow_result = WorkflowResult(
            workflow_id=f"{document_id}_create_{flow_run.id}",
            document_id=document_id,
            operation="create",
            status=TaskStatus.FAILED,
            message=error_msg,
            successful_indexes=[],
            failed_indexes=index_types,
            total_indexes=len(index_types),
            index_results=[]
        )
        
        return workflow_result.to_dict()


@flow(name="delete-document-indexes")
def delete_document_indexes_flow(document_id: str, index_types: List[str]) -> dict:
    """
    Delete indexes for a document using dynamic workflow orchestration.
    
    Args:
        document_id: Document ID to process
        index_types: List of index types to delete
        
    Returns:
        Serialized WorkflowResult
    """
    logger.info(f"Starting delete indexes flow for document {document_id} with types: {index_types}")
    
    try:
        # Create parallel index deletion tasks
        index_task_futures = []
        for index_type in index_types:
            future = delete_index_task.submit(document_id, index_type)
            index_task_futures.append(future)
        
        # Wait for all index tasks to complete
        index_results = []
        for future in index_task_futures:
            try:
                result = future.result()
                index_results.append(result)
            except Exception as e:
                # Create failure result for this index type
                error_result = IndexTaskResult(
                    status=TaskStatus.FAILED,
                    index_type="unknown",
                    document_id=document_id,
                    success=False,
                    error=str(e)
                )
                index_results.append(error_result.to_dict())
        
        # Aggregate results
        workflow_result = _create_workflow_result(
            document_id, "delete", index_types, index_results
        )
        
        logger.info(f"Delete indexes flow completed for document {document_id}")
        return workflow_result.to_dict()
        
    except Exception as e:
        error_msg = f"Delete indexes flow failed for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Return failure result
        workflow_result = WorkflowResult(
            workflow_id=f"{document_id}_delete_{flow_run.id}",
            document_id=document_id,
            operation="delete",
            status=TaskStatus.FAILED,
            message=error_msg,
            successful_indexes=[],
            failed_indexes=index_types,
            total_indexes=len(index_types),
            index_results=[]
        )
        
        return workflow_result.to_dict()


@flow(name="update-document-indexes")
def update_document_indexes_flow(document_id: str, index_types: List[str]) -> dict:
    """
    Update indexes for a document using dynamic workflow orchestration.
    
    This flow:
    1. Re-parses the document to get updated content
    2. Dynamically triggers parallel index updates based on parsed content
    3. Aggregates results and provides completion status
    
    Args:
        document_id: Document ID to process
        index_types: List of index types to update
        
    Returns:
        Serialized WorkflowResult
    """
    logger.info(f"Starting update indexes flow for document {document_id} with types: {index_types}")
    
    try:
        # Parse document first
        parsed_data_dict = parse_document_task(document_id)
        
        # Create parallel index update tasks
        index_task_futures = []
        for index_type in index_types:
            future = update_index_task.submit(document_id, index_type, parsed_data_dict)
            index_task_futures.append(future)
        
        # Wait for all index tasks to complete
        index_results = []
        for future in index_task_futures:
            try:
                result = future.result()
                index_results.append(result)
            except Exception as e:
                # Create failure result for this index type
                error_result = IndexTaskResult(
                    status=TaskStatus.FAILED,
                    index_type="unknown",
                    document_id=document_id,
                    success=False,
                    error=str(e)
                )
                index_results.append(error_result.to_dict())
        
        # Aggregate results
        workflow_result = _create_workflow_result(
            document_id, "update", index_types, index_results
        )
        
        logger.info(f"Update indexes flow completed for document {document_id}")
        return workflow_result.to_dict()
        
    except Exception as e:
        error_msg = f"Update indexes flow failed for document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Return failure result
        workflow_result = WorkflowResult(
            workflow_id=f"{document_id}_update_{flow_run.id}",
            document_id=document_id,
            operation="update",
            status=TaskStatus.FAILED,
            message=error_msg,
            successful_indexes=[],
            failed_indexes=index_types,
            total_indexes=len(index_types),
            index_results=[]
        )
        
        return workflow_result.to_dict()


@flow(name="collection-init")
def collection_init_flow(collection_id: str, document_user_quota: int) -> dict:
    """
    Initialize collection flow
    
    Args:
        collection_id: Collection ID to initialize
        document_user_quota: User quota for documents
        
    Returns:
        Serialized task result
    """
    logger.info(f"Starting collection initialization flow for {collection_id}")
    
    try:
        result = collection_init_task(collection_id, document_user_quota)
        logger.info(f"Collection initialization flow completed for {collection_id}")
        return result
        
    except Exception as e:
        error_msg = f"Collection initialization flow failed for {collection_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@flow(name="collection-delete")
def collection_delete_flow(collection_id: str) -> dict:
    """
    Delete collection flow
    
    Args:
        collection_id: Collection ID to delete
        
    Returns:
        Serialized task result
    """
    logger.info(f"Starting collection deletion flow for {collection_id}")
    
    try:
        result = collection_delete_task(collection_id)
        logger.info(f"Collection deletion flow completed for {collection_id}")
        return result
        
    except Exception as e:
        error_msg = f"Collection deletion flow failed for {collection_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


@flow(name="reconcile-indexes")
def reconcile_indexes_flow() -> str:
    """
    Index reconciliation flow
    
    Returns:
        Status message
    """
    logger.info("Starting index reconciliation flow")
    
    try:
        result = reconcile_indexes_task()
        logger.info("Index reconciliation flow completed")
        return result
        
    except Exception as e:
        error_msg = f"Index reconciliation flow failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


# ========== Helper Functions ==========

def _handle_index_success(document_id: str, index_type: str, index_data: dict = None):
    """Handle successful index creation/update"""
    try:
        from aperag.index.reconciler import index_task_callbacks
        import json
        index_data_json = json.dumps(index_data) if index_data else None
        index_task_callbacks.on_index_created(document_id, index_type, index_data_json)
        logger.info(f"Index success callback executed for {index_type} index of document {document_id}")
    except Exception as e:
        logger.warning(f"Failed to execute index success callback for {index_type} of {document_id}: {e}", exc_info=True)


def _handle_index_deletion_success(document_id: str, index_type: str):
    """Handle successful index deletion"""
    try:
        from aperag.index.reconciler import index_task_callbacks
        index_task_callbacks.on_index_deleted(document_id, index_type)
        logger.info(f"Index deletion callback executed for {index_type} index of document {document_id}")
    except Exception as e:
        logger.warning(f"Failed to execute index deletion callback for {index_type} of {document_id}: {e}", exc_info=True)


def _handle_index_failure(document_id: str, index_types: List[str], error_msg: str):
    """Handle index operation failure"""
    try:
        from aperag.index.reconciler import index_task_callbacks
        index_task_callbacks.on_index_failed(document_id, index_types, error_msg)
        logger.info(f"Index failure callback executed for {index_types} indexes of document {document_id}")
    except Exception as e:
        logger.warning(f"Failed to execute index failure callback for {document_id}: {e}", exc_info=True)


def _create_workflow_result(document_id: str, operation: str, index_types: List[str], index_results: List[dict]) -> WorkflowResult:
    """Create workflow result from index task results"""
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
    return WorkflowResult(
        workflow_id=f"{document_id}_{operation}_{flow_run.id}",
        document_id=document_id,
        operation=operation,
        status=status,
        message=status_message,
        successful_indexes=successful_tasks,
        failed_indexes=[f.split(':')[0] for f in failed_tasks],
        total_indexes=len(index_types),
        index_results=[IndexTaskResult.from_dict(r) for r in index_results]
    ) 