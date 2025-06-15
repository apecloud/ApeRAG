# K8s-Inspired Document Index Management System

This document describes the new, simplified document index management system that replaces the previous complex declarative approach.

## Overview

The new system is inspired by Kubernetes' declarative resource management pattern, using a simple spec/status model with a reconciliation controller.

### Key Principles

1. **Simplicity**: Simple state model without complex locking mechanisms
2. **Declarative**: Declare desired state, controller ensures actual state matches
3. **Self-healing**: Automatically recovers from failures and inconsistencies
4. **Idempotent**: Safe to run multiple times, naturally handles duplicates
5. **Observable**: Clear separation between desired and actual state

## Architecture

### Components

1. **DocumentIndexSpec**: Declares desired state (present/absent)
2. **DocumentIndexStatus**: Tracks actual state (absent/creating/present/deleting/failed)
3. **ReconciliationController**: Ensures actual state matches desired state
4. **IndexSpecManager**: API for managing specifications
5. **SimpleIndexService**: High-level service API

### State Model

#### Desired States (DocumentIndexSpec)
- `present`: Index should exist
- `absent`: Index should not exist

#### Actual States (DocumentIndexStatus)
- `absent`: Index does not exist
- `creating`: Index is being created
- `present`: Index exists and is ready
- `deleting`: Index is being deleted
- `failed`: Index operation failed

### Reconciliation Logic

The reconciliation controller runs every 30 seconds and:

1. Compares desired state (specs) with actual state (statuses)
2. Takes action when states don't match:
   - Desired=present, Actual=absent → Start creating
   - Desired=present, Actual=failed → Retry if possible
   - Desired=absent, Actual=present → Start deleting
3. Handles orphaned statuses (no corresponding spec)
4. Avoids thrashing with cooldown periods

## Usage

### Creating Document Indexes

```python
from aperag.services.simple_index_service import simple_index_service

# Create all index types for a document
result = await simple_index_service.create_document_indexes(
    document_id="doc123",
    user="user@example.com"
)

# Create specific index types
result = await simple_index_service.create_document_indexes(
    document_id="doc123", 
    user="user@example.com",
    index_types=["vector", "fulltext"]
)
```

### Deleting Document Indexes

```python
# Delete all indexes for a document
result = await simple_index_service.delete_document_indexes(
    document_id="doc123"
)

# Delete specific index types
result = await simple_index_service.delete_document_indexes(
    document_id="doc123",
    index_types=["vector"]
)
```

### Checking Index Status

```python
# Get status for a specific document
status = await simple_index_service.get_document_index_status("doc123")

# Get status for all documents in a collection
status = await simple_index_service.get_collection_index_status("col456")

# Get system-wide status
status = await simple_index_service.get_system_index_status()
```

### Retrying Failed Indexes

```python
# Retry failed indexes for a document
result = await simple_index_service.retry_failed_indexes(
    document_id="doc123",
    user="user@example.com"
)

# Force recreate all indexes
result = await simple_index_service.force_recreate_indexes(
    document_id="doc123",
    user="user@example.com"
)
```

### Cleanup (Document Deletion)

```python
# Clean up all specs when document is deleted
result = await simple_index_service.cleanup_document_indexes("doc123")
```

## Database Schema

### DocumentIndexSpec Table
```sql
CREATE TABLE document_index_specs (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(24) NOT NULL,
    index_type documentindextype NOT NULL,
    desired_state indexdesiredstate NOT NULL DEFAULT 'present',
    created_by VARCHAR(256) NOT NULL,
    gmt_created TIMESTAMP WITH TIME ZONE NOT NULL,
    gmt_updated TIMESTAMP WITH TIME ZONE NOT NULL,
    UNIQUE(document_id, index_type)
);
```

### DocumentIndexStatus Table
```sql
CREATE TABLE document_index_statuses (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(24) NOT NULL,
    index_type documentindextype NOT NULL,
    actual_state indexactualstate NOT NULL DEFAULT 'absent',
    index_data TEXT,
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    gmt_created TIMESTAMP WITH TIME ZONE NOT NULL,
    gmt_updated TIMESTAMP WITH TIME ZONE NOT NULL,
    gmt_last_reconciled TIMESTAMP WITH TIME ZONE,
    UNIQUE(document_id, index_type)
);
```

## Celery Tasks

### Reconciliation Task
- **Task**: `reconcile_document_indexes`
- **Schedule**: Every 30 seconds
- **Purpose**: Main reconciliation loop

### Configuration
```python
# In celery beat configuration
CELERY_BEAT_SCHEDULE = {
    'reconcile-document-indexes': {
        'task': 'aperag.tasks.celery_tasks.reconcile_document_indexes',
        'schedule': 30.0,
        'options': {'expires': 25}
    }
}
```

## Migration from Old System

### What Was Removed
- Complex `DocumentIndex` table with locking mechanisms
- State machine with complex event handling
- Declarative manager with batch processing
- Multiple Celery tasks for different operations
- Lock-based concurrency control

### What Was Added
- Simple spec/status tables
- Single reconciliation controller
- One periodic Celery task
- Clean service API
- Self-healing reconciliation logic

### Migration Steps
1. Run database migration to create new tables
2. Update application code to use `SimpleIndexService`
3. Configure Celery Beat with reconciliation task
4. Remove old complex task configurations
5. Monitor reconciliation logs

## Benefits

1. **Reliability**: Self-healing system that recovers from any state
2. **Simplicity**: Easy to understand and debug
3. **Performance**: Single reconciliation loop vs multiple task types
4. **Maintainability**: Clean separation of concerns
5. **Observability**: Clear state tracking and logging

## Monitoring

### Key Metrics to Monitor
- Reconciliation loop execution time
- Number of indexes in each state
- Failed reconciliation attempts
- Retry counts for failed indexes

### Log Messages to Watch
- `"Reconciling {document_id}:{index_type}"` - Active reconciliation
- `"Successfully created index"` - Successful creation
- `"Failed to create index"` - Creation failures
- `"Cleaning up orphaned status"` - Orphan cleanup

## Troubleshooting

### Common Issues

1. **Indexes stuck in 'creating' state**
   - Check reconciliation controller logs
   - Verify indexer implementations
   - Check for resource constraints

2. **High retry counts**
   - Investigate error messages in status table
   - Check external dependencies (vector DB, etc.)
   - Verify document accessibility

3. **Reconciliation not running**
   - Check Celery Beat configuration
   - Verify task registration
   - Check worker availability

### Debug Commands

```python
# Check reconciliation controller status
from aperag.index.reconciliation_controller import reconciliation_controller
await reconciliation_controller.reconcile_all()

# Check specific document status
from aperag.index.reconciliation_controller import index_spec_manager
statuses = await index_spec_manager.get_document_index_status("doc123")
``` 