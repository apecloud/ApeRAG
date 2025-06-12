import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class TaskResult:
    """Represents the result of a task execution"""
    
    def __init__(self, task_id: str, success: bool = True, error: str = None, data: Any = None):
        self.task_id = task_id
        self.success = success
        self.error = error
        self.data = data


class TaskScheduler(ABC):
    """Abstract base class for task schedulers"""
    
    @abstractmethod
    def schedule_create_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """
        Schedule single index creation task (legacy support)
        
        Args:
            document_id: Document ID to process
            index_types: List of index types (vector, fulltext, graph)
            **kwargs: Additional arguments
            
        Returns:
            Task ID for tracking
        """
        pass
    
    @abstractmethod
    def schedule_update_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """
        Schedule single index update task (legacy support)
        
        Args:
            document_id: Document ID to process
            index_types: List of index types (vector, fulltext, graph)
            **kwargs: Additional arguments
            
        Returns:
            Task ID for tracking
        """
        pass
    
    @abstractmethod
    def schedule_delete_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """
        Schedule single index deletion task (legacy support)
        
        Args:
            document_id: Document ID to process
            index_types: List of index types (vector, fulltext, graph)
            **kwargs: Additional arguments
            
        Returns:
            Task ID for tracking
        """
        pass
    
    @abstractmethod
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """
        Get task execution status
        
        Args:
            task_id: Task ID to check
            
        Returns:
            TaskResult or None if task not found
        """
        pass 


class LocalTaskScheduler(TaskScheduler):
    """Local synchronous implementation for testing or single-machine deployments"""

    def __init__(self):
        self._task_counter = 0
        self._results = {}

    def _execute_task(self, task_func, *args, **kwargs) -> str:
        """Execute task synchronously and store result"""
        self._task_counter += 1
        task_id = f"local_task_{self._task_counter}"

        try:
            result = task_func(*args, **kwargs)
            self._results[task_id] = TaskResult(task_id, success=True, data=result)
        except Exception as e:
            self._results[task_id] = TaskResult(task_id, success=False, error=str(e))

        return task_id

    def schedule_create_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index creation task"""
        from aperag.index.fulltext_index import fulltext_indexer
        from aperag.index.graph_index import graph_indexer
        from aperag.index.operations import cleanup_local_document, get_document_and_collection, parse_document_content
        from aperag.index.vector_index import vector_indexer

        def batch_process():
            # Parse document once
            document, collection = get_document_and_collection(document_id)
            content, doc_parts, local_doc = parse_document_content(document, collection)
            file_path = local_doc.path

            results = {}

            try:
                # Process each requested index type
                for index_type in index_types:
                    if index_type == 'vector':
                        try:
                            result = vector_indexer.create_index(
                                document_id=int(document_id),
                                content=content,
                                doc_parts=doc_parts,
                                collection=collection,
                                file_path=file_path
                            )
                            results['vector'] = {'success': result.success, 'data': result.data}
                        except Exception as e:
                            results['vector'] = {'success': False, 'error': str(e)}

                    elif index_type == 'fulltext':
                        try:
                            result = fulltext_indexer.create_index(
                                document_id=int(document_id),
                                content=content,
                                doc_parts=doc_parts,
                                collection=collection,
                                file_path=file_path
                            )
                            results['fulltext'] = {'success': result.success, 'data': result.data}
                        except Exception as e:
                            results['fulltext'] = {'success': False, 'error': str(e)}

                    elif index_type == 'graph':
                        if graph_indexer.is_enabled(collection):
                            try:
                                from aperag.graph.lightrag_manager import process_document_for_celery
                                result = process_document_for_celery(
                                    collection=collection,
                                    content=content,
                                    doc_id=document_id,
                                    file_path=file_path
                                )
                                results['graph'] = {'success': True, 'data': result}
                            except Exception as e:
                                results['graph'] = {'success': False, 'error': str(e)}
                        else:
                            results['graph'] = {'success': True, 'data': None, 'message': 'Graph indexing disabled'}

            finally:
                # Cleanup local document
                cleanup_local_document(local_doc, collection)

            return results

        return self._execute_task(batch_process)

    def schedule_update_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index update task"""
        # For local scheduler, treat update same as create
        return self.schedule_create_index(document_id, index_types, **kwargs)

    def schedule_delete_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index deletion task"""
        from aperag.index.fulltext_index import fulltext_indexer
        from aperag.index.graph_index import graph_indexer
        from aperag.index.operations import get_document_and_collection
        from aperag.index.vector_index import vector_indexer

        def delete_single_index():
            document, collection = get_document_and_collection(document_id)

            for index_type in index_types:
                if index_type == 'vector':
                    result = vector_indexer.delete_index(int(document_id), collection)
                    if not result.success:
                        raise Exception(result.error)
                elif index_type == 'fulltext':
                    result = fulltext_indexer.delete_index(int(document_id), collection)
                    if not result.success:
                        raise Exception(result.error)
                elif index_type == 'graph':
                    if graph_indexer.is_enabled(collection):
                        from aperag.graph.lightrag_manager import delete_document_for_celery
                        result = delete_document_for_celery(collection=collection, doc_id=document_id)
                        if result.get("status") != "success":
                            raise Exception(result.get("message", "Unknown error"))
                else:
                    raise ValueError(f"Unknown index type: {index_type}")

            return f"Deleted {index_type} index for document {document_id}"

        return self._execute_task(delete_single_index)

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get local task status"""
        return self._results.get(task_id)


class CeleryTaskScheduler(TaskScheduler):
    """Celery implementation of TaskScheduler"""

    def schedule_create_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index creation task using Celery"""
        from aperag.tasks.document_create_index_task import create_index_task

        task = create_index_task.delay(document_id, index_types)
        logger.debug(f"Scheduled create {index_types} index task {task.id} for document {document_id}")
        return task.id

    def schedule_update_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index update task using Celery"""
        from aperag.tasks.document_update_index_task import update_index_task

        task = update_index_task.delay(document_id, index_types)
        logger.debug(f"Scheduled update {index_types} index task {task.id} for document {document_id}")
        return task.id

    def schedule_delete_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index deletion task using Celery"""
        from aperag.tasks.document_delete_index_task import delete_index_task

        task = delete_index_task.delay(document_id, index_types, kwargs.get('index_data'))
        logger.debug(f"Scheduled delete {index_types} index task {task.id} for document {document_id}")
        return task.id

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get Celery task status"""
        try:
            from celery.result import AsyncResult

            result = AsyncResult(task_id)

            if result.state == 'PENDING':
                return TaskResult(task_id, success=False, error="Task pending")
            elif result.state == 'SUCCESS':
                return TaskResult(task_id, success=True, data=result.result)
            elif result.state == 'FAILURE':
                return TaskResult(task_id, success=False, error=str(result.info))
            else:
                return TaskResult(task_id, success=False, error=f"Unknown state: {result.state}")

        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {str(e)}")
            return TaskResult(task_id, success=False, error=str(e))