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

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from aperag.tasks.utils import cleanup_local_document, parse_document_content

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


def create_task_scheduler(scheduler_type: str):
    if scheduler_type == "local":
        return LocalTaskScheduler()
    elif scheduler_type == "celery":
        return CeleryTaskScheduler()
    elif scheduler_type == "prefect":
        return PrefectTaskScheduler()
    else:
        raise Exception("unknown task scheduler type: %s" % scheduler_type)


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
        from aperag.index.vector_index import vector_indexer
        from aperag.tasks.utils import get_document_and_collection

        def batch_process():
            # Parse document once
            document, collection = get_document_and_collection(document_id)
            content, doc_parts, local_doc = parse_document_content(document, collection)
            file_path = local_doc.path

            results = {}

            try:
                # Process each requested index type
                for index_type in index_types:
                    if index_type == "VECTOR":
                        try:
                            result = vector_indexer.create_index(
                                document_id=document_id,
                                content=content,
                                doc_parts=doc_parts,
                                collection=collection,
                                file_path=file_path,
                            )
                            results["VECTOR"] = {"success": result.success, "data": result.data}
                        except Exception as e:
                            results["VECTOR"] = {"success": False, "error": str(e)}

                    elif index_type == "FULLTEXT":
                        try:
                            result = fulltext_indexer.create_index(
                                document_id=document_id,
                                content=content,
                                doc_parts=doc_parts,
                                collection=collection,
                                file_path=file_path,
                            )
                            results["FULLTEXT"] = {"success": result.success, "data": result.data}
                        except Exception as e:
                            results["FULLTEXT"] = {"success": False, "error": str(e)}

                    elif index_type == "GRAPH":
                        if graph_indexer.is_enabled(collection):
                            try:
                                from aperag.graph.lightrag_manager import process_document_for_celery

                                result = process_document_for_celery(
                                    collection=collection, content=content, doc_id=document_id, file_path=file_path
                                )
                                results["GRAPH"] = {"success": True, "data": result}
                            except Exception as e:
                                results["GRAPH"] = {"success": False, "error": str(e)}
                        else:
                            results["GRAPH"] = {"success": True, "data": None, "message": "Graph indexing disabled"}

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
        from aperag.index.vector_index import vector_indexer
        from aperag.tasks.utils import get_document_and_collection

        def delete_single_index():
            document, collection = get_document_and_collection(document_id)

            for index_type in index_types:
                if index_type == "VECTOR":
                    result = vector_indexer.delete_index(document_id, collection)
                    if not result.success:
                        raise Exception(result.error)
                elif index_type == "FULLTEXT":
                    result = fulltext_indexer.delete_index(document_id, collection)
                    if not result.success:
                        raise Exception(result.error)
                elif index_type == "GRAPH":
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
    """Celery implementation of TaskScheduler - Direct workflow execution"""

    def schedule_create_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index creation workflow"""
        from config.celery_tasks import create_document_indexes_workflow

        try:
            # Execute workflow and return AsyncResult ID (not calling .get())
            workflow_result = create_document_indexes_workflow(document_id, index_types)
            workflow_id = workflow_result.id  # Use .id instead of .get('workflow_id')
            logger.debug(
                f"Scheduled create indexes workflow {workflow_id} for document {document_id} with types {index_types}"
            )
            return workflow_id
        except Exception as e:
            logger.error(f"Failed to schedule create indexes workflow for document {document_id}: {str(e)}")
            raise

    def schedule_update_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index update workflow"""
        from config.celery_tasks import update_document_indexes_workflow

        try:
            # Execute workflow and return AsyncResult ID (not calling .get())
            workflow_result = update_document_indexes_workflow(document_id, index_types)
            workflow_id = workflow_result.id  # Use .id instead of .get('workflow_id')
            logger.debug(
                f"Scheduled update indexes workflow {workflow_id} for document {document_id} with types {index_types}"
            )
            return workflow_id
        except Exception as e:
            logger.error(f"Failed to schedule update indexes workflow for document {document_id}: {str(e)}")
            raise

    def schedule_delete_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index deletion workflow"""
        from config.celery_tasks import delete_document_indexes_workflow

        try:
            # Execute workflow and return AsyncResult ID (not calling .get())
            workflow_result = delete_document_indexes_workflow(document_id, index_types)
            workflow_id = workflow_result.id  # Use .id instead of .get('workflow_id')
            logger.debug(
                f"Scheduled delete indexes workflow {workflow_id} for document {document_id} with types {index_types}"
            )
            return workflow_id
        except Exception as e:
            logger.error(f"Failed to schedule delete indexes workflow for document {document_id}: {str(e)}")
            raise

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get workflow status using Celery AsyncResult (non-blocking)"""
        try:
            from celery.result import AsyncResult

            from config.celery import app

            # Get AsyncResult without calling .get()
            workflow_result = AsyncResult(task_id, app=app)

            # Check status without blocking
            if workflow_result.state == "PENDING":
                return TaskResult(task_id, success=False, error="Workflow is pending")
            elif workflow_result.state == "STARTED":
                return TaskResult(task_id, success=False, error="Workflow is running")
            elif workflow_result.state == "SUCCESS":
                return TaskResult(task_id, success=True, data=workflow_result.result)
            elif workflow_result.state == "FAILURE":
                return TaskResult(task_id, success=False, error=str(workflow_result.info))
            else:
                return TaskResult(task_id, success=False, error=f"Unknown state: {workflow_result.state}")

        except Exception as e:
            logger.error(f"Failed to get workflow status for {task_id}: {str(e)}")
            return TaskResult(task_id, success=False, error=str(e))


class PrefectTaskScheduler(TaskScheduler):
    """Prefect implementation of TaskScheduler - Direct workflow execution"""

    def schedule_create_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index creation workflow"""
        raise NotImplementedError("Prefect task scheduler is not implemented")

    def schedule_update_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index update workflow"""
        raise NotImplementedError("Prefect task scheduler is not implemented")

    def schedule_delete_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index deletion workflow"""
        raise NotImplementedError("Prefect task scheduler is not implemented")

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get workflow status using Prefect AsyncResult (non-blocking)"""
        raise NotImplementedError("Prefect task scheduler is not implemented")
