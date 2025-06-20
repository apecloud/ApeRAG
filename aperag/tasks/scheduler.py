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
    def schedule_collection_init(self, collection_id: str, document_user_quota: int, **kwargs) -> str:
        """
        Schedule collection initialization task

        Args:
            collection_id: Collection ID to initialize
            document_user_quota: User quota for documents
            **kwargs: Additional arguments

        Returns:
            Task ID for tracking
        """
        pass

    @abstractmethod
    def schedule_collection_delete(self, collection_id: str, **kwargs) -> str:
        """
        Schedule collection deletion task

        Args:
            collection_id: Collection ID to delete
            **kwargs: Additional arguments

        Returns:
            Task ID for tracking
        """
        pass

    @abstractmethod
    def schedule_reconcile_indexes(self, **kwargs) -> str:
        """
        Schedule index reconciliation task

        Args:
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
                    if index_type == "vector":
                        try:
                            result = vector_indexer.create_index(
                                document_id=document_id,
                                content=content,
                                doc_parts=doc_parts,
                                collection=collection,
                                file_path=file_path,
                            )
                            results["vector"] = {"success": result.success, "data": result.data}
                        except Exception as e:
                            results["vector"] = {"success": False, "error": str(e)}

                    elif index_type == "fulltext":
                        try:
                            result = fulltext_indexer.create_index(
                                document_id=document_id,
                                content=content,
                                doc_parts=doc_parts,
                                collection=collection,
                                file_path=file_path,
                            )
                            results["fulltext"] = {"success": result.success, "data": result.data}
                        except Exception as e:
                            results["fulltext"] = {"success": False, "error": str(e)}

                    elif index_type == "graph":
                        if graph_indexer.is_enabled(collection):
                            try:
                                from aperag.graph.lightrag_manager import process_document_for_celery

                                result = process_document_for_celery(
                                    collection=collection, content=content, doc_id=document_id, file_path=file_path
                                )
                                results["graph"] = {"success": True, "data": result}
                            except Exception as e:
                                results["graph"] = {"success": False, "error": str(e)}
                        else:
                            results["graph"] = {"success": True, "data": None, "message": "Graph indexing disabled"}

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
                if index_type == "vector":
                    result = vector_indexer.delete_index(document_id, collection)
                    if not result.success:
                        raise Exception(result.error)
                elif index_type == "fulltext":
                    result = fulltext_indexer.delete_index(document_id, collection)
                    if not result.success:
                        raise Exception(result.error)
                elif index_type == "graph":
                    if graph_indexer.is_enabled(collection):
                        from aperag.graph.lightrag_manager import delete_document_for_celery

                        result = delete_document_for_celery(collection=collection, doc_id=document_id)
                        if result.get("status") != "success":
                            raise Exception(result.get("message", "Unknown error"))
                else:
                    raise ValueError(f"Unknown index type: {index_type}")

            return f"Deleted {index_type} index for document {document_id}"

        return self._execute_task(delete_single_index)

    def schedule_collection_init(self, collection_id: str, document_user_quota: int, **kwargs) -> str:
        """Schedule collection initialization task"""
        def init_collection():
            from aperag.tasks.collection import collection_task
            result = collection_task.initialize_collection(collection_id, document_user_quota)
            if not result.success:
                raise Exception(result.error)
            return result.to_dict()

        return self._execute_task(init_collection)

    def schedule_collection_delete(self, collection_id: str, **kwargs) -> str:
        """Schedule collection deletion task"""
        def delete_collection():
            from aperag.tasks.collection import collection_task
            result = collection_task.delete_collection(collection_id)
            if not result.success:
                raise Exception(result.error)
            return result.to_dict()

        return self._execute_task(delete_collection)

    def schedule_reconcile_indexes(self, **kwargs) -> str:
        """Schedule index reconciliation task"""
        def reconcile_indexes():
            from aperag.index.reconciler import index_reconciler
            index_reconciler.reconcile_all()
            return "Index reconciliation completed"

        return self._execute_task(reconcile_indexes)

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

    def schedule_collection_init(self, collection_id: str, document_user_quota: int, **kwargs) -> str:
        """Schedule collection initialization task"""
        from config.celery_tasks import collection_init_task
        
        try:
            task_result = collection_init_task.delay(collection_id, document_user_quota)
            task_id = task_result.id
            logger.debug(f"Scheduled collection init task {task_id} for collection {collection_id}")
            return task_id
        except Exception as e:
            logger.error(f"Failed to schedule collection init task for collection {collection_id}: {str(e)}")
            raise

    def schedule_collection_delete(self, collection_id: str, **kwargs) -> str:
        """Schedule collection deletion task"""
        from config.celery_tasks import collection_delete_task
        
        try:
            task_result = collection_delete_task.delay(collection_id)
            task_id = task_result.id
            logger.debug(f"Scheduled collection delete task {task_id} for collection {collection_id}")
            return task_id
        except Exception as e:
            logger.error(f"Failed to schedule collection delete task for collection {collection_id}: {str(e)}")
            raise

    def schedule_reconcile_indexes(self, **kwargs) -> str:
        """Schedule index reconciliation task"""
        from config.celery_tasks import reconcile_indexes_task
        
        try:
            task_result = reconcile_indexes_task.delay()
            task_id = task_result.id
            logger.debug(f"Scheduled reconcile indexes task {task_id}")
            return task_id
        except Exception as e:
            logger.error(f"Failed to schedule reconcile indexes task: {str(e)}")
            raise


class PrefectTaskScheduler(TaskScheduler):
    """Prefect implementation of TaskScheduler - Direct workflow execution"""

    def schedule_create_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index creation workflow"""
        from config.prefect_tasks import create_document_indexes_flow

        try:
            # Execute workflow and return flow run ID
            flow_state = create_document_indexes_flow(document_id, index_types)
            flow_run_id = flow_state.id if hasattr(flow_state, 'id') else str(flow_state)
            logger.debug(
                f"Scheduled create indexes flow {flow_run_id} for document {document_id} with types {index_types}"
            )
            return flow_run_id
        except Exception as e:
            logger.error(f"Failed to schedule create indexes flow for document {document_id}: {str(e)}")
            raise

    def schedule_update_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index update workflow"""
        from config.prefect_tasks import update_document_indexes_flow

        try:
            # Execute workflow and return flow run ID
            flow_state = update_document_indexes_flow(document_id, index_types)
            flow_run_id = flow_state.id if hasattr(flow_state, 'id') else str(flow_state)
            logger.debug(
                f"Scheduled update indexes flow {flow_run_id} for document {document_id} with types {index_types}"
            )
            return flow_run_id
        except Exception as e:
            logger.error(f"Failed to schedule update indexes flow for document {document_id}: {str(e)}")
            raise

    def schedule_delete_index(self, document_id: str, index_types: List[str], **kwargs) -> str:
        """Schedule index deletion workflow"""
        from config.prefect_tasks import delete_document_indexes_flow

        try:
            # Execute workflow and return flow run ID
            flow_state = delete_document_indexes_flow(document_id, index_types)
            flow_run_id = flow_state.id if hasattr(flow_state, 'id') else str(flow_state)
            logger.debug(
                f"Scheduled delete indexes flow {flow_run_id} for document {document_id} with types {index_types}"
            )
            return flow_run_id
        except Exception as e:
            logger.error(f"Failed to schedule delete indexes flow for document {document_id}: {str(e)}")
            raise

    def schedule_collection_init(self, collection_id: str, document_user_quota: int, **kwargs) -> str:
        """Schedule collection initialization workflow"""
        from config.prefect_tasks import collection_init_flow

        try:
            flow_state = collection_init_flow(collection_id, document_user_quota)
            flow_run_id = flow_state.id if hasattr(flow_state, 'id') else str(flow_state)
            logger.debug(f"Scheduled collection init flow {flow_run_id} for collection {collection_id}")
            return flow_run_id
        except Exception as e:
            logger.error(f"Failed to schedule collection init flow for collection {collection_id}: {str(e)}")
            raise

    def schedule_collection_delete(self, collection_id: str, **kwargs) -> str:
        """Schedule collection deletion workflow"""
        from config.prefect_tasks import collection_delete_flow

        try:
            flow_state = collection_delete_flow(collection_id)
            flow_run_id = flow_state.id if hasattr(flow_state, 'id') else str(flow_state)
            logger.debug(f"Scheduled collection delete flow {flow_run_id} for collection {collection_id}")
            return flow_run_id
        except Exception as e:
            logger.error(f"Failed to schedule collection delete flow for collection {collection_id}: {str(e)}")
            raise

    def schedule_reconcile_indexes(self, **kwargs) -> str:
        """Schedule index reconciliation workflow"""
        from config.prefect_tasks import reconcile_indexes_flow

        try:
            flow_state = reconcile_indexes_flow()
            flow_run_id = flow_state.id if hasattr(flow_state, 'id') else str(flow_state)
            logger.debug(f"Scheduled reconcile indexes flow {flow_run_id}")
            return flow_run_id
        except Exception as e:
            logger.error(f"Failed to schedule reconcile indexes flow: {str(e)}")
            raise

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get workflow status using Prefect flow run state (non-blocking)"""
        try:
            from prefect.client.schemas import FlowRun
            from prefect import get_client
            import asyncio

            async def _get_flow_status():
                async with get_client() as client:
                    try:
                        flow_run = await client.read_flow_run(task_id)
                        
                        if flow_run.state.is_pending():
                            return TaskResult(task_id, success=False, error="Flow is pending")
                        elif flow_run.state.is_running():
                            return TaskResult(task_id, success=False, error="Flow is running")
                        elif flow_run.state.is_completed():
                            return TaskResult(task_id, success=True, data=flow_run.state.data)
                        elif flow_run.state.is_failed():
                            return TaskResult(task_id, success=False, error=str(flow_run.state.message))
                        else:
                            return TaskResult(task_id, success=False, error=f"Unknown state: {flow_run.state.type}")
                    except Exception as e:
                        return TaskResult(task_id, success=False, error=f"Failed to read flow run: {str(e)}")

            # Run async function in current event loop or create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, we can't use asyncio.run()
                    # Instead, we'll return a placeholder result
                    return TaskResult(task_id, success=False, error="Status check not available in async context")
                else:
                    return loop.run_until_complete(_get_flow_status())
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(_get_flow_status())

        except Exception as e:
            logger.error(f"Failed to get flow status for {task_id}: {str(e)}")
            return TaskResult(task_id, success=False, error=str(e))


def get_task_scheduler() -> TaskScheduler:
    """
    Factory function to get the appropriate task scheduler based on configuration.
    
    Returns:
        TaskScheduler: The configured task scheduler instance
    """
    try:
        from aperag.config import get_config
        config = get_config()
        scheduler_type = config.task_scheduler_type
        
        logger.debug(f"Creating task scheduler of type: {scheduler_type}")
        return create_task_scheduler(scheduler_type)
    except Exception as e:
        logger.warning(f"Failed to get scheduler type from config, falling back to local: {str(e)}")
        return LocalTaskScheduler()
