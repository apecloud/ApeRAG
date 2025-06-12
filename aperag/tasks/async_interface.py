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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class AsyncTaskInterface(ABC):
    """Abstract interface for async task queue implementations"""
    
    @abstractmethod
    def delay(self, *args, **kwargs) -> Any:
        """Submit task for asynchronous execution"""
        pass
    
    @abstractmethod
    def apply_async(self, args=None, kwargs=None, **options) -> Any:
        """Submit task with options for asynchronous execution"""
        pass


class TaskResult:
    """Standardized task result format"""
    
    def __init__(self, success: bool, data: Any = None, error: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'metadata': self.metadata
        }


class TaskScheduler:
    """Task scheduler interface for different queue implementations"""
    
    def __init__(self, queue_impl: str = 'celery'):
        self.queue_impl = queue_impl
    
    def schedule_collection_init(self, collection_id: str, document_user_quota: int) -> Any:
        """Schedule collection initialization task"""
        if self.queue_impl == 'celery':
            from aperag.tasks.celery_tasks import init_collection_task
            return init_collection_task.delay(collection_id, document_user_quota)
        else:
            raise NotImplementedError(f"Queue implementation {self.queue_impl} not supported")
    
    def schedule_collection_delete(self, collection_id: str) -> Any:
        """Schedule collection deletion task"""
        if self.queue_impl == 'celery':
            from aperag.tasks.celery_tasks import delete_collection_task
            return delete_collection_task.delay(collection_id)
        else:
            raise NotImplementedError(f"Queue implementation {self.queue_impl} not supported")
    
    def schedule_document_index(self, document_id: int) -> Any:
        """Schedule document indexing task"""
        if self.queue_impl == 'celery':
            from aperag.tasks.celery_tasks import add_index_for_document_task
            return add_index_for_document_task.delay(document_id)
        else:
            raise NotImplementedError(f"Queue implementation {self.queue_impl} not supported")
    
    def schedule_document_update(self, document_id: int) -> Any:
        """Schedule document update task"""
        if self.queue_impl == 'celery':
            from aperag.tasks.celery_tasks import update_index_for_document_task
            return update_index_for_document_task.delay(document_id)
        else:
            raise NotImplementedError(f"Queue implementation {self.queue_impl} not supported")
    
    def schedule_document_delete(self, document_id: int) -> Any:
        """Schedule document deletion task"""
        if self.queue_impl == 'celery':
            from aperag.tasks.celery_tasks import remove_index_task
            return remove_index_task.delay(document_id)
        else:
            raise NotImplementedError(f"Queue implementation {self.queue_impl} not supported") 