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
Celery tasks entry points for non-index operations
Index operations are now handled by the K8s-inspired declarative system
"""

import logging
from typing import Any

from aperag.workflow.collection_workflow import collection_workflow
from config.celery import app

logger = logging.getLogger(__name__)


# Configuration constants
class TaskConfig:
    RETRY_COUNTDOWN_COLLECTION = 60
    RETRY_MAX_RETRIES_COLLECTION = 2


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


# Document index operations are now handled by the K8s-inspired declarative system
# See aperag/tasks/index_tasks.py for the new implementation 