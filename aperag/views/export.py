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

from fastapi import APIRouter, Depends, Request

from aperag.db.models import User
from aperag.schema import view_models
from aperag.service.export_service import export_service
from aperag.views.auth import required_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/collections/{collection_id}/export",
    tags=["export"],
    status_code=202,
    operation_id="create_export_task",
)
async def create_export_task_view(
    request: Request,
    collection_id: str,
    user: User = Depends(required_user),
) -> view_models.ExportTaskResponse:
    """Create an async export task to package all object-store files under the collection."""
    return await export_service.create_export_task(str(user.id), collection_id)


@router.get(
    "/export-tasks/{task_id}",
    tags=["export"],
    operation_id="get_export_task",
)
async def get_export_task_view(
    request: Request,
    task_id: str,
    user: User = Depends(required_user),
) -> view_models.ExportTaskResponse:
    """Query the status and progress of an export task."""
    return await export_service.get_export_task(str(user.id), task_id)


@router.get(
    "/export-tasks/{task_id}/download",
    tags=["export"],
    operation_id="download_export",
)
async def download_export_view(
    request: Request,
    task_id: str,
    user: User = Depends(required_user),
):
    """Stream the completed export ZIP file to the client."""
    return await export_service.download_export(str(user.id), task_id)
