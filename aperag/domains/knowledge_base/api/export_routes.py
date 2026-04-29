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

"""Export operations HTTP router (Phase 8 #47 G1 carve, D7 v2 hard-cut).

Carved from legacy ``aperag/views/export.py``. URLs migrated to v2
(``/api/v2/collections/{id}/export`` + ``/api/v2/export-tasks/*``);
mounted at ``/api/v2`` in ``aperag/app.py`` per D7 canonical
(uniform ``/api/v2`` for all backend routes; OpenAI-compat
``/api/v1/embeddings`` is the only remaining v1 allowlist).
"""

import logging

from fastapi import APIRouter, Depends, Request

from aperag.domains.identity.service.auth_dependencies import required_user
from aperag.domains.knowledge_base.ports import AuthenticatedUser
from aperag.domains.knowledge_base.schemas import ExportTaskResponse
from aperag.domains.knowledge_base.service.export_service import export_service

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
    user: AuthenticatedUser = Depends(required_user),
) -> ExportTaskResponse:
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
    user: AuthenticatedUser = Depends(required_user),
) -> ExportTaskResponse:
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
    user: AuthenticatedUser = Depends(required_user),
):
    """Stream the completed export ZIP file to the client."""
    return await export_service.download_export(str(user.id), task_id)
