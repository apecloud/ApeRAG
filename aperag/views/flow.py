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

from typing import Union

from fastapi import APIRouter, Depends, Request

from aperag.db.models import User
from aperag.schema.view_models import WorkflowDefinition
from aperag.service.flow_service import flow_service_global
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user

router = APIRouter()


@router.get("/bots/{bot_id}/flow", tags=["flows"])
async def get_flow_view(
    request: Request, bot_id: str, user: User = Depends(required_user)
) -> Union[WorkflowDefinition, dict]:
    return await flow_service_global.get_flow(str(user.id), bot_id)


@router.put("/bots/{bot_id}/flow", tags=["flows"])
@audit(resource_type="flow", api_name="UpdateFlow")
async def update_flow_view(
    request: Request,
    bot_id: str,
    data: WorkflowDefinition,
    user: User = Depends(required_user),
):
    return await flow_service_global.update_flow(str(user.id), bot_id, data)
