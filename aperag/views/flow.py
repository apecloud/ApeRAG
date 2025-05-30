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

from django.http import HttpRequest
from ninja import Router

from aperag.schema.view_models import WorkflowDefinition
from aperag.service.flow_service import get_flow, update_flow
from aperag.utils.request import get_user

router = Router()


@router.get("/bots/{bot_id}/flow")
async def get_flow_view(request: HttpRequest, bot_id: str) -> WorkflowDefinition:
    user = get_user(request)
    return await get_flow(user, bot_id)


@router.put("/bots/{bot_id}/flow")
async def update_flow_view(request: HttpRequest, bot_id: str, data: WorkflowDefinition):
    user = get_user(request)
    return await update_flow(user, bot_id, data)
