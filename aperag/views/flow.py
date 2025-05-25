from ninja import Router
from django.http import HttpRequest
from aperag.utils.request import get_user
from aperag.schema.view_models import WorkflowDefinition
from aperag.service.flow_service import get_flow, update_flow

router = Router()

@router.get("/bots/{bot_id}/flow", response=WorkflowDefinition)
async def get_flow_view(request: HttpRequest, bot_id: str):
    user = get_user(request)
    logic = get_flow(user, bot_id)
    flow, error = await logic()
    if error:
        return error
    return flow

@router.put("/bots/{bot_id}/flow", response=WorkflowDefinition)
async def update_flow_view(request: HttpRequest, bot_id: str, data: WorkflowDefinition):
    user = get_user(request)
    logic = update_flow(user, bot_id, data)
    result, error = await logic()
    if error:
        return error
    return result
