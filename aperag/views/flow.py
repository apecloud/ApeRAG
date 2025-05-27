from ninja import Router
from django.http import HttpRequest
from aperag.utils.request import get_user
from aperag.schema.view_models import WorkflowDefinition
from aperag.service.flow_service import get_flow, update_flow

router = Router()

@router.get("/bots/{bot_id}/flow")
async def get_flow_view(request: HttpRequest, bot_id: str) -> WorkflowDefinition:
    user = get_user(request)
    return await get_flow(user, bot_id)

@router.put("/bots/{bot_id}/flow")
async def update_flow_view(request: HttpRequest, bot_id: str, data: WorkflowDefinition):
    user = get_user(request)
    return await update_flow(user, bot_id, data)
