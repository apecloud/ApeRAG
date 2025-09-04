import logging
from aperag.db.models import User
from aperag.schema import view_models
from aperag.service.bot_service import bot_service
from aperag.service.flow_service import flow_service_global
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user
from fastapi import APIRouter, Depends, Request, Response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["bots"])


@router.post("/bots")
@audit(resource_type="bot", api_name="CreateBot")
async def create_bot_view(
    request: Request,
    bot_in: view_models.BotCreate,
    user: User = Depends(required_user),
) -> view_models.Bot:
    return await bot_service.create_bot(str(user.id), bot_in)


@router.get("/bots")
async def list_bots_view(request: Request, user: User = Depends(required_user)) -> view_models.BotList:
    return await bot_service.list_bots(str(user.id))


@router.get("/bots/{bot_id}")
async def get_bot_view(request: Request, bot_id: str, user: User = Depends(required_user)) -> view_models.Bot:
    return await bot_service.get_bot(str(user.id), bot_id)


@router.put("/bots/{bot_id}")
@audit(resource_type="bot", api_name="UpdateBot")
async def update_bot_view(
    request: Request,
    bot_id: str,
    bot_in: view_models.BotUpdate,
    user: User = Depends(required_user),
) -> view_models.Bot:
    return await bot_service.update_bot(str(user.id), bot_id, bot_in)


@router.delete("/bots/{bot_id}")
@audit(resource_type="bot", api_name="DeleteBot")
async def delete_bot_view(request: Request, bot_id: str, user: User = Depends(required_user)):
    await bot_service.delete_bot(str(user.id), bot_id)
    return Response(status_code=204)


@router.post("/bots/{bot_id}/flow/debug", tags=["flows"])
async def debug_flow_stream_view(
    request: Request,
    bot_id: str,
    debug: view_models.DebugFlowRequest,
    user: User = Depends(required_user),
):
    return await flow_service_global.debug_flow_stream(str(user.id), bot_id, debug)