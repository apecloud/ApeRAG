from ninja import Router
from django.http import HttpRequest
from aperag.utils.request import get_user
from aperag.views.models import Flow
from aperag.views.utils import success, fail
from http import HTTPStatus

router = Router()

from aperag.db.ops import query_bot
import json

@router.get("/bots/{bot_id}/flow", response=Flow)
async def get_flow(request: HttpRequest, bot_id: str):
    user = get_user(request)
    bot = await query_bot(user, bot_id)
    if not bot:
        return fail(HTTPStatus.NOT_FOUND, message="Bot not found")
    try:
        config = json.loads(bot.config or '{}')
        flow = config.get('flow')
        if not flow:
            return fail(HTTPStatus.NOT_FOUND, message="Flow config not found")
        return success(flow)
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, message=str(e))

@router.put("/bots/{bot_id}/flow", response=Flow)
async def update_flow(request: HttpRequest, bot_id: str, data: Flow):
    user = get_user(request)
    bot = await query_bot(user, bot_id)
    if not bot:
        return fail(HTTPStatus.NOT_FOUND, message="Bot not found")
    try:
        config = json.loads(bot.config or '{}')
        config['flow'] = data.dict(exclude_unset=True)
        bot.config = json.dumps(config, ensure_ascii=False)
        await bot.asave()
        return success(data)
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, message=str(e))
