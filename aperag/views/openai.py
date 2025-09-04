import logging
from aperag.db.models import User
from aperag.service.chat_completion_service import OpenAIFormatter, chat_completion_service
from aperag.views.auth import required_user
from aperag.views.bot import router
from aperag.views.chat import router


from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai"])

@router.post("/chat/completions")
async def openai_chat_completions_view(request: Request, user: User = Depends(required_user)):
    try:
        body_data = await request.json()
        query_params = dict(request.query_params)
        result, error = await chat_completion_service.openai_chat_completions(str(user.id), body_data, query_params)
        if error:
            return error
        api_request, formatter, async_generator = result
        if api_request.stream:
            return StreamingResponse(
                chat_completion_service.stream_openai_sse_response(async_generator(), formatter, api_request.msg_id),
                media_type="text/event-stream",
            )
        else:
            full_content = ""
            async for chunk in async_generator():
                full_content += chunk
            return formatter.format_complete_response(api_request.msg_id, full_content)
    except Exception as e:
        logger.exception(e)
        return OpenAIFormatter.format_error(str(e))