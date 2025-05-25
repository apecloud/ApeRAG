import logging
import json
from django.http import HttpRequest, StreamingHttpResponse
from ninja import Router
from aperag.utils.request import get_user
from aperag.service.chat_completion_service import stream_openai_sse_response, openai_chat_completions
from aperag.chat.sse.openai_consumer import OpenAIFormatter

logger = logging.getLogger(__name__)

router = Router()

@router.post("/chat/completions")
async def openai_chat_completions_view(request: HttpRequest):
    try:
        user = get_user(request)
        body_data = json.loads(request.body.decode("utf-8"))
        query_params = dict(request.GET.items())
        result, error = await openai_chat_completions(user, body_data, query_params)
        if error:
            return StreamingHttpResponse(json.dumps(error), content_type="application/json")
        api_request, formatter, async_generator = result
        if api_request.stream:
            return StreamingHttpResponse(
                stream_openai_sse_response(
                    async_generator(),
                    formatter,
                    api_request.msg_id
                ),
                content_type="text/event-stream",
            )
        else:
            full_content = ""
            async for chunk in async_generator():
                full_content += chunk
            return StreamingHttpResponse(
                json.dumps(formatter.format_complete_response(api_request.msg_id, full_content)),
                content_type="application/json"
            )
    except Exception as e:
        logger.exception(e)
        return StreamingHttpResponse(
            json.dumps(OpenAIFormatter.format_error(str(e))),
            content_type="application/json"
        )


