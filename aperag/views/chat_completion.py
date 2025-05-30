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

import json
import logging

from django.http import HttpRequest, StreamingHttpResponse
from ninja import Router

from aperag.chat.sse.openai_consumer import OpenAIFormatter
from aperag.service.chat_completion_service import openai_chat_completions, stream_openai_sse_response
from aperag.utils.request import get_user

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
                stream_openai_sse_response(async_generator(), formatter, api_request.msg_id),
                content_type="text/event-stream",
            )
        else:
            full_content = ""
            async for chunk in async_generator():
                full_content += chunk
            return StreamingHttpResponse(
                json.dumps(formatter.format_complete_response(api_request.msg_id, full_content)),
                content_type="application/json",
            )
    except Exception as e:
        logger.exception(e)
        return StreamingHttpResponse(json.dumps(OpenAIFormatter.format_error(str(e))), content_type="application/json")
