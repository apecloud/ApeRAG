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

import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from aperag.db.models import User
from aperag.service.chat_completion_service import OpenAIFormatter, chat_completion_service
from aperag.views.auth import current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat/completions", tags=["chats"])
async def openai_chat_completions_view(request: Request, user: User = Depends(current_user)):
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
