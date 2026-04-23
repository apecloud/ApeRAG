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

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from aperag.db.models import User
from aperag.service.chat_completion_service import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIErrorResponse,
    OpenAIFormatter,
    chat_completion_service,
)
from aperag.views.auth import required_user

router = APIRouter(tags=["openai"])


@router.post(
    "/chat/completions",
    response_model=OpenAIChatCompletionResponse | OpenAIErrorResponse,
    responses={
        200: {
            "description": "OpenAI-compatible JSON response or text/event-stream chunks when stream=true",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": "Server-sent OpenAI chat.completion.chunk events.",
                    }
                }
            },
        }
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": OpenAIChatCompletionRequest.model_json_schema(ref_template="#/components/schemas/{model}")
                }
            },
        }
    },
)
async def openai_chat_completions_view(
    request: Request,
    bot_id: str | None = Query(default=None, description="Agent bot id that backs this OpenAI-compatible call"),
    chat_id: str | None = Query(default=None, description="Existing chat id. Omit to create an ephemeral chat"),
    language: str = Query(default="en-US", description="Response language passed through to Agent Runtime"),
    user: User = Depends(required_user),
):
    """OpenAI-compatible chat completions endpoint backed by Agent Runtime V3."""
    try:
        body_data = await request.json()
    except Exception:
        return OpenAIFormatter.format_error("Invalid JSON request body")

    stream, response = await chat_completion_service.openai_chat_completions(
        str(user.id),
        body_data,
        {
            "bot_id": bot_id,
            "chat_id": chat_id,
            "language": language,
        },
        request.headers,
    )
    if stream is not None:
        return StreamingResponse(stream, media_type="text/event-stream")
    return response
