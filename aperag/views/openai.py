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

from fastapi import APIRouter, Depends, Request

from aperag.db.models import User
from aperag.service.chat_completion_service import OpenAIFormatter
from aperag.views.auth import required_user

router = APIRouter(tags=["openai"])


@router.post("/chat/completions")
async def openai_chat_completions_view(request: Request, user: User = Depends(required_user)):
    """OpenAI-compatible chat completions endpoint - Not implemented for agent-type bots"""
    return OpenAIFormatter.format_error(
        "The /v1/chat/completions endpoint is not implemented. Please use WebSocket API for agent-type bots."
    )
