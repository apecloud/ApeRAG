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

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from aperag.db.models import User
from aperag.exceptions import BusinessException
from aperag.schema import view_models
from aperag.service.bot_service import bot_service
from aperag.service.chat_service import chat_service_global
from aperag.service.chat_title_service import chat_title_service
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user

router = APIRouter(tags=["bots-v2"])


@router.post("/bots", response_model=view_models.Bot)
@audit(resource_type="bot", api_name="CreateBotV2")
async def create_bot_v2_view(
    request: Request,
    body: view_models.BotCreate,
    user: User = Depends(required_user),
) -> view_models.Bot:
    """Create an agent bot owned by the current user."""

    return await bot_service.create_bot(str(user.id), body)


@router.get("/bots", response_model=view_models.BotList)
async def list_bots_v2_view(
    user: User = Depends(required_user),
) -> view_models.BotList:
    """List bots owned by the current user."""

    return await bot_service.list_bots(str(user.id))


@router.get("/bots/{bot_id}", response_model=view_models.Bot)
async def get_bot_v2_view(
    bot_id: str,
    user: User = Depends(required_user),
) -> view_models.Bot:
    """Return one bot owned by the current user."""

    return await bot_service.get_bot(str(user.id), bot_id)


@router.put("/bots/{bot_id}", response_model=view_models.Bot)
@audit(resource_type="bot", api_name="UpdateBotV2")
async def update_bot_v2_view(
    request: Request,
    bot_id: str,
    body: view_models.BotUpdateRequest,
    user: User = Depends(required_user),
) -> view_models.Bot:
    """Update one bot. The path bot_id is canonical."""

    bot_update = view_models.BotUpdate(
        title=body.title,
        description=body.description,
        config=body.config,
    )
    return await bot_service.update_bot(str(user.id), bot_id, bot_update)


@router.delete("/bots/{bot_id}", status_code=204)
@audit(resource_type="bot", api_name="DeleteBotV2")
async def delete_bot_v2_view(
    request: Request,
    bot_id: str,
    user: User = Depends(required_user),
) -> Response:
    """Delete one bot owned by the current user. Idempotent."""

    await bot_service.delete_bot(str(user.id), bot_id)
    return Response(status_code=204)


@router.post("/bots/{bot_id}/chats", response_model=view_models.Chat)
@audit(resource_type="chat", api_name="CreateChatV2")
async def create_chat_v2_view(
    request: Request,
    bot_id: str,
    user: User = Depends(required_user),
) -> view_models.Chat:
    """Create a chat under one agent bot."""

    return await chat_service_global.create_chat(str(user.id), bot_id)


@router.get("/bots/{bot_id}/chats", response_model=view_models.ChatList)
async def list_chats_v2_view(
    bot_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    user: User = Depends(required_user),
) -> view_models.ChatList:
    """List chats under one agent bot."""

    return await chat_service_global.list_chats(str(user.id), bot_id, page, page_size)


@router.get("/bots/{bot_id}/chats/{chat_id}", response_model=view_models.ChatDetails)
async def get_chat_v2_view(
    bot_id: str,
    chat_id: str,
    user: User = Depends(required_user),
) -> view_models.ChatDetails:
    """Return one chat with history."""

    return await chat_service_global.get_chat(str(user.id), bot_id, chat_id)


@router.put("/bots/{bot_id}/chats/{chat_id}", response_model=view_models.Chat)
@audit(resource_type="chat", api_name="UpdateChatV2")
async def update_chat_v2_view(
    request: Request,
    bot_id: str,
    chat_id: str,
    body: view_models.ChatUpdate,
    user: User = Depends(required_user),
) -> view_models.Chat:
    """Update one chat. Path bot_id and chat_id are canonical."""

    return await chat_service_global.update_chat(str(user.id), bot_id, chat_id, body)


@router.delete("/bots/{bot_id}/chats/{chat_id}", status_code=204)
@audit(resource_type="chat", api_name="DeleteChatV2")
async def delete_chat_v2_view(
    request: Request,
    bot_id: str,
    chat_id: str,
    user: User = Depends(required_user),
) -> Response:
    """Delete one chat under one agent bot. Idempotent."""

    await chat_service_global.delete_chat(str(user.id), bot_id, chat_id)
    return Response(status_code=204)


@router.post("/bots/{bot_id}/chats/{chat_id}/title", response_model=view_models.TitleGenerateResponse)
async def generate_chat_title_v2_view(
    bot_id: str,
    chat_id: str,
    body: view_models.TitleGenerateRequest = view_models.TitleGenerateRequest(),
    user: User = Depends(required_user),
) -> view_models.TitleGenerateResponse:
    """Generate a title for one chat."""

    try:
        title = await chat_title_service.generate_title(
            user_id=str(user.id),
            bot_id=bot_id,
            chat_id=chat_id,
            max_length=body.max_length,
            language=body.language,
            turns=body.turns,
        )
        return view_models.TitleGenerateResponse(title=title)
    except BusinessException as be:
        raise HTTPException(status_code=400, detail={"error_code": be.error_code.name, "message": str(be)})
