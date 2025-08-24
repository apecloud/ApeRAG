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
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, Request, Response, WebSocket

from aperag.db.models import User
from aperag.exceptions import BusinessException
from aperag.schema import view_models
from aperag.service.bot_service import bot_service
from aperag.service.chat_service import chat_service_global
from aperag.service.chat_title_service import chat_title_service
from aperag.service.default_model_service import default_model_service
from aperag.service.flow_service import flow_service_global
from aperag.service.llm_available_model_service import llm_available_model_service
from aperag.service.llm_provider_service import (
    create_llm_provider,
    create_llm_provider_model,
    delete_llm_provider,
    delete_llm_provider_model,
    get_llm_configuration,
    get_llm_provider,
    list_llm_provider_models,
    update_llm_provider,
    update_llm_provider_model,
)
from aperag.service.prompt_template_service import list_prompt_templates
from aperag.utils.audit_decorator import audit

# Import authentication dependencies
from aperag.views.auth import (
    UserManager,
    authenticate_websocket_user,
    current_user,
    get_current_active_user,
    get_user_manager,
)
from aperag.views.quota import router as quota_router

logger = logging.getLogger(__name__)

router = APIRouter()

# Include quota routes
router.include_router(quota_router, tags=["quotas"])


@router.get("/prompt-templates", tags=["templates"])
async def list_prompt_templates_view(
    request: Request, user: User = Depends(current_user)
) -> view_models.PromptTemplateList:
    language = request.headers.get("Lang", "zh-CN")
    return list_prompt_templates(language)


@router.post("/bots/{bot_id}/chats", tags=["chats"])
@audit(resource_type="chat", api_name="CreateChat")
async def create_chat_view(request: Request, bot_id: str, user: User = Depends(current_user)) -> view_models.Chat:
    return await chat_service_global.create_chat(str(user.id), bot_id)


@router.get("/bots/{bot_id}/chats", tags=["chats"])
async def list_chats_view(
    request: Request,
    bot_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    user: User = Depends(current_user),
) -> view_models.ChatList:
    return await chat_service_global.list_chats(str(user.id), bot_id, page, page_size)


@router.get("/bots/{bot_id}/chats/{chat_id}", tags=["chats"])
async def get_chat_view(
    request: Request, bot_id: str, chat_id: str, user: User = Depends(current_user)
) -> view_models.ChatDetails:
    return await chat_service_global.get_chat(str(user.id), bot_id, chat_id)


@router.put("/bots/{bot_id}/chats/{chat_id}", tags=["chats"])
@audit(resource_type="chat", api_name="UpdateChat")
async def update_chat_view(
    request: Request,
    bot_id: str,
    chat_id: str,
    chat_in: view_models.ChatUpdate,
    user: User = Depends(current_user),
) -> view_models.Chat:
    return await chat_service_global.update_chat(str(user.id), bot_id, chat_id, chat_in)


@router.post("/bots/{bot_id}/chats/{chat_id}/messages/{message_id}", tags=["chats"])
@audit(resource_type="message", api_name="FeedbackMessage")
async def feedback_message_view(
    request: Request,
    bot_id: str,
    chat_id: str,
    message_id: str,
    feedback: view_models.Feedback,
    user: User = Depends(current_user),
):
    return await chat_service_global.feedback_message(
        str(user.id), chat_id, message_id, feedback.type, feedback.tag, feedback.message
    )


@router.delete("/bots/{bot_id}/chats/{chat_id}", tags=["chats"])
@audit(resource_type="chat", api_name="DeleteChat")
async def delete_chat_view(request: Request, bot_id: str, chat_id: str, user: User = Depends(current_user)):
    await chat_service_global.delete_chat(str(user.id), bot_id, chat_id)
    return Response(status_code=204)


@router.post("/bots", tags=["bots"])
@audit(resource_type="bot", api_name="CreateBot")
async def create_bot_view(
    request: Request,
    bot_in: view_models.BotCreate,
    user: User = Depends(current_user),
) -> view_models.Bot:
    return await bot_service.create_bot(str(user.id), bot_in)


@router.get("/bots", tags=["bots"])
async def list_bots_view(request: Request, user: User = Depends(current_user)) -> view_models.BotList:
    return await bot_service.list_bots(str(user.id))


@router.get("/bots/{bot_id}", tags=["bots"])
async def get_bot_view(request: Request, bot_id: str, user: User = Depends(current_user)) -> view_models.Bot:
    return await bot_service.get_bot(str(user.id), bot_id)


@router.put("/bots/{bot_id}", tags=["bots"])
@audit(resource_type="bot", api_name="UpdateBot")
async def update_bot_view(
    request: Request,
    bot_id: str,
    bot_in: view_models.BotUpdate,
    user: User = Depends(current_user),
) -> view_models.Bot:
    return await bot_service.update_bot(str(user.id), bot_id, bot_in)


@router.delete("/bots/{bot_id}", tags=["bots"])
@audit(resource_type="bot", api_name="DeleteBot")
async def delete_bot_view(request: Request, bot_id: str, user: User = Depends(current_user)):
    await bot_service.delete_bot(str(user.id), bot_id)
    return Response(status_code=204)


@router.post("/available_models", tags=["llm_models"])
async def get_available_models_view(
    request: Request,
    tag_filter_request: Optional[view_models.TagFilterRequest] = Body(None),
    user: User = Depends(current_user),
) -> view_models.ModelConfigList:
    """Get available models with optional tag filtering"""
    # If no request body provided, create default request
    if tag_filter_request is None:
        tag_filter_request = view_models.TagFilterRequest()

    return await llm_available_model_service.get_available_models(str(user.id), tag_filter_request)


@router.get("/default_models", tags=["default_models"])
async def get_default_models_view(
    request: Request, user: User = Depends(current_user)
) -> view_models.DefaultModelsResponse:
    """Get default model configurations for different scenarios"""
    return await default_model_service.get_default_models(str(user.id))


@router.put("/default_models", tags=["default_models"])
async def update_default_models_view(
    request: Request, update_request: view_models.DefaultModelsUpdateRequest, user: User = Depends(current_user)
) -> view_models.DefaultModelsResponse:
    """Update default model configurations for different scenarios"""
    return await default_model_service.update_default_models(str(user.id), update_request)


@router.post("/chat/completions/frontend", tags=["chats"])
async def frontend_chat_completions_view(request: Request, user: User = Depends(current_user)):
    body = await request.body()
    message = body.decode("utf-8")
    query_params = dict(request.query_params)
    stream = query_params.get("stream", "false").lower() == "true"
    bot_id = query_params.get("bot_id", "")
    chat_id = query_params.get("chat_id", "")
    msg_id = request.headers.get("msg_id", "")
    return await chat_service_global.frontend_chat_completions(str(user.id), message, stream, bot_id, chat_id, msg_id)


@router.post("/bots/{bot_id}/flow/debug", tags=["flows"])
async def debug_flow_stream_view(
    request: Request,
    bot_id: str,
    debug: view_models.DebugFlowRequest,
    user: User = Depends(current_user),
):
    return await flow_service_global.debug_flow_stream(str(user.id), bot_id, debug)


@router.websocket("/bots/{bot_id}/chats/{chat_id}/connect")
async def websocket_chat_endpoint(
    websocket: WebSocket, bot_id: str, chat_id: str, user_manager: UserManager = Depends(get_user_manager)
):
    """WebSocket endpoint for real-time chat with bots

    Supports cookie-based authentication to get user_id
    """
    # Authenticate user from WebSocket cookies
    user_id = await authenticate_websocket_user(websocket, user_manager)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    await chat_service_global.handle_websocket_chat(websocket, user_id, bot_id, chat_id)


@router.post("/bots/{bot_id}/chats/{chat_id}/title", tags=["chats"])
async def generate_chat_title_view(
    bot_id: str,
    chat_id: str,
    request_body: view_models.TitleGenerateRequest = view_models.TitleGenerateRequest(),
    user: User = Depends(get_current_active_user),
) -> view_models.TitleGenerateResponse:
    try:
        title = await chat_title_service.generate_title(
            user_id=str(user.id),
            bot_id=bot_id,
            chat_id=chat_id,
            max_length=request_body.max_length,
            language=request_body.language,
            turns=request_body.turns,
        )
        return {"title": title}
    except BusinessException as be:
        raise HTTPException(status_code=400, detail={"error_code": be.error_code.name, "message": str(be)})


# LLM Configuration API endpoints
@router.get("/llm_configuration", tags=["llm_providers"])
async def get_llm_configuration_view(request: Request, user: User = Depends(current_user)):
    """Get complete LLM configuration including providers and models"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await get_llm_configuration(str(user.id), is_admin)


@router.post("/llm_providers", tags=["llm_providers"])
@audit(resource_type="llm_provider", api_name="CreateLLMProvider")
async def create_llm_provider_view(
    request: Request,
    provider_data: view_models.LlmProviderCreateWithApiKey,
    user: User = Depends(current_user),
):
    """Create a new LLM provider with optional API key"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await create_llm_provider(provider_data.model_dump(), str(user.id), is_admin)


@router.get("/llm_providers/{provider_name}", tags=["llm_providers"])
async def get_llm_provider_view(request: Request, provider_name: str, user: User = Depends(current_user)):
    """Get a specific LLM provider"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await get_llm_provider(provider_name, str(user.id), is_admin)


@router.put("/llm_providers/{provider_name}", tags=["llm_providers"])
@audit(resource_type="llm_provider", api_name="UpdateLLMProvider")
async def update_llm_provider_view(
    request: Request,
    provider_name: str,
    provider_data: view_models.LlmProviderUpdateWithApiKey,
    user: User = Depends(current_user),
):
    """Update an existing LLM provider with optional API key"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await update_llm_provider(provider_name, provider_data.model_dump(), str(user.id), is_admin)


@router.delete("/llm_providers/{provider_name}", tags=["llm_providers"])
@audit(resource_type="llm_provider", api_name="DeleteLLMProvider")
async def delete_llm_provider_view(request: Request, provider_name: str, user: User = Depends(current_user)):
    """Delete an LLM provider"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await delete_llm_provider(provider_name, str(user.id), is_admin)


@router.get("/llm_provider_models", tags=["llm_models"])
async def list_llm_provider_models_view(
    request: Request, provider_name: str = None, user: User = Depends(current_user)
):
    """List LLM provider models, optionally filtered by provider"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await list_llm_provider_models(provider_name, str(user.id), is_admin)


@router.get("/llm_providers/{provider_name}/models", tags=["llm_models"])
async def get_provider_models_view(request: Request, provider_name: str, user: User = Depends(current_user)):
    """Get all models for a specific provider"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await list_llm_provider_models(provider_name=provider_name, user_id=str(user.id), is_admin=is_admin)


@router.post("/llm_providers/{provider_name}/models", tags=["llm_models"])
@audit(resource_type="llm_provider_model", api_name="CreateProviderModel")
async def create_provider_model_view(request: Request, provider_name: str, user: User = Depends(current_user)):
    """Create a new model for a specific provider"""
    import json

    from aperag.db.models import Role

    body = await request.body()
    data = json.loads(body.decode("utf-8"))
    is_admin = user.role == Role.ADMIN
    return await create_llm_provider_model(provider_name, data, str(user.id), is_admin)


@router.put("/llm_providers/{provider_name}/models/{api}/{model:path}", tags=["llm_models"])
@audit(resource_type="llm_provider_model", api_name="UpdateProviderModel")
async def update_provider_model_view(
    request: Request,
    provider_name: str,
    api: str,
    model: str = Path(..., description="Model name (supports names with slashes)"),
    user: User = Depends(current_user),
):
    """Update a specific model"""
    import json

    from aperag.db.models import Role

    body = await request.body()
    data = json.loads(body.decode("utf-8"))
    is_admin = user.role == Role.ADMIN
    return await update_llm_provider_model(provider_name, api, model, data, str(user.id), is_admin)


@router.delete("/llm_providers/{provider_name}/models/{api}/{model:path}", tags=["llm_models"])
@audit(resource_type="llm_provider_model", api_name="DeleteProviderModel")
async def delete_provider_model_view(
    request: Request,
    provider_name: str,
    api: str,
    model: str = Path(..., description="Model name (supports names with slashes)"),
    user: User = Depends(current_user),
):
    """Delete a specific model"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await delete_llm_provider_model(provider_name, api, model, str(user.id), is_admin)
