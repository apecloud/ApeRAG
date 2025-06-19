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
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, File, HTTPException, Request, Response, UploadFile, WebSocket

from aperag.db.models import User
from aperag.schema import view_models
from aperag.service.bot_service import bot_service
from aperag.service.chat_service import chat_service_global
from aperag.service.collection_service import collection_service
from aperag.service.document_service import document_service
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

# Import authentication dependencies
from aperag.views.auth import UserManager, authenticate_websocket_user, current_user, get_user_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/prompt-templates")
async def list_prompt_templates_view(
    request: Request, user: User = Depends(current_user)
) -> view_models.PromptTemplateList:
    language = request.headers.get("Lang", "zh-CN")
    return list_prompt_templates(language)


@router.post("/collections")
async def create_collection_view(
    request: Request,
    collection: view_models.CollectionCreate,
    user: User = Depends(current_user),
) -> view_models.Collection:
    return await collection_service.create_collection(str(user.id), collection)


@router.get("/collections")
async def list_collections_view(request: Request, user: User = Depends(current_user)) -> view_models.CollectionList:
    return await collection_service.list_collections(str(user.id))


@router.get("/collections/{collection_id}")
async def get_collection_view(
    request: Request, collection_id: str, user: User = Depends(current_user)
) -> view_models.Collection:
    return await collection_service.get_collection(str(user.id), collection_id)


@router.put("/collections/{collection_id}")
async def update_collection_view(
    request: Request,
    collection_id: str,
    collection: view_models.CollectionUpdate,
    user: User = Depends(current_user),
) -> view_models.Collection:
    return await collection_service.update_collection(str(user.id), collection_id, collection)


@router.delete("/collections/{collection_id}")
async def delete_collection_view(
    request: Request, collection_id: str, user: User = Depends(current_user)
) -> view_models.Collection:
    return await collection_service.delete_collection(str(user.id), collection_id)


@router.post("/collections/{collection_id}/documents")
async def create_documents_view(
    request: Request,
    collection_id: str,
    files: List[UploadFile] = File(...),
    user: User = Depends(current_user),
) -> view_models.DocumentList:
    return await document_service.create_documents(str(user.id), collection_id, files)


@router.get("/collections/{collection_id}/documents")
async def list_documents_view(
    request: Request, collection_id: str, user: User = Depends(current_user)
) -> view_models.DocumentList:
    return await document_service.list_documents(str(user.id), collection_id)


@router.get("/collections/{collection_id}/documents/{document_id}")
async def get_document_view(
    request: Request,
    collection_id: str,
    document_id: str,
    user: User = Depends(current_user),
) -> view_models.Document:
    return await document_service.get_document(str(user.id), collection_id, document_id)


@router.put("/collections/{collection_id}/documents/{document_id}")
async def update_document_view(
    request: Request,
    collection_id: str,
    document_id: str,
    document: view_models.DocumentUpdate,
    user: User = Depends(current_user),
) -> view_models.Document:
    return await document_service.update_document(str(user.id), collection_id, document_id, document)


@router.delete("/collections/{collection_id}/documents/{document_id}")
async def delete_document_view(
    request: Request,
    collection_id: str,
    document_id: str,
    user: User = Depends(current_user),
) -> view_models.Document:
    return await document_service.delete_document(str(user.id), collection_id, document_id)


@router.delete("/collections/{collection_id}/documents")
async def delete_documents_view(
    request: Request,
    collection_id: str,
    document_ids: List[str],
    user: User = Depends(current_user),
):
    return await document_service.delete_documents(str(user.id), collection_id, document_ids)


@router.post("/bots/{bot_id}/chats")
async def create_chat_view(request: Request, bot_id: str, user: User = Depends(current_user)) -> view_models.Chat:
    return await chat_service_global.create_chat(str(user.id), bot_id)


@router.get("/bots/{bot_id}/chats")
async def list_chats_view(request: Request, bot_id: str, user: User = Depends(current_user)) -> view_models.ChatList:
    return await chat_service_global.list_chats(str(user.id), bot_id)


@router.get("/bots/{bot_id}/chats/{chat_id}")
async def get_chat_view(
    request: Request, bot_id: str, chat_id: str, user: User = Depends(current_user)
) -> view_models.ChatDetails:
    return await chat_service_global.get_chat(str(user.id), bot_id, chat_id)


@router.put("/bots/{bot_id}/chats/{chat_id}")
async def update_chat_view(
    request: Request,
    bot_id: str,
    chat_id: str,
    chat_in: view_models.ChatUpdate,
    user: User = Depends(current_user),
) -> view_models.Chat:
    return await chat_service_global.update_chat(str(user.id), bot_id, chat_id, chat_in)


@router.post("/bots/{bot_id}/chats/{chat_id}/messages/{message_id}")
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


@router.delete("/bots/{bot_id}/chats/{chat_id}")
async def delete_chat_view(request: Request, bot_id: str, chat_id: str, user: User = Depends(current_user)):
    await chat_service_global.delete_chat(str(user.id), bot_id, chat_id)
    return Response(status_code=204)


@router.post("/bots")
async def create_bot_view(
    request: Request,
    bot_in: view_models.BotCreate,
    user: User = Depends(current_user),
) -> view_models.Bot:
    return await bot_service.create_bot(str(user.id), bot_in)


@router.get("/bots")
async def list_bots_view(request: Request, user: User = Depends(current_user)) -> view_models.BotList:
    return await bot_service.list_bots(str(user.id))


@router.get("/bots/{bot_id}")
async def get_bot_view(request: Request, bot_id: str, user: User = Depends(current_user)) -> view_models.Bot:
    return await bot_service.get_bot(str(user.id), bot_id)


@router.put("/bots/{bot_id}")
async def update_bot_view(
    request: Request,
    bot_id: str,
    bot_in: view_models.BotUpdate,
    user: User = Depends(current_user),
) -> view_models.Bot:
    return await bot_service.update_bot(str(user.id), bot_id, bot_in)


@router.delete("/bots/{bot_id}")
async def delete_bot_view(request: Request, bot_id: str, user: User = Depends(current_user)):
    await bot_service.delete_bot(str(user.id), bot_id)
    return Response(status_code=204)


@router.post("/available_models")
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


@router.post("/chat/completions/frontend")
async def frontend_chat_completions_view(request: Request, user: User = Depends(current_user)):
    body = await request.body()
    message = body.decode("utf-8")
    query_params = dict(request.query_params)
    stream = query_params.get("stream", "false").lower() == "true"
    bot_id = query_params.get("bot_id", "")
    chat_id = query_params.get("chat_id", "")
    msg_id = request.headers.get("msg_id", "")
    return await chat_service_global.frontend_chat_completions(str(user.id), message, stream, bot_id, chat_id, msg_id)


@router.post("/collections/{collection_id}/searchTests")
async def create_search_test_view(
    request: Request,
    collection_id: str,
    data: view_models.SearchTestRequest,
    user: User = Depends(current_user),
) -> view_models.SearchTestResult:
    return await collection_service.create_search_test(str(user.id), collection_id, data)


@router.delete("/collections/{collection_id}/searchTests/{search_test_id}")
async def delete_search_test_view(
    request: Request,
    collection_id: str,
    search_test_id: str,
    user: User = Depends(current_user),
):
    return await collection_service.delete_search_test(str(user.id), collection_id, search_test_id)


@router.get("/collections/{collection_id}/searchTests")
async def list_search_tests_view(
    request: Request, collection_id: str, user: User = Depends(current_user)
) -> view_models.SearchTestResultList:
    return await collection_service.list_search_tests(str(user.id), collection_id)


@router.post("/bots/{bot_id}/flow/debug")
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


# LLM Configuration API endpoints
@router.get("/llm_configuration")
async def get_llm_configuration_view(request: Request, user: User = Depends(current_user)):
    """Get complete LLM configuration including providers and models"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await get_llm_configuration(str(user.id), is_admin)


@router.post("/llm_providers")
async def create_llm_provider_view(
    request: Request,
    provider_data: view_models.LlmProviderCreateWithApiKey,
    user: User = Depends(current_user),
):
    """Create a new LLM provider with optional API key"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await create_llm_provider(provider_data.model_dump(), str(user.id), is_admin)


@router.get("/llm_providers/{provider_name}")
async def get_llm_provider_view(request: Request, provider_name: str, user: User = Depends(current_user)):
    """Get a specific LLM provider"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await get_llm_provider(provider_name, str(user.id), is_admin)


@router.put("/llm_providers/{provider_name}")
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


@router.delete("/llm_providers/{provider_name}")
async def delete_llm_provider_view(request: Request, provider_name: str, user: User = Depends(current_user)):
    """Delete an LLM provider"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await delete_llm_provider(provider_name, str(user.id), is_admin)


@router.get("/llm_provider_models")
async def list_llm_provider_models_view(
    request: Request, provider_name: str = None, user: User = Depends(current_user)
):
    """List LLM provider models, optionally filtered by provider"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await list_llm_provider_models(provider_name, str(user.id), is_admin)


@router.get("/llm_providers/{provider_name}/models")
async def get_provider_models_view(request: Request, provider_name: str, user: User = Depends(current_user)):
    """Get all models for a specific provider"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await list_llm_provider_models(provider_name=provider_name, user_id=str(user.id), is_admin=is_admin)


@router.post("/llm_providers/{provider_name}/models")
async def create_provider_model_view(request: Request, provider_name: str, user: User = Depends(current_user)):
    """Create a new model for a specific provider"""
    import json

    from aperag.db.models import Role

    body = await request.body()
    data = json.loads(body.decode("utf-8"))
    is_admin = user.role == Role.ADMIN
    return await create_llm_provider_model(provider_name, data, str(user.id), is_admin)


@router.put("/llm_providers/{provider_name}/models/{api}/{model}")
async def update_provider_model_view(
    request: Request, provider_name: str, api: str, model: str, user: User = Depends(current_user)
):
    """Update a specific model"""
    import json

    from aperag.db.models import Role

    body = await request.body()
    data = json.loads(body.decode("utf-8"))
    is_admin = user.role == Role.ADMIN
    return await update_llm_provider_model(provider_name, api, model, data, str(user.id), is_admin)


@router.delete("/llm_providers/{provider_name}/models/{api}/{model}")
async def delete_provider_model_view(
    request: Request, provider_name: str, api: str, model: str, user: User = Depends(current_user)
):
    """Delete a specific model"""
    from aperag.db.models import Role

    is_admin = user.role == Role.ADMIN
    return await delete_llm_provider_model(provider_name, api, model, str(user.id), is_admin)
