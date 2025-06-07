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

from fastapi import APIRouter, Depends, File, Request, UploadFile, WebSocket

from aperag.config import settings
from aperag.db.models import User
from aperag.schema import view_models
from aperag.service.bot_service import bot_service
from aperag.service.chat_service import chat_service_global
from aperag.service.collection_service import collection_service
from aperag.service.document_service import document_service
from aperag.service.flow_service import flow_service_global
from aperag.service.model_service import model_service_provider_service
from aperag.service.prompt_template_service import list_prompt_templates

# Import authentication dependencies
from aperag.views.auth import UserManager, current_user, get_jwt_strategy, get_user_manager

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


@router.post("/collections/{collection_id}/urls")
async def create_url_document_view(
    request: Request, collection_id: str, user: User = Depends(current_user)
) -> view_models.DocumentList:
    from aperag.utils.request import get_urls

    urls = get_urls(request)
    return await document_service.create_url_document(str(user.id), collection_id, urls)


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
) -> view_models.Chat:
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
async def delete_chat_view(
    request: Request, bot_id: str, chat_id: str, user: User = Depends(current_user)
) -> view_models.Chat:
    return await chat_service_global.delete_chat(str(user.id), bot_id, chat_id)


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
async def delete_bot_view(request: Request, bot_id: str, user: User = Depends(current_user)) -> view_models.Bot:
    return await bot_service.delete_bot(str(user.id), bot_id)


@router.get("/supported_model_service_providers")
async def list_supported_model_service_providers_view(
    request: Request, user: User = Depends(current_user)
) -> view_models.ModelServiceProviderList:
    return await model_service_provider_service.list_supported_model_service_providers()


@router.get("/model_service_providers")
async def list_model_service_providers_view(
    request: Request, user: User = Depends(current_user)
) -> view_models.ModelServiceProviderList:
    return await model_service_provider_service.list_model_service_providers(str(user.id))


@router.put("/model_service_providers/{provider}")
async def update_model_service_provider_view(
    request: Request,
    provider: str,
    mspIn: view_models.ModelServiceProviderUpdate,
    user: User = Depends(current_user),
):
    from aperag.schema.view_models import ModelConfig

    supported_providers = [ModelConfig(**item) for item in settings.model_configs]
    return await model_service_provider_service.update_model_service_provider(
        str(user.id), provider, mspIn, supported_providers
    )


@router.delete("/model_service_providers/{provider}")
async def delete_model_service_provider_view(request: Request, provider: str, user: User = Depends(current_user)):
    return await model_service_provider_service.delete_model_service_provider(str(user.id), provider)


@router.get("/available_models")
async def list_available_models_view(
    request: Request, user: User = Depends(current_user)
) -> view_models.ModelConfigList:
    return await model_service_provider_service.list_available_models(str(user.id))


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


async def authenticate_websocket_user(websocket: WebSocket, user_manager: UserManager) -> Optional[str]:
    """Authenticate WebSocket connection using session cookie

    Returns:
        str: User ID if authenticated, None otherwise
    """
    try:
        # Extract cookies from WebSocket headers
        cookies_header = None

        # Try different ways to access headers
        if hasattr(websocket, "headers"):
            # WebSocket headers might be a mapping (dict-like)
            if hasattr(websocket.headers, "get"):
                cookie_value = websocket.headers.get("cookie") or websocket.headers.get(b"cookie")
                if cookie_value:
                    cookies_header = cookie_value.decode() if isinstance(cookie_value, bytes) else cookie_value
            else:
                # WebSocket headers might be an iterable of tuples/pairs
                try:
                    for header_item in websocket.headers:
                        if isinstance(header_item, (list, tuple)) and len(header_item) >= 2:
                            name, value = header_item[0], header_item[1]
                            if name == b"cookie" or name == "cookie":
                                cookies_header = value.decode() if isinstance(value, bytes) else value
                                break
                except (TypeError, ValueError):
                    # If iteration fails, headers format is unexpected
                    logger.debug("WebSocket headers format not supported for authentication")
                    pass

        if not cookies_header:
            logger.debug("No cookies found in WebSocket headers")
            return None

        # Parse cookies to find session cookie
        session_token = None
        for cookie in cookies_header.split(";"):
            cookie = cookie.strip()
            if cookie.startswith("session="):
                session_token = cookie.split("=", 1)[1]
                break

        if not session_token:
            logger.debug("No session cookie found")
            return None

        logger.debug(f"Found session token: {session_token[:20]}...")

        # Verify JWT token using the same strategy as HTTP authentication
        jwt_strategy = get_jwt_strategy()

        # Manually decode and verify the JWT token
        try:
            user_data = await jwt_strategy.read_token(session_token, user_manager)
            if user_data:
                logger.debug(f"Successfully authenticated user from WebSocket: {user_data.id}")
                return str(user_data.id)
            else:
                logger.debug("JWT token validation returned no user data")
                return None
        except Exception as e:
            logger.debug(f"WebSocket JWT verification failed: {e}")
            return None

    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        return None

    return None


@router.websocket("/bots/{bot_id}/chats/{chat_id}/connect")
async def websocket_chat_endpoint(
    websocket: WebSocket, bot_id: str, chat_id: str, user_manager: UserManager = Depends(get_user_manager)
):
    """WebSocket endpoint for real-time chat with bots

    Supports cookie-based authentication to get user_id
    """
    # Authenticate user from WebSocket cookies
    user_id = await authenticate_websocket_user(websocket, user_manager)

    if user_id:
        logger.info(f"WebSocket connected with authenticated user: {user_id}")
    else:
        logger.info("WebSocket connected without authentication (anonymous mode)")

    await chat_service_global.handle_websocket_chat(websocket, user_id, bot_id, chat_id)
