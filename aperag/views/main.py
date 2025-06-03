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
from typing import List
from urllib.parse import parse_qsl

from fastapi import APIRouter, Depends, File, Request, UploadFile

from aperag.chat.message import feedback_message
from aperag.config import SessionDep, settings
from aperag.db.models import User
from aperag.schema import view_models
from aperag.service import (
    bot_service,
    chat_service,
    collection_service,
)
from aperag.service.bot_service import create_bot
from aperag.service.chat_service import create_chat, frontend_chat_completions
from aperag.service.collection_service import (
    create_collection,
    create_search_test,
    delete_search_test,
    list_search_tests,
)
from aperag.service.document_service import (
    create_documents,
    create_url_document,
    delete_document,
    delete_documents,
    get_document,
    list_documents,
    update_document,
)
from aperag.service.flow_service import debug_flow_stream
from aperag.service.model_service import (
    delete_model_service_provider,
    list_available_models,
    list_model_service_providers,
    list_supported_model_service_providers,
    update_model_service_provider,
)
from aperag.service.prompt_template_service import list_prompt_templates
from aperag.utils.request import get_urls

# Import authentication dependencies
from aperag.views.auth import get_current_user_with_state
from aperag.views.utils import success

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/prompt-templates")
async def list_prompt_templates_view(
    request: Request, user: User = Depends(get_current_user_with_state)
) -> view_models.PromptTemplateList:
    language = request.headers.get("Lang", "zh-CN")
    return list_prompt_templates(language)


@router.post("/collections")
async def create_collection_view(
    session: SessionDep,
    request: Request,
    collection: view_models.CollectionCreate,
    user: User = Depends(get_current_user_with_state),
) -> view_models.Collection:
    return await create_collection(session, str(user.id), collection)


@router.get("/collections")
async def list_collections_view(
    session: SessionDep, request: Request, user: User = Depends(get_current_user_with_state)
) -> view_models.CollectionList:
    return await collection_service.list_collections(session, str(user.id))


@router.get("/collections/{collection_id}")
async def get_collection_view(
    request: Request, collection_id: str, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.Collection:
    return await collection_service.get_collection(session, str(user.id), collection_id)


@router.put("/collections/{collection_id}")
async def update_collection_view(
    request: Request,
    collection_id: str,
    collection: view_models.CollectionUpdate,
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
) -> view_models.Collection:
    return await collection_service.update_collection(session, str(user.id), collection_id, collection)


@router.delete("/collections/{collection_id}")
async def delete_collection_view(
    request: Request, collection_id: str, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.Collection:
    return await collection_service.delete_collection(session, str(user.id), collection_id)


@router.post("/collections/{collection_id}/documents")
async def create_documents_view(
    request: Request,
    collection_id: str,
    session: SessionDep,
    files: List[UploadFile] = File(...),
    user: User = Depends(get_current_user_with_state),
) -> List[view_models.Document]:
    return await create_documents(session, str(user.id), collection_id, files)


@router.post("/collections/{collection_id}/urls")
async def create_url_document_view(
    request: Request, collection_id: str, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> List[view_models.Document]:
    urls = get_urls(request)
    return await create_url_document(session, str(user.id), collection_id, urls)


@router.get("/collections/{collection_id}/documents")
async def list_documents_view(
    request: Request, collection_id: str, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.DocumentList:
    return await list_documents(session, str(user.id), collection_id)


@router.get("/collections/{collection_id}/documents/{document_id}")
async def get_document_view(
    request: Request,
    collection_id: str,
    document_id: str,
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
) -> view_models.Document:
    return await get_document(session, str(user.id), collection_id, document_id)


@router.put("/collections/{collection_id}/documents/{document_id}")
async def update_document_view(
    request: Request,
    collection_id: str,
    document_id: str,
    document: view_models.Document,
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
) -> view_models.Document:
    return await update_document(session, str(user.id), collection_id, document_id, document)


@router.delete("/collections/{collection_id}/documents/{document_id}")
async def delete_document_view(
    request: Request,
    collection_id: str,
    document_id: str,
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
) -> view_models.Document:
    return await delete_document(session, str(user.id), collection_id, document_id)


@router.delete("/collections/{collection_id}/documents")
async def delete_documents_view(
    request: Request,
    collection_id: str,
    document_ids: List[str],
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
):
    return await delete_documents(session, str(user.id), collection_id, document_ids)


@router.post("/bots/{bot_id}/chats")
async def create_chat_view(
    request: Request, bot_id: str, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.Chat:
    return await create_chat(session, str(user.id), bot_id)


@router.get("/bots/{bot_id}/chats")
async def list_chats_view(
    request: Request, bot_id: str, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.ChatList:
    return await chat_service.list_chats(session, str(user.id), bot_id)


@router.get("/bots/{bot_id}/chats/{chat_id}")
async def get_chat_view(
    request: Request, bot_id: str, chat_id: str, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.Chat:
    return await chat_service.get_chat(session, str(user.id), bot_id, chat_id)


@router.put("/bots/{bot_id}/chats/{chat_id}")
async def update_chat_view(
    request: Request,
    bot_id: str,
    chat_id: str,
    chat_in: view_models.ChatUpdate,
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
) -> view_models.Chat:
    return await chat_service.update_chat(session, str(user.id), bot_id, chat_id, chat_in)


@router.post("/bots/{bot_id}/chats/{chat_id}/messages/{message_id}")
async def feedback_message_view(
    request: Request,
    bot_id: str,
    chat_id: str,
    message_id: str,
    feedback: view_models.Feedback,
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
):
    await feedback_message(session, str(user.id), chat_id, message_id, feedback.type, feedback.tag, feedback.message)
    return success({})


@router.delete("/bots/{bot_id}/chats/{chat_id}")
async def delete_chat_view(
    request: Request, bot_id: str, chat_id: str, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.Chat:
    return await chat_service.delete_chat(session, str(user.id), bot_id, chat_id)


@router.post("/bots")
async def create_bot_view(
    request: Request,
    bot_in: view_models.BotCreate,
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
) -> view_models.Bot:
    return await create_bot(session, str(user.id), bot_in)


@router.get("/bots")
async def list_bots_view(
    request: Request, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.BotList:
    return await bot_service.list_bots(session, str(user.id))


@router.get("/bots/{bot_id}")
async def get_bot_view(
    request: Request, bot_id: str, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.Bot:
    return await bot_service.get_bot(session, str(user.id), bot_id)


@router.put("/bots/{bot_id}")
async def update_bot_view(
    request: Request,
    bot_id: str,
    bot_in: view_models.BotUpdate,
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
) -> view_models.Bot:
    return await bot_service.update_bot(session, str(user.id), bot_id, bot_in)


@router.delete("/bots/{bot_id}")
async def delete_bot_view(
    request: Request, bot_id: str, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.Bot:
    return await bot_service.delete_bot(session, str(user.id), bot_id)


@router.get("/supported_model_service_providers")
async def list_supported_model_service_providers_view(
    request: Request, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.ModelServiceProviderList:
    return await list_supported_model_service_providers()


@router.get("/model_service_providers")
async def list_model_service_providers_view(
    request: Request, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.ModelServiceProviderList:
    return await list_model_service_providers(session, str(user.id))


@router.put("/model_service_providers/{provider}")
async def update_model_service_provider_view(
    request: Request,
    provider: str,
    mspIn: view_models.ModelServiceProviderUpdate,
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
):
    from aperag.schema.view_models import ModelConfig

    supported_providers = [ModelConfig(**item) for item in settings.model_configs]
    return await update_model_service_provider(session, str(user.id), provider, mspIn, supported_providers)


@router.delete("/model_service_providers/{provider}")
async def delete_model_service_provider_view(
    request: Request, provider: str, session: SessionDep, user: User = Depends(get_current_user_with_state)
):
    return await delete_model_service_provider(session, str(user.id), provider)


@router.get("/available_models")
async def list_available_models_view(
    request: Request, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.ModelConfigList:
    return await list_available_models(session, str(user.id))


@router.post("/chat/completions/frontend")
async def frontend_chat_completions_view(
    request: Request, session: SessionDep, user: User = Depends(get_current_user_with_state)
):
    message = request.body.decode("utf-8")
    query_params = dict(parse_qsl(request.GET.urlencode()))
    stream = query_params.get("stream", "false").lower() == "true"
    bot_id = query_params.get("bot_id", "")
    chat_id = query_params.get("chat_id", "")
    msg_id = request.headers.get("msg_id", "")
    return await frontend_chat_completions(session, str(user.id), message, stream, bot_id, chat_id, msg_id)


@router.post("/collections/{collection_id}/searchTests")
async def create_search_test_view(
    request: Request,
    collection_id: str,
    data: view_models.SearchTestRequest,
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
) -> view_models.SearchTestResult:
    return await create_search_test(session, str(user.id), collection_id, data)


@router.delete("/collections/{collection_id}/searchTests/{search_test_id}")
async def delete_search_test_view(
    request: Request,
    collection_id: str,
    search_test_id: str,
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
):
    return await delete_search_test(session, str(user.id), collection_id, search_test_id)


@router.get("/collections/{collection_id}/searchTests")
async def list_search_tests_view(
    request: Request, collection_id: str, session: SessionDep, user: User = Depends(get_current_user_with_state)
) -> view_models.SearchTestResultList:
    return await list_search_tests(session, str(user.id), collection_id)


@router.post("/bots/{bot_id}/flow/debug")
async def debug_flow_stream_view(
    request: Request,
    bot_id: str,
    debug: view_models.DebugFlowRequest,
    session: SessionDep,
    user: User = Depends(get_current_user_with_state),
):
    return await debug_flow_stream(session, str(user.id), bot_id, debug)
