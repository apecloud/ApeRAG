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

from fastapi import APIRouter, File, Request, UploadFile

from aperag.chat.message import feedback_message
from aperag.config import SessionDep, settings
from aperag.db.ops import build_pq
from aperag.schema import view_models
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
from aperag.utils.request import get_urls, get_user
from aperag.views.utils import success

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/prompt-templates")
async def list_prompt_templates_view(request) -> view_models.PromptTemplateList:
    language = request.headers.get("Lang", "zh-CN")
    return list_prompt_templates(language)


@router.post("/collections")
async def create_collection_view(
    request, collection: view_models.CollectionCreate, session: SessionDep
) -> view_models.Collection:
    user = get_user(request)
    return await create_collection(session, user, collection)


@router.get("/collections")
async def list_collections_view(request, session: SessionDep) -> view_models.CollectionList:
    from aperag.service.collection_service import list_collections

    user = get_user(request)
    return await list_collections(session, user, build_pq(request))


@router.get("/collections/{collection_id}")
async def get_collection_view(request, collection_id: str, session: SessionDep) -> view_models.Collection:
    from aperag.service.collection_service import get_collection

    user = get_user(request)
    return await get_collection(session, user, collection_id)


@router.put("/collections/{collection_id}")
async def update_collection_view(
    request, collection_id: str, collection: view_models.CollectionUpdate, session: SessionDep
) -> view_models.Collection:
    from aperag.service.collection_service import update_collection

    user = get_user(request)
    return await update_collection(session, user, collection_id, collection)


@router.delete("/collections/{collection_id}")
async def delete_collection_view(request, collection_id: str, session: SessionDep) -> view_models.Collection:
    from aperag.service.collection_service import delete_collection

    user = get_user(request)
    return await delete_collection(session, user, collection_id)


@router.post("/collections/{collection_id}/documents")
async def create_documents_view(
    request: Request, collection_id: str, session: SessionDep, files: List[UploadFile] = File(...)
) -> List[view_models.Document]:
    user = get_user(request)
    return await create_documents(session, user, collection_id, files)


@router.post("/collections/{collection_id}/urls")
async def create_url_document_view(request, collection_id: str, session: SessionDep) -> List[view_models.Document]:
    user = get_user(request)
    urls = get_urls(request)
    return await create_url_document(session, user, collection_id, urls)


@router.get("/collections/{collection_id}/documents")
async def list_documents_view(request, collection_id: str, session: SessionDep) -> view_models.DocumentList:
    user = get_user(request)
    return await list_documents(session, user, collection_id, build_pq(request))


@router.get("/collections/{collection_id}/documents/{document_id}")
async def get_document_view(request, collection_id: str, document_id: str, session: SessionDep) -> view_models.Document:
    user = get_user(request)
    return await get_document(session, user, collection_id, document_id)


@router.put("/collections/{collection_id}/documents/{document_id}")
async def update_document_view(
    request, collection_id: str, document_id: str, document: view_models.Document, session: SessionDep
) -> view_models.Document:
    user = get_user(request)
    return await update_document(session, user, collection_id, document_id, document)


@router.delete("/collections/{collection_id}/documents/{document_id}")
async def delete_document_view(
    request, collection_id: str, document_id: str, session: SessionDep
) -> view_models.Document:
    user = get_user(request)
    return await delete_document(session, user, collection_id, document_id)


@router.delete("/collections/{collection_id}/documents")
async def delete_documents_view(request, collection_id: str, document_ids: List[str], session: SessionDep):
    user = get_user(request)
    return await delete_documents(session, user, collection_id, document_ids)


@router.post("/bots/{bot_id}/chats")
async def create_chat_view(request, bot_id: str, session: SessionDep) -> view_models.Chat:
    user = get_user(request)
    return await create_chat(session, user, bot_id)


@router.get("/bots/{bot_id}/chats")
async def list_chats_view(request, bot_id: str, session: SessionDep) -> view_models.ChatList:
    from aperag.service.chat_service import list_chats

    user = get_user(request)
    return await list_chats(session, user, bot_id, build_pq(request))


@router.get("/bots/{bot_id}/chats/{chat_id}")
async def get_chat_view(request, bot_id: str, chat_id: str, session: SessionDep) -> view_models.Chat:
    from aperag.service.chat_service import get_chat

    user = get_user(request)
    return await get_chat(session, user, bot_id, chat_id)


@router.put("/bots/{bot_id}/chats/{chat_id}")
async def update_chat_view(
    request, bot_id: str, chat_id: str, chat_in: view_models.ChatUpdate, session: SessionDep
) -> view_models.Chat:
    from aperag.service.chat_service import update_chat

    user = get_user(request)
    return await update_chat(session, user, bot_id, chat_id, chat_in)


@router.post("/bots/{bot_id}/chats/{chat_id}/messages/{message_id}")
async def feedback_message_view(
    request, bot_id: str, chat_id: str, message_id: str, feedback: view_models.Feedback, session: SessionDep
):
    user = get_user(request)
    await feedback_message(session, user, chat_id, message_id, feedback.type, feedback.tag, feedback.message)
    return success({})


@router.delete("/bots/{bot_id}/chats/{chat_id}")
async def delete_chat_view(request, bot_id: str, chat_id: str, session: SessionDep) -> view_models.Chat:
    from aperag.service.chat_service import delete_chat

    user = get_user(request)
    return await delete_chat(session, user, bot_id, chat_id)


@router.post("/bots")
async def create_bot_view(request, bot_in: view_models.BotCreate, session: SessionDep) -> view_models.Bot:
    user = get_user(request)
    return await create_bot(session, user, bot_in)


@router.get("/bots")
async def list_bots_view(request, session: SessionDep) -> view_models.BotList:
    from aperag.service.bot_service import list_bots

    user = get_user(request)
    return await list_bots(session, user, build_pq(request))


@router.get("/bots/{bot_id}")
async def get_bot_view(request, bot_id: str, session: SessionDep) -> view_models.Bot:
    from aperag.service.bot_service import get_bot

    user = get_user(request)
    return await get_bot(session, user, bot_id)


@router.put("/bots/{bot_id}")
async def update_bot_view(request, bot_id: str, bot_in: view_models.BotUpdate, session: SessionDep) -> view_models.Bot:
    from aperag.service.bot_service import update_bot

    user = get_user(request)
    return await update_bot(session, user, bot_id, bot_in)


@router.delete("/bots/{bot_id}")
async def delete_bot_view(request, bot_id: str, session: SessionDep) -> view_models.Bot:
    from aperag.service.bot_service import delete_bot

    user = get_user(request)
    return await delete_bot(session, user, bot_id)


@router.get("/supported_model_service_providers")
async def list_supported_model_service_providers_view(
    request, session: SessionDep
) -> view_models.ModelServiceProviderList:
    return await list_supported_model_service_providers()


@router.get("/model_service_providers")
async def list_model_service_providers_view(request, session: SessionDep) -> view_models.ModelServiceProviderList:
    user = get_user(request)
    return await list_model_service_providers(session, user)


@router.put("/model_service_providers/{provider}")
async def update_model_service_provider_view(
    request, provider: str, mspIn: view_models.ModelServiceProviderUpdate, session: SessionDep
):
    from aperag.schema.view_models import ModelConfig

    user = get_user(request)
    supported_providers = [ModelConfig(**item) for item in settings.model_configs]
    return await update_model_service_provider(session, user, provider, mspIn, supported_providers)


@router.delete("/model_service_providers/{provider}")
async def delete_model_service_provider_view(request, provider, session: SessionDep):
    user = get_user(request)
    return await delete_model_service_provider(session, user, provider)


@router.get("/available_models")
async def list_available_models_view(request, session: SessionDep) -> view_models.ModelConfigList:
    user = get_user(request)
    return await list_available_models(session, user)


@router.post("/chat/completions/frontend")
async def frontend_chat_completions_view(request: Request, session: SessionDep):
    user = get_user(request)
    message = request.body.decode("utf-8")
    query_params = dict(parse_qsl(request.GET.urlencode()))
    stream = query_params.get("stream", "false").lower() == "true"
    bot_id = query_params.get("bot_id", "")
    chat_id = query_params.get("chat_id", "")
    msg_id = request.headers.get("msg_id", "")
    return await frontend_chat_completions(session, user, message, stream, bot_id, chat_id, msg_id)


@router.post("/collections/{collection_id}/searchTests")
async def create_search_test_view(
    request, collection_id: str, data: view_models.SearchTestRequest, session: SessionDep
) -> view_models.SearchTestResult:
    user = get_user(request)
    return await create_search_test(session, user, collection_id, data)


@router.delete("/collections/{collection_id}/searchTests/{search_test_id}")
async def delete_search_test_view(request, collection_id: str, search_test_id: str, session: SessionDep):
    user = get_user(request)
    return await delete_search_test(session, user, collection_id, search_test_id)


@router.get("/collections/{collection_id}/searchTests")
async def list_search_tests_view(request, collection_id: str, session: SessionDep) -> view_models.SearchTestResultList:
    user = get_user(request)
    return await list_search_tests(session, user, collection_id)


@router.post("/bots/{bot_id}/flow/debug")
async def debug_flow_stream_view(
    request: Request, bot_id: str, debug: view_models.DebugFlowRequest, session: SessionDep
):
    user = get_user(request)
    return await debug_flow_stream(session, user, bot_id, debug)
