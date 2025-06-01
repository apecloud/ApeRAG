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

from django.http import HttpRequest
from django.shortcuts import render
from ninja import File, Router
from ninja.files import UploadedFile

from aperag.chat.message import feedback_message
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
from aperag.service.dashboard_service import dashboard_service
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
from aperag.service.sync_service import cancel_sync, get_sync_history, list_sync_histories, sync_immediately
from aperag.utils.request import get_urls, get_user
from aperag.views.utils import success
from config import settings

logger = logging.getLogger(__name__)

router = Router()


@router.get("/prompt-templates")
async def list_prompt_templates_view(request) -> view_models.PromptTemplateList:
    language = request.headers.get("Lang", "zh-CN")
    return list_prompt_templates(language)


@router.post("/collections/{collection_id}/sync")
async def sync_immediately_view(request, collection_id):
    user = get_user(request)
    return await sync_immediately(user, collection_id)


@router.post("/collections/{collection_id}/cancel_sync/{collection_sync_id}")
async def cancel_sync_view(request, collection_id, collection_sync_id):
    user = get_user(request)
    return await cancel_sync(user, collection_id, collection_sync_id)


@router.get("/collections/{collection_id}/sync/history")
async def list_sync_histories_view(request, collection_id):
    user = get_user(request)
    return await list_sync_histories(user, collection_id, build_pq(request))


@router.get("/collections/{collection_id}/sync/{sync_history_id}")
async def get_sync_history_view(request, collection_id, sync_history_id):
    user = get_user(request)
    return await get_sync_history(user, collection_id, sync_history_id)


@router.post("/collections")
async def create_collection_view(request, collection: view_models.CollectionCreate) -> view_models.Collection:
    user = get_user(request)
    return await create_collection(user, collection)


@router.get("/collections")
async def list_collections_view(request) -> view_models.CollectionList:
    from aperag.service.collection_service import list_collections

    user = get_user(request)
    return await list_collections(user, build_pq(request))


@router.get("/collections/{collection_id}")
async def get_collection_view(request, collection_id: str) -> view_models.Collection:
    from aperag.service.collection_service import get_collection

    user = get_user(request)
    return await get_collection(user, collection_id)


@router.put("/collections/{collection_id}")
async def update_collection_view(
    request, collection_id: str, collection: view_models.CollectionUpdate
) -> view_models.Collection:
    from aperag.service.collection_service import update_collection

    user = get_user(request)
    return await update_collection(user, collection_id, collection)


@router.delete("/collections/{collection_id}")
async def delete_collection_view(request, collection_id: str) -> view_models.Collection:
    from aperag.service.collection_service import delete_collection

    user = get_user(request)
    return await delete_collection(user, collection_id)


@router.post("/collections/{collection_id}/documents")
async def create_documents_view(
    request, collection_id: str, files: List[UploadedFile] = File(...)
) -> List[view_models.Document]:
    user = get_user(request)
    return await create_documents(user, collection_id, files)


@router.post("/collections/{collection_id}/urls")
async def create_url_document_view(request, collection_id: str) -> List[view_models.Document]:
    user = get_user(request)
    urls = get_urls(request)
    return await create_url_document(user, collection_id, urls)


@router.get("/collections/{collection_id}/documents")
async def list_documents_view(request, collection_id: str) -> view_models.DocumentList:
    user = get_user(request)
    return await list_documents(user, collection_id, build_pq(request))


@router.get("/collections/{collection_id}/documents/{document_id}")
async def get_document_view(request, collection_id: str, document_id: str) -> view_models.Document:
    user = get_user(request)
    return await get_document(user, collection_id, document_id)


@router.put("/collections/{collection_id}/documents/{document_id}")
async def update_document_view(
    request, collection_id: str, document_id: str, document: view_models.Document
) -> view_models.Document:
    user = get_user(request)
    return await update_document(user, collection_id, document_id, document)


@router.delete("/collections/{collection_id}/documents/{document_id}")
async def delete_document_view(request, collection_id: str, document_id: str) -> view_models.Document:
    user = get_user(request)
    return await delete_document(user, collection_id, document_id)


@router.delete("/collections/{collection_id}/documents")
async def delete_documents_view(request, collection_id: str, document_ids: List[str]):
    user = get_user(request)
    return await delete_documents(user, collection_id, document_ids)


@router.post("/bots/{bot_id}/chats")
async def create_chat_view(request, bot_id: str) -> view_models.Chat:
    user = get_user(request)
    return await create_chat(user, bot_id)


@router.get("/bots/{bot_id}/chats")
async def list_chats_view(request, bot_id: str) -> view_models.ChatList:
    from aperag.service.chat_service import list_chats

    user = get_user(request)
    return await list_chats(user, bot_id, build_pq(request))


@router.get("/bots/{bot_id}/chats/{chat_id}")
async def get_chat_view(request, bot_id: str, chat_id: str) -> view_models.Chat:
    from aperag.service.chat_service import get_chat

    user = get_user(request)
    return await get_chat(user, bot_id, chat_id)


@router.put("/bots/{bot_id}/chats/{chat_id}")
async def update_chat_view(request, bot_id: str, chat_id: str, chat_in: view_models.ChatUpdate) -> view_models.Chat:
    from aperag.service.chat_service import update_chat

    user = get_user(request)
    return await update_chat(user, bot_id, chat_id, chat_in)


@router.post("/bots/{bot_id}/chats/{chat_id}/messages/{message_id}")
async def feedback_message_view(request, bot_id: str, chat_id: str, message_id: str, feedback: view_models.Feedback):
    user = get_user(request)
    await feedback_message(user, chat_id, message_id, feedback.type, feedback.tag, feedback.message)
    return success({})


@router.delete("/bots/{bot_id}/chats/{chat_id}")
async def delete_chat_view(request, bot_id: str, chat_id: str) -> view_models.Chat:
    from aperag.service.chat_service import delete_chat

    user = get_user(request)
    return await delete_chat(user, bot_id, chat_id)


@router.post("/bots")
async def create_bot_view(request, bot_in: view_models.BotCreate) -> view_models.Bot:
    user = get_user(request)
    return await create_bot(user, bot_in)


@router.get("/bots")
async def list_bots_view(request) -> view_models.BotList:
    from aperag.service.bot_service import list_bots

    user = get_user(request)
    return await list_bots(user, build_pq(request))


@router.get("/bots/{bot_id}")
async def get_bot_view(request, bot_id: str) -> view_models.Bot:
    from aperag.service.bot_service import get_bot

    user = get_user(request)
    return await get_bot(user, bot_id)


@router.put("/bots/{bot_id}")
async def update_bot_view(request, bot_id: str, bot_in: view_models.BotUpdate) -> view_models.Bot:
    from aperag.service.bot_service import update_bot

    user = get_user(request)
    return await update_bot(user, bot_id, bot_in)


@router.delete("/bots/{bot_id}")
async def delete_bot_view(request, bot_id: str) -> view_models.Bot:
    from aperag.service.bot_service import delete_bot

    user = get_user(request)
    return await delete_bot(user, bot_id)


@router.get("/supported_model_service_providers")
async def list_supported_model_service_providers_view(request) -> view_models.ModelServiceProviderList:
    return await list_supported_model_service_providers()


@router.get("/model_service_providers")
async def list_model_service_providers_view(request) -> view_models.ModelServiceProviderList:
    user = get_user(request)
    return await list_model_service_providers(user)


@router.put("/model_service_providers/{provider}")
async def update_model_service_provider_view(request, provider: str, mspIn: view_models.ModelServiceProviderUpdate):
    from aperag.schema.view_models import ModelConfig

    user = get_user(request)
    supported_providers = [ModelConfig(**item) for item in settings.MODEL_CONFIGS]
    return await update_model_service_provider(user, provider, mspIn, supported_providers)


@router.delete("/model_service_providers/{provider}")
async def delete_model_service_provider_view(request, provider):
    user = get_user(request)
    return await delete_model_service_provider(user, provider)


@router.get("/available_models")
async def list_available_models_view(request) -> view_models.ModelConfigList:
    user = get_user(request)
    return await list_available_models(user)


def default_page(request, exception):
    return render(request, "404.html")


def dashboard(request):
    return dashboard_service(request)


@router.post("/chat/completions/frontend")
async def frontend_chat_completions_view(request: HttpRequest):
    user = get_user(request)
    message = request.body.decode("utf-8")
    query_params = dict(parse_qsl(request.GET.urlencode()))
    stream = query_params.get("stream", "false").lower() == "true"
    bot_id = query_params.get("bot_id", "")
    chat_id = query_params.get("chat_id", "")
    msg_id = request.headers.get("msg_id", "")
    return await frontend_chat_completions(user, message, stream, bot_id, chat_id, msg_id)


@router.post("/collections/{collection_id}/searchTests")
async def create_search_test_view(
    request, collection_id: str, data: view_models.SearchTestRequest
) -> view_models.SearchTestResult:
    user = get_user(request)
    return await create_search_test(user, collection_id, data)


@router.delete("/collections/{collection_id}/searchTests/{search_test_id}")
async def delete_search_test_view(request, collection_id: str, search_test_id: str):
    user = get_user(request)
    return await delete_search_test(user, collection_id, search_test_id)


@router.get("/collections/{collection_id}/searchTests")
async def list_search_tests_view(request, collection_id: str) -> view_models.SearchTestResultList:
    user = get_user(request)
    return await list_search_tests(user, collection_id)


@router.post("/bots/{bot_id}/flow/debug")
async def debug_flow_stream_view(request: HttpRequest, bot_id: str, debug: view_models.DebugFlowRequest):
    user = get_user(request)
    return await debug_flow_stream(user, bot_id, debug)
