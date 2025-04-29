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

import json
from http import HTTPStatus
from typing import Dict, Any, Generic, TypeVar, Tuple

from django.http import HttpRequest, HttpResponse
from langchain_core.prompts import PromptTemplate
from ninja.main import Exc
from ninja.errors import HttpError

from pydantic import ValidationError
from asgiref.sync import sync_to_async
from aperag.chat.history.redis import RedisChatMessageHistory
from aperag.chat.utils import get_async_redis_client
from aperag.db.models import Bot
from aperag.db.ops import PagedResult, logger, query_chat_feedbacks
from aperag.llm.base import Predictor 
from aperag.source.base import CustomSourceInitializationError, get_source
from aperag.utils.utils import AVAILABLE_SOURCE
from django.conf import settings
from aperag.auth.authentication import GlobalAuth
from aperag.utils import constant
from aperag.views import models as view_models

async def query_chat_messages(user: str, chat_id: str) -> list[view_models.ChatMessage]:
    pr = await query_chat_feedbacks(user, chat_id)
    feedback_map = {}
    async for feedback in pr.data:
        feedback_map[feedback.message_id] = feedback

    history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
    messages = []
    for message in await history.messages:
        try:
            item = json.loads(message.content)
        except Exception as e:
            logger.exception(e)
            continue
        role = message.additional_kwargs.get("role", "")
        if not role:
            continue
        msg = view_models.ChatMessage(
            id=item["id"],
            type="message",
            timestamp=item["timestamp"],
            role=role,
        )
        if role == "human":
            msg.data = item["query"]
        else:
            msg.data = item["response"]
            msg.references = []
            for ref in item.get("references", []):
                msg.references.append(view_models.Reference(
                    score=ref["score"],
                    text=ref["text"],
                    metadata=ref["metadata"]
                ))
            msg.urls = item.get("urls", [])
        feedback = feedback_map.get(item.get("id", ""), None)
        if role == "ai" and feedback:
            msg.upvote = feedback.upvote
            msg.downvote = feedback.downvote
            msg.revised_answer = feedback.revised_answer
            msg.feed_back_status = feedback.status
        messages.append(msg)
    return messages


def validate_source_connect_config(config: Dict) -> Tuple[bool, str]:
    if "source" not in config.keys():
        return False, ""
    if config.get("source") not in AVAILABLE_SOURCE:
        return False, ""
    try:
        get_source(config)
    except CustomSourceInitializationError as e:
        return False, str(e)
    return True, ""


def validate_bot_config(model_service_provider, model_name, base_url, api_key, config: Dict, type, memory) -> Tuple[bool, str]:
    try:
        Predictor.get_completion_service(model_service_provider, model_name, base_url, api_key, **config)
    except Exception as e:
        return False, str(e)

    try:
        # validate the prompt
        prompt_template = config.get("prompt_template", None)
        if not prompt_template and type == Bot.Type.COMMON:
            return False, "prompt of common bot cannot be null"
        if prompt_template and type == Bot.Type.KNOWLEDGE:
            PromptTemplate(template=prompt_template, input_variables=["query", "context"])
        elif prompt_template and type == Bot.Type.COMMON:
            PromptTemplate(template=prompt_template, input_variables=["query"])
            # pass
    except ValidationError:
        return False, "Invalid prompt template"

    context_window = config.get("context_window")
    if context_window > 5120000 or context_window < 0:
        return False, "Invalid context window"

    similarity_score_threshold = config.get("similarity_score_threshold")
    if similarity_score_threshold > 1.0 or similarity_score_threshold <= 0:
        return False, "Invalid similarity score threshold"

    similarity_topk = config.get("similarity_topk")
    if similarity_topk > 10 or similarity_topk <= 0:
        return False, "Invalid similarity topk"

    temperature = config.get("temperature")
    if temperature > 1.0 or temperature < 0:
        return False, "Invalid temperature"

    return True, ""


def validate_url(url):
    from urllib.parse import urlparse
    try:
        parsed_url = urlparse(url)

        if parsed_url.scheme not in ['http', 'https']:
            return False

        if not parsed_url.netloc:
            return False

        return True
    except Exception:
        return False

def success(data, pr: PagedResult = None):
    if not hasattr(data, "pageResult") or pr is None:
        return data
    
    data.pageResult = view_models.PageResult(
        count=pr.count,
        page_number=pr.page_number,
        page_size=pr.page_size,
    )
    return data


def fail(status: HTTPStatus, message: str, raise_exception: bool = True):
    if raise_exception:
        raise HttpError(status, message)
    return status, view_models.FailResponse(code=status.name, message=message)

if not settings.AUTH_TYPE:
    auth_middleware = None
else:
    auth_middleware = GlobalAuth()

def validation_errors(request: HttpRequest, exc: Exc) -> HttpResponse:
    msgs = []
    for err in exc.errors:
        for field in err["loc"]:
            msgs.append(f"{err['msg']}: {field}")
    status, content = fail(HTTPStatus.UNPROCESSABLE_ENTITY, ", ".join(msgs), raise_exception=False)
    return HttpResponse(status=status, content=content.model_dump_json())


def auth_errors(request: HttpRequest, exc: Exc) -> HttpResponse:
    status, content = fail(HTTPStatus.UNAUTHORIZED, "Unauthorized", raise_exception=False)
    return HttpResponse(status=status, content=content.model_dump_json())

