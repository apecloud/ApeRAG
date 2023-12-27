import json
import os
import shutil
from http import HTTPStatus
from typing import Dict

from django.http import HttpRequest, HttpResponse
from langchain import PromptTemplate
from ninja.main import Exc
from pydantic import ValidationError

from config import settings
from kubechat.chat.history.redis import RedisChatMessageHistory
from kubechat.llm.base import Predictor, PredictorType
from kubechat.db.models import ssl_file_path, ssl_temp_file_path, BotType
from kubechat.db.ops import query_chat_feedbacks, logger, PagedResult
from kubechat.source.base import get_source, CustomSourceInitializationError
from kubechat.utils.utils import AVAILABLE_SOURCE


def add_ssl_file(config, collection):
    if not os.path.exists(ssl_file_path(collection, "")):
        os.makedirs(ssl_file_path(collection, ""))

    for ssl_file_type in ["ca_cert", "client_key", "client_cert"]:
        if ssl_file_type in config.keys():
            ssl_temp_name = config[ssl_file_type]
            _, file_extension = os.path.splitext(ssl_temp_name)
            ssl_file_name = ssl_file_type + file_extension
            whole_ssl_file_path = ssl_file_path(collection, ssl_file_name)
            shutil.move(ssl_temp_file_path(ssl_temp_name), whole_ssl_file_path)
            config[ssl_file_type] = whole_ssl_file_path


async def query_chat_messages(user, chat_id):
    pr = await query_chat_feedbacks(user, chat_id)
    feedback_map = {}
    async for feedback in pr.data:
        feedback_map[feedback.message_id] = feedback

    history = RedisChatMessageHistory(chat_id, url=settings.MEMORY_REDIS_URL)
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
        msg = {
            "id": item["id"],
            "type": "message",
            "timestamp": item["timestamp"],
            "role": role,
        }
        if role == "human":
            msg["data"] = item["query"]
        else:
            msg["data"] = item["response"]
            msg["references"] = item.get("references")
        feedback = feedback_map.get(item.get("id", ""), None)
        if role == "ai" and feedback:
            msg["upvote"] = feedback.upvote
            msg["downvote"] = feedback.downvote
            msg["revised_answer"] = feedback.revised_answer
            msg["feed_back_status"] = feedback.status
        messages.append(msg)
    return messages


def validate_source_connect_config(config: Dict) -> (bool, str):
    if "source" not in config.keys():
        return False, ""
    if config.get("source") not in AVAILABLE_SOURCE:
        return False, ""
    try:
        get_source(config)
    except CustomSourceInitializationError as e:
        return False, str(e)
    return True, ""


def validate_bot_config(model, config: Dict, bot) -> (bool, str):
    try:
        Predictor.from_model(model, PredictorType.CUSTOM_LLM, **config)
    except Exception as e:
        return False, str(e)

    try:
        # validate the prompt
        prompt_template = config.get("prompt_template", None)
        if bot.type == BotType.KNOWLEDGE:
            PromptTemplate(template=prompt_template, input_variables=["query", "context"])
        elif bot.type == BotType.COMMON:
            PromptTemplate(template=prompt_template, input_variables=["query"])
            # pass
        else:
            return False, "Unsupported bot type"
    except ValidationError:
        return False, "Invalid prompt template"

    try:
        # validate the memory prompt
        prompt_template = config.get("memory_prompt_template", None)
        if prompt_template and bot.type == BotType.KNOWLEDGE:
            PromptTemplate(template=prompt_template, input_variables=["query", "context"])
        elif prompt_template and bot.type == BotType.COMMON:
            PromptTemplate(template=prompt_template, input_variables=["query"])
            # pass
    except ValidationError:
        return False, "Invalid memory prompt template"

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
    response = {
        "code": "%d" % HTTPStatus.OK,
        "data": data,
    }
    if pr is not None and pr.count > 0:
        response["page_number"] = pr.page_number
        response["page_size"] = pr.page_size
        response["count"] = pr.count
    return response


def fail(code, message):
    return {
        "code": "%d" % code,
        "message": message,
    }


def validation_errors(request: HttpRequest, exc: Exc) -> HttpResponse:
    msgs = []
    for err in exc.errors:
        for field in err["loc"]:
            msgs.append(f"{err['msg']}: {field}")
    content = json.dumps(fail(HTTPStatus.UNPROCESSABLE_ENTITY, ", ".join(msgs)))
    return HttpResponse(status=HTTPStatus.OK, content=content)


def auth_errors(request: HttpRequest, exc: Exc) -> HttpResponse:
    content = json.dumps(fail(HTTPStatus.UNAUTHORIZED, "Unauthorized"))
    return HttpResponse(status=HTTPStatus.OK, content=content)

