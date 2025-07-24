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
import logging
import uuid
from typing import Dict, Tuple

from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from aperag.db.models import BotType
from aperag.db.ops import async_db_ops
from aperag.schema import view_models
from aperag.schema.view_models import CollectionConfig
from aperag.source.base import CustomSourceInitializationError, get_source
from aperag.utils.history import RedisChatMessageHistory, get_async_redis_client
from aperag.utils.utils import AVAILABLE_SOURCE

logger = logging.getLogger(__name__)


async def query_chat_messages(user: str, chat_id: str) -> list[view_models.ChatMessage]:
    feedbacks = await async_db_ops.query_chat_feedbacks(user, chat_id)
    feedback_map = {}
    for feedback in feedbacks:
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
                msg.references.append(
                    view_models.Reference(
                        score=ref["score"],
                        text=ref.get("text", None),
                        image_uri=ref.get("image_uri", None),
                        metadata=ref["metadata"],
                    )
                )
            msg.urls = item.get("urls", [])
        feedback = feedback_map.get(item.get("id", ""), None)
        if role == "ai" and feedback:
            msg.feedback = view_models.Feedback(type=feedback.type, tag=feedback.tag, message=feedback.message)
        messages.append(msg)
    return messages


def validate_source_connect_config(config: CollectionConfig) -> Tuple[bool, str]:
    if config.source is None:
        return False, ""
    if config.source not in AVAILABLE_SOURCE:
        return False, ""
    try:
        get_source(config)
    except CustomSourceInitializationError as e:
        return False, str(e)
    return True, ""


def validate_bot_config(
    model_service_provider, model_name, base_url, api_key, config: Dict, type, memory
) -> Tuple[bool, str]:
    try:
        # validate the prompt
        prompt_template = config.get("prompt_template", None)
        if not prompt_template and type == BotType.COMMON:
            return False, "prompt of common bot cannot be null"
        if prompt_template and type == BotType.KNOWLEDGE:
            PromptTemplate(template=prompt_template, input_variables=["query", "context"])
        elif prompt_template and type == BotType.COMMON:
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

        if parsed_url.scheme not in ["http", "https"]:
            return False

        if not parsed_url.netloc:
            return False

        return True
    except Exception:
        return False


def mask_api_key(api_key: str) -> str:
    """
    Mask API key for security, showing only first 4 and last 4 characters.
    The number of mask characters reflects the actual length of the hidden part.

    Args:
        api_key: The original API key

    Returns:
        Masked API key string

    Examples:
        - sk-1234567890abcdef -> sk-1********cdef (16 chars total, 8 masked)
        - short_key -> short_key (if length <= 8, return as-is)
        - very_long_api_key_123456789 -> very********************6789 (25 chars total, 17 masked)
    """
    if not api_key or len(api_key) <= 8:
        return api_key

    # Calculate the number of characters to mask (total - first 4 - last 4)
    masked_length = len(api_key) - 8
    mask_chars = "*" * masked_length

    # Show first 4 and last 4 characters, mask the middle with actual length
    return f"{api_key[:4]}{mask_chars}{api_key[-4:]}"


def generate_random_provider_name() -> str:
    """
    Generate a random provider name using UUID.

    Returns:
        A random provider name in the format: provider_xxxxxxxx

    Examples:
        - provider_a1b2c3d4
        - provider_f7e8d9c0
    """
    # Generate a short UUID (first 8 characters)
    short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
    return f"provider_{short_uuid}"
