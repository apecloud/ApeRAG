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
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain.schema import AIMessage, BaseMessage, ChatMessage, FunctionMessage, HumanMessage, SystemMessage

from aperag.chat.history import (
    StoredChatMessage,
    create_assistant_message,
    create_user_message,
    message_to_storage_dict,
    storage_dict_to_message,
)

logger = logging.getLogger(__name__)


class BaseChatMessageHistory(ABC):
    """Abstract base class for storing chat message history.

    See `ChatMessageHistory` for default implementation.

    Example:
        .. code-block:: python

            class FileChatMessageHistory(BaseChatMessageHistory):
                storage_path:  str
                session_id: str

               @property
               def messages(self):
                   with open(os.path.join(storage_path, session_id), 'r:utf-8') as f:
                       messages = json.loads(f.read())
                    return messages_from_dict(messages)

               def add_message(self, message: BaseMessage) -> None:
                   messages = self.messages.append(_message_to_dict(message))
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       json.dump(f, messages)

               def clear(self):
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       f.write("[]")
    """

    async def add_user_message(self, message: str, message_id: str) -> None:
        """Convenience method for adding a human message string to the store.

        Args:
            message: The string contents of a human message.
        """
        raise NotImplementedError()

    async def add_ai_message(
        self,
        content: str,
        chat_id: str,
        message_id: str = None,
        tool_use_list: List = None,
        references: List[Dict[str, Any]] = None,
        urls: List[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Convenience method for adding an AI message string to the store.

        Args:
            message: The string contents of an AI message.
        """
        raise NotImplementedError()

    @abstractmethod
    async def clear(self) -> None:
        """Remove all messages from the store"""
        raise NotImplementedError()

    @property
    async def messages(self) -> List[StoredChatMessage]:
        """Retrieve all messages from the store.

        Returns:
            A list of BaseMessage objects.
        """
        raise NotImplementedError()


async def message_from_dict(message: dict) -> BaseMessage:
    _type = message["type"]
    if _type == "human":
        return HumanMessage(**message["data"])
    elif _type == "ai":
        return AIMessage(**message["data"])
    elif _type == "system":
        return SystemMessage(**message["data"])
    elif _type == "chat":
        return ChatMessage(**message["data"])
    elif _type == "function":
        return FunctionMessage(**message["data"])
    else:
        raise ValueError(f"Got unexpected message type: {_type}")


class RedisChatMessageHistory:
    """Chat message history stored in a Redis database using ApeRAG StoredChatMessage format."""

    def __init__(
        self,
        session_id: str,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "message_store:",
        ttl: Optional[int] = None,
        redis_client=None,
    ):
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "Could not import redis.asyncio python package. Please make sure that redis version >= 4.0.0"
            )
        try:
            self.redis_client = redis_client or redis.Redis.from_url(url)
        except Exception as e:
            logger.error(e)

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl

    @property
    def key(self) -> str:
        """Construct the record key to use"""
        return self.key_prefix + self.session_id

    @property
    async def messages(self) -> List[StoredChatMessage]:
        """Retrieve the messages from Redis as StoredChatMessage objects"""
        _items = await self.redis_client.lrange(self.key, 0, -1)
        items = [json.loads(m.decode("utf-8")) for m in _items[::-1]]  # Reverse to get chronological order
        messages = []
        for item in items:
            try:
                message = storage_dict_to_message(item)
                messages.append(message)
            except Exception as e:
                logger.warning(f"Failed to parse message in history for {self.session_id}: {e}")
                continue
        return messages

    async def add_stored_message(self, message: StoredChatMessage) -> None:
        """Add a StoredChatMessage directly to Redis"""
        message_json = json.dumps(message_to_storage_dict(message))
        await self.redis_client.lpush(self.key, message_json)
        if self.ttl:
            await self.redis_client.expire(self.key, self.ttl)

    async def add_user_message(self, message: str, message_id: str) -> None:
        """Add a user message using new format"""
        stored_message = create_user_message(
            content=message,
            chat_id=self.session_id,
            message_id=message_id,
        )
        await self.add_stored_message(stored_message)

    async def add_ai_message(
        self,
        content: str,
        chat_id: str,
        message_id: str = None,
        tool_use_list: List = None,
        references: List[Dict[str, Any]] = None,
        urls: List[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add an AI message using new format"""
        stored_message = create_assistant_message(
            content=content,
            chat_id=self.session_id,
            message_id=message_id,
            tool_use_list=tool_use_list,
            references=references,
            urls=urls,
            trace_id=trace_id,
            metadata=metadata,
        )
        await self.add_stored_message(stored_message)

    async def clear(self) -> None:
        """Clear session memory from Redis"""
        await self.redis_client.delete(self.key)

    async def release_redis(self):
        await self.redis_client.close(close_connection_pool=True)


async def query_chat_messages(user: str, chat_id: str):
    """
    Query chat messages from Redis and convert to frontend format.

    Returns:
        Array of conversation turns, where each turn is an array of message parts
        格式: [[turn1_parts], [turn2_parts], ...]
    """
    from aperag.db.ops import async_db_ops
    from aperag.schema import view_models

    try:
        # Get all stored messages (each StoredChatMessage represents one conversation turn)
        chat_history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
        stored_messages = await chat_history.messages

        if not stored_messages:
            return []

        # Get feedbacks for this chat
        feedbacks = await async_db_ops.query_chat_feedbacks(user, chat_id)
        feedback_map = {feedback.message_id: feedback for feedback in feedbacks}

        # Convert each StoredChatMessage (conversation turn) to frontend format
        conversation_turns = []
        for stored_message in stored_messages:
            # Convert this turn to frontend format (returns array of parts)
            chat_message_list = stored_message.to_frontend_format()

            # Add feedback data if available
            for chat_msg in chat_message_list:
                msg_id = chat_msg.id
                feedback = feedback_map.get(msg_id)
                if feedback and chat_msg.role == "ai":
                    chat_msg.feedback = view_models.Feedback(
                        type=feedback.type, tag=feedback.tag, message=feedback.message
                    )

            conversation_turns.append(chat_message_list)

        return conversation_turns

    except Exception as e:
        logger.error(f"Error querying chat messages: {e}")
        return []


def success_response(message_id, data):
    return json.dumps(
        {
            "type": "message",
            "id": message_id,
            "data": data,
            "timestamp": int(time.time()),
        }
    )


def fail_response(message_id, error):
    return json.dumps(
        {
            "type": "error",
            "id": message_id,
            "data": error,
            "timestamp": int(time.time()),
        }
    )


def start_response(message_id):
    return json.dumps(
        {
            "type": "start",
            "id": message_id,
            "timestamp": int(time.time()),
        }
    )


def references_response(message_id, references, memory_count=0, urls=[]):
    if references is None:
        references = []
    return json.dumps(
        {
            "type": "references",
            "id": message_id,
            "data": references,
            "memoryCount": memory_count,
            "urls": urls,
            "timestamp": int(time.time()),
        }
    )


def stop_response(message_id):
    return json.dumps(
        {
            "type": "stop",
            "id": message_id,
            "timestamp": int(time.time()),
        }
    )


def get_async_redis_client():
    global async_redis_client
    if not async_redis_client:
        import redis.asyncio as redis

        from aperag.config import settings

        async_redis_client = redis.Redis.from_url(settings.memory_redis_url)
    return async_redis_client


async_redis_client = None
