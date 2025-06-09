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
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain.schema.messages import AIMessage, BaseMessage, ChatMessage, FunctionMessage, HumanMessage, SystemMessage

from aperag.utils.utils import now_unix_milliseconds

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

    async def add_user_message(self, message: str) -> None:
        """Convenience method for adding a human message string to the store.

        Args:
            message: The string contents of a human message.
        """
        await self.add_message(HumanMessage(content=message))

    async def add_ai_message(self, message: str) -> None:
        """Convenience method for adding an AI message string to the store.

        Args:
            message: The string contents of an AI message.
        """
        await self.add_message(AIMessage(content=message))

    @abstractmethod
    async def add_message(self, message: BaseMessage) -> None:
        """Add a Message object to the store.

        Args:
            message: A BaseMessage object to store.
        """
        raise NotImplementedError()

    @abstractmethod
    async def clear(self) -> None:
        """Remove all messages from the store"""

    @property
    async def messages(self) -> List[BaseMessage]:
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
    """Chat message history stored in a Redis database."""

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
    async def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from Redis"""
        _items = await self.redis_client.lrange(self.key, 0, -1)
        items = [json.loads(m.decode("utf-8")) for m in _items[::-1]]
        messages = []
        for m in items:
            message = await message_from_dict(m)
            messages.append(message)
        return messages

    async def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Redis"""
        message_json = json.dumps({"type": message.type, "data": message.dict()})
        await self.redis_client.lpush(self.key, message_json)
        if self.ttl:
            await self.redis_client.expire(self.key, self.ttl)

    async def add_user_message(self, message: str) -> None:
        await self.add_message(HumanMessage(content=message))

    async def add_ai_message(self, message: str) -> None:
        await self.add_message(AIMessage(content=message))

    async def clear(self) -> None:
        """Clear session memory from Redis"""
        await self.redis_client.delete(self.key)

    async def release_redis(self):
        await self.redis_client.close(close_connection_pool=True)


async_redis_client = None
sync_redis_client = None


def get_async_redis_client():
    global async_redis_client
    if not async_redis_client:
        import redis.asyncio as redis

        from aperag.config import settings

        async_redis_client = redis.Redis.from_url(settings.memory_redis_url)
    return async_redis_client


def get_sync_redis_client():
    global sync_redis_client
    if not sync_redis_client:
        import redis

        from aperag.config import settings

        sync_redis_client = redis.Redis.from_url(settings.memory_redis_url)
    return sync_redis_client


def success_response(message_id, data):
    return json.dumps(
        {
            "type": "message",
            "id": message_id,
            "data": data,
            "timestamp": now_unix_milliseconds(),
        }
    )


def fail_response(message_id, error):
    return json.dumps(
        {
            "type": "error",
            "id": message_id,
            "data": error,
            "timestamp": now_unix_milliseconds(),
        }
    )


def start_response(message_id):
    return json.dumps(
        {
            "type": "start",
            "id": message_id,
            "timestamp": now_unix_milliseconds(),
        }
    )


def stop_response(message_id, references, memory_count=0, urls=[]):
    if references is None:
        references = []
    return json.dumps(
        {
            "type": "stop",
            "id": message_id,
            "data": references,
            "memoryCount": memory_count,
            "urls": urls,
            "timestamp": now_unix_milliseconds(),
        }
    )
