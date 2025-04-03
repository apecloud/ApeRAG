import json
import logging
from typing import List, Optional

from langchain.schema.messages import AIMessage, BaseMessage, ChatMessage, FunctionMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class RedisChatMessageHistory:
    """Chat message history stored in a Redis database."""

    def __init__(self,
                 session_id: str,
                 url: str = "redis://localhost:6379/0",
                 key_prefix: str = "message_store:",
                 ttl: Optional[int] = None,
                 redis_client=None,):
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError("Could not import redis.asyncio python package. "
                              "Please make sure that redis version >= 4.0.0")
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
