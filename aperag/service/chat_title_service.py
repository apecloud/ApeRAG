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

import re
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import BusinessException, ErrorCode
from aperag.service.default_model_service import default_model_service
from aperag.utils.history import RedisChatMessageHistory, get_async_redis_client


class ChatTitleService:
    """Service to generate chat titles using default background-task model configuration."""

    def __init__(self, session: Optional[AsyncSession] = None):
        self.db_ops = async_db_ops if session is None else AsyncDatabaseOps(session)

    async def generate_title(
        self,
        user_id: str,
        bot_id: str,
        chat_id: str,
        *,
        max_length: int = 20,
        language: str = "zh-CN",
        turns: int = 1,
    ) -> str:
        # Validate inputs
        max_length = max(6, min(max_length, 50))
        turns = max(1, turns)

        # Verify bot and chat ownership
        bot = await self.db_ops.query_bot(user_id, bot_id)
        if not bot:
            raise BusinessException(ErrorCode.BOT_NOT_FOUND, "Bot not found")

        chat = await self.db_ops.query_chat(user_id, bot_id, chat_id)
        if not chat:
            raise BusinessException(ErrorCode.CHAT_NOT_FOUND, "Chat not found")

        # Load default model configuration
        model, provider_name, custom_provider = await default_model_service.get_default_background_task_config(user_id)
        if not (model and provider_name and custom_provider):
            raise BusinessException(ErrorCode.LLM_MODEL_NOT_FOUND, "Background task default model not configured")

        # Resolve provider base_url and api_key
        provider = await self.db_ops.query_llm_provider_by_name(provider_name)
        if not provider:
            raise BusinessException(ErrorCode.LLM_MODEL_NOT_FOUND, f"Provider '{provider_name}' not found")
        base_url = provider.base_url
        api_key = await self.db_ops.query_provider_api_key(provider_name, user_id, True)
        if not api_key:
            raise BusinessException(
                ErrorCode.API_KEY_NOT_FOUND, f"API key for provider '{provider_name}' not configured"
            )

        # Read recent conversation turns from Redis
        history = RedisChatMessageHistory(chat_id, redis_client=get_async_redis_client())
        stored_messages = await history.messages
        # Take most recent N turns
        recent_turns = stored_messages[-turns:] if turns < len(stored_messages) else stored_messages
        # Convert to OpenAI format messages
        openai_messages = []
        for turn in recent_turns:
            openai_messages.extend(turn.to_openai_format())

        # Build prompt
        prompt = self._build_prompt(language=language, max_length=max_length)

        # Call completion service
        from aperag.llm.completion.completion_service import CompletionService

        completion_service = CompletionService(
            provider=custom_provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.2,
            max_tokens=64,
        )

        response = await completion_service.agenerate(
            history=openai_messages, prompt=prompt, images=[], memory=bool(openai_messages)
        )
        title = self._postprocess_title(response, max_length=max_length)
        return title

    @staticmethod
    def _build_prompt(language: str, max_length: int) -> str:
        if language == "en-US":
            return (
                "Generate a concise chat title summarizing the recent conversation. "
                f"Only return the title text, no quotes, no punctuation at the end. Max {max_length} characters. The title is:"
            )
        # Default zh-CN
        return (
            "请基于最近的对话内容生成一个简短的中文标题。只返回标题文本，不要引号，末尾不要标点符号。"
            f"最多 {max_length} 个字符。标题是:"
        )

    @staticmethod
    def _postprocess_title(raw: str, max_length: int) -> str:
        if not raw:
            return "Untitled"
        title = raw.strip()
        title = title.replace("\n", " ").replace("\r", " ")
        title = re.sub(r"\s+", " ", title)
        # Trim trailing punctuation
        title = re.sub(r"[\s\-—_:;,.!?，。！；：、…]+$", "", title)
        if len(title) > max_length:
            title = title[:max_length].rstrip()
        return title or "Untitled"


chat_title_service = ChatTitleService()
