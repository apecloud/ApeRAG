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
        """Build language-specific prompt for title generation

        Args:
            language: Language code (zh-CN, en-US, etc.)
            max_length: Maximum length of the title

        Returns:
            Formatted prompt string for the specified language
        """
        prompts = {
            "en-US": (
                "You MUST respond in English only. Create a concise English title that summarizes the conversation. "
                f"Requirements: 1) MUST be in English, 2) Max {max_length} characters, 3) No quotes or punctuation at end, "
                "4) Clear and descriptive. "
                "Regardless of the conversation language, your response MUST be in English. Title:"
            ),
            "zh-CN": (
                "你必须只用中文回答。基于最近的对话内容，生成一个简洁的中文标题。"
                f"要求：1) 必须使用中文，2) 最多 {max_length} 个字符，3) 不要引号和末尾标点，"
                "4) 清晰且具有描述性。"
                "无论对话是什么语言，你的回答必须是中文。标题："
            ),
            "ja-JP": (
                "日本語でのみ応答してください。会話を要約する簡潔な日本語タイトルを作成してください。"
                f"要件：1) 必ず日本語で、2) 最大 {max_length} 文字、3) 引用符や末尾の句読点なし、"
                "4) 明確で説明的。"
                "会話がどの言語であっても、回答は必ず日本語でお願いします。タイトル："
            ),
            "ko-KR": (
                "한국어로만 응답해야 합니다. 대화를 요약하는 간결한 한국어 제목을 만드세요. "
                f"요구사항: 1) 반드시 한국어로, 2) 최대 {max_length}자, 3) 따옴표나 끝의 구두점 없음, "
                "4) 명확하고 설명적. "
                "대화가 어떤 언어든 상관없이 반드시 한국어로 응답하세요. 제목:"
            ),
        }

        # Get prompt for specified language, fallback to zh-CN
        return prompts.get(language, prompts["zh-CN"])

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
