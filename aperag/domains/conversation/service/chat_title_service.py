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

"""Chat-title generation service moved to the conversation domain in
Phase 5 step 5-S4e.

Title generation reads the recent ``AgentTurn`` rows for the chat
(canonical post-D8.5 #92), composes an OpenAI-format prompt from
each turn's ``input_text`` (user) plus the persisted assistant
``UIMessagePart`` text content (via :class:`UIMessageStore`), and
invokes the background-task ``ModelUse`` via the shared model
invocation service.

Phase 8 D8.6 (#80) hard-cut removed the legacy
``RedisChatMessageHistory`` read path; chat history now flows through
the ``agent_message`` table only.
"""

from __future__ import annotations

import re
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.domains.agent_runtime.storage import AgentRuntimeRedisStore
from aperag.domains.agent_runtime.uimessage_store import UIMessageDbOps, UIMessageStore
from aperag.exceptions import BusinessException, ErrorCode
from aperag.llm.runtime.invocation_service import model_invocation_service


class ChatTitleService:
    """Service to generate chat titles using default background-task model configuration."""

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        *,
        uimessage_store: Optional[UIMessageStore] = None,
    ):
        self.db_ops = async_db_ops if session is None else AsyncDatabaseOps(session)
        self.uimessage_store = uimessage_store

    async def _resolve_uimessage_store(self) -> UIMessageStore:
        if self.uimessage_store is not None:
            return self.uimessage_store
        from aperag.config import get_async_session

        self.uimessage_store = UIMessageStore(
            db_ops=UIMessageDbOps(session_factory=get_async_session),
            redis_store=AgentRuntimeRedisStore(),
        )
        return self.uimessage_store

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

        # Read recent conversation turns from the canonical agent_turn table.
        agent_turns = await self.db_ops.query_agent_turns(user_id, chat_id)
        if not agent_turns:
            current_title = getattr(chat, "title", None)
            return current_title.strip() if current_title and current_title.strip() else "Untitled"

        model_id = None
        for item in await self.db_ops.query_model_uses(user_id):
            if item.scenario == "background_task" and item.primary_model_id:
                model_id = item.primary_model_id
                break
        if not model_id:
            raise BusinessException(ErrorCode.LLM_MODEL_NOT_FOUND, "Background task model is not configured")

        # Take most recent N turns and compose an OpenAI-format prompt
        # from each turn's user input + persisted assistant text parts.
        recent_turns = agent_turns[-turns:] if turns < len(agent_turns) else agent_turns
        store = await self._resolve_uimessage_store()
        openai_messages: list[dict[str, str]] = []
        for turn in recent_turns:
            if turn.input_text:
                openai_messages.append({"role": "user", "content": turn.input_text})
            assistant_text = await self._extract_assistant_text(store, turn.id)
            if assistant_text:
                openai_messages.append({"role": "assistant", "content": assistant_text})

        if not openai_messages:
            current_title = getattr(chat, "title", None)
            return current_title.strip() if current_title and current_title.strip() else "Untitled"

        # Build prompt
        prompt = self._build_prompt(language=language, max_length=max_length)

        messages = openai_messages + [{"role": "user", "content": prompt}]
        response = await model_invocation_service.chat(
            model_id,
            user_id,
            messages,
            temperature=0.2,
            max_tokens=64,
        )
        response = response.choices[0].message.content
        title = self._postprocess_title(response, max_length=max_length)
        return title

    @staticmethod
    async def _extract_assistant_text(store: UIMessageStore, turn_id: str) -> Optional[str]:
        """Pull a plain-text rendering of the assistant turn's parts.

        The canonical assistant payload is the persisted UIMessage's
        ``parts`` array; ``text`` parts contribute their ``text`` field
        joined into a single string. Other part types (tool calls,
        citations, source URLs, etc.) are intentionally skipped — title
        generation only needs the assistant's narrative text.
        """

        message = await store.read(turn_id)
        if message is None or not message.parts:
            return None
        chunks: list[str] = []
        for part in message.parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                chunks.append(text)
        if not chunks:
            return None
        return "\n".join(chunks)

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


# Global service instance — wired by the legacy shim at
# ``aperag.service.chat_title_service``.
chat_title_service = ChatTitleService()
