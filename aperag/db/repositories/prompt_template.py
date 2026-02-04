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

from typing import List, Optional

from sqlalchemy import select

from aperag.db.models import PromptTemplate
from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.utils.utils import utc_now


class AsyncPromptTemplateRepositoryMixin(AsyncRepositoryProtocol):
    """Prompt Template Repository for managing user and system default prompts"""

    async def query_prompt_template(
        self, prompt_type: str, scope: str, user_id: Optional[str], language: str
    ) -> Optional[PromptTemplate]:
        """
        Query a single prompt template by type, scope, user_id and language.

        Args:
            prompt_type: Type of prompt (agent_system, agent_query, index_graph, etc.)
            scope: 'user' or 'system'
            user_id: User ID (required for scope='user', None for scope='system')
            language: Language code (en-US, zh-CN)

        Returns:
            PromptTemplate instance or None
        """

        async def _query(session):
            stmt = select(PromptTemplate).where(
                PromptTemplate.prompt_type == prompt_type,
                PromptTemplate.scope == scope,
                PromptTemplate.language == language,
                PromptTemplate.gmt_deleted.is_(None),
            )

            if scope == "user":
                stmt = stmt.where(PromptTemplate.user_id == user_id)
            else:
                stmt = stmt.where(PromptTemplate.user_id.is_(None))

            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_user_prompt_templates(self, user_id: str, language: Optional[str] = None) -> List[PromptTemplate]:
        """
        Query all prompt templates for a specific user.

        Args:
            user_id: User ID
            language: Optional language filter

        Returns:
            List of PromptTemplate instances
        """

        async def _query(session):
            stmt = select(PromptTemplate).where(
                PromptTemplate.scope == "user", PromptTemplate.user_id == user_id, PromptTemplate.gmt_deleted.is_(None)
            )

            if language:
                stmt = stmt.where(PromptTemplate.language == language)

            stmt = stmt.order_by(PromptTemplate.prompt_type, PromptTemplate.language)

            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_system_prompt_templates(self, language: Optional[str] = None) -> List[PromptTemplate]:
        """
        Query all system default prompt templates.

        Args:
            language: Optional language filter

        Returns:
            List of PromptTemplate instances
        """

        async def _query(session):
            stmt = select(PromptTemplate).where(PromptTemplate.scope == "system", PromptTemplate.gmt_deleted.is_(None))

            if language:
                stmt = stmt.where(PromptTemplate.language == language)

            stmt = stmt.order_by(PromptTemplate.prompt_type, PromptTemplate.language)

            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def create_or_update_prompt_template(
        self,
        prompt_type: str,
        scope: str,
        user_id: Optional[str],
        language: str,
        content: str,
        description: Optional[str] = None,
    ) -> PromptTemplate:
        """
        Create or update a prompt template.

        Args:
            prompt_type: Type of prompt
            scope: 'user' or 'system'
            user_id: User ID (required for scope='user')
            language: Language code
            content: Prompt content
            description: Optional description

        Returns:
            PromptTemplate instance
        """

        async def _operation(session):
            # Try to find existing template
            stmt = select(PromptTemplate).where(
                PromptTemplate.prompt_type == prompt_type,
                PromptTemplate.scope == scope,
                PromptTemplate.language == language,
                PromptTemplate.gmt_deleted.is_(None),
            )

            if scope == "user":
                stmt = stmt.where(PromptTemplate.user_id == user_id)
            else:
                stmt = stmt.where(PromptTemplate.user_id.is_(None))

            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                # Update existing
                instance.content = content
                if description is not None:
                    instance.description = description
                instance.gmt_updated = utc_now()
            else:
                # Create new
                instance = PromptTemplate(
                    prompt_type=prompt_type,
                    scope=scope,
                    user_id=user_id,
                    language=language,
                    content=content,
                    description=description,
                )

            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def delete_prompt_template(self, prompt_type: str, scope: str, user_id: Optional[str], language: str) -> bool:
        """
        Soft delete a prompt template.

        Args:
            prompt_type: Type of prompt
            scope: 'user' or 'system'
            user_id: User ID (required for scope='user')
            language: Language code

        Returns:
            True if deleted, False if not found
        """

        async def _operation(session):
            stmt = select(PromptTemplate).where(
                PromptTemplate.prompt_type == prompt_type,
                PromptTemplate.scope == scope,
                PromptTemplate.language == language,
                PromptTemplate.gmt_deleted.is_(None),
            )

            if scope == "user":
                stmt = stmt.where(PromptTemplate.user_id == user_id)
            else:
                stmt = stmt.where(PromptTemplate.user_id.is_(None))

            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                instance.gmt_deleted = utc_now()
                session.add(instance)
                await session.flush()
                return True

            return False

        return await self.execute_with_transaction(_operation)
