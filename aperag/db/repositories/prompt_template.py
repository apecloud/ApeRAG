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
        self, prompt_type: str, scope: str, user_id: Optional[str]
    ) -> Optional[PromptTemplate]:
        """
        Query a single prompt template by type, scope, and user_id.

        Args:
            prompt_type: Type of prompt (agent_system, agent_query, index_graph, etc.)
            scope: 'user' or 'system'
            user_id: User ID (required for scope='user', None for scope='system')

        Returns:
            PromptTemplate instance or None
        """

        async def _query(session):
            stmt = select(PromptTemplate).where(
                PromptTemplate.prompt_type == prompt_type,
                PromptTemplate.scope == scope,
                PromptTemplate.gmt_deleted.is_(None),
            )

            if scope == "user":
                stmt = stmt.where(PromptTemplate.user_id == user_id)
            else:
                stmt = stmt.where(PromptTemplate.user_id.is_(None))

            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_user_prompt_templates(self, user_id: str) -> List[PromptTemplate]:
        """
        Query all prompt templates for a specific user.

        Args:
            user_id: User ID

        Returns:
            List of PromptTemplate instances
        """

        async def _query(session):
            stmt = (
                select(PromptTemplate)
                .where(
                    PromptTemplate.scope == "user",
                    PromptTemplate.user_id == user_id,
                    PromptTemplate.gmt_deleted.is_(None),
                )
                .order_by(PromptTemplate.prompt_type)
            )

            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_system_prompt_templates(self) -> List[PromptTemplate]:
        """
        Query all system default prompt templates.

        Returns:
            List of PromptTemplate instances
        """

        async def _query(session):
            stmt = (
                select(PromptTemplate)
                .where(PromptTemplate.scope == "system", PromptTemplate.gmt_deleted.is_(None))
                .order_by(PromptTemplate.prompt_type)
            )

            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def create_or_update_prompt_template(
        self,
        prompt_type: str,
        scope: str,
        user_id: Optional[str],
        content: str,
        description: Optional[str] = None,
    ) -> PromptTemplate:
        """
        Create or update a prompt template.

        Args:
            prompt_type: Type of prompt
            scope: 'user' or 'system'
            user_id: User ID (required for scope='user')
            content: Prompt content
            description: Optional description

        Returns:
            PromptTemplate instance
        """

        async def _operation(session):
            stmt = select(PromptTemplate).where(
                PromptTemplate.prompt_type == prompt_type,
                PromptTemplate.scope == scope,
                PromptTemplate.gmt_deleted.is_(None),
            )

            if scope == "user":
                stmt = stmt.where(PromptTemplate.user_id == user_id)
            else:
                stmt = stmt.where(PromptTemplate.user_id.is_(None))

            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                instance.content = content
                if description is not None:
                    instance.description = description
                instance.gmt_updated = utc_now()
            else:
                instance = PromptTemplate(
                    prompt_type=prompt_type,
                    scope=scope,
                    user_id=user_id,
                    content=content,
                    description=description,
                )

            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def delete_prompt_template(self, prompt_type: str, scope: str, user_id: Optional[str]) -> bool:
        """
        Soft delete a prompt template.

        Args:
            prompt_type: Type of prompt
            scope: 'user' or 'system'
            user_id: User ID (required for scope='user')

        Returns:
            True if deleted, False if not found
        """

        async def _operation(session):
            stmt = select(PromptTemplate).where(
                PromptTemplate.prompt_type == prompt_type,
                PromptTemplate.scope == scope,
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
