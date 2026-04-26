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

from typing import Optional

from sqlalchemy import select

from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.domains.agent_runtime.db.models import (
    AgentTimelineEvent,
    AgentTurn,
)
from aperag.utils.utils import utc_now


class AsyncAgentRuntimeRepositoryMixin(AsyncRepositoryProtocol):
    async def query_agent_turn_by_id(self, turn_id: str) -> Optional[AgentTurn]:
        async def _query(session):
            stmt = select(AgentTurn).where(AgentTurn.id == turn_id)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_agent_turn(self, user: str, chat_id: str, turn_id: str) -> Optional[AgentTurn]:
        async def _query(session):
            stmt = select(AgentTurn).where(
                AgentTurn.id == turn_id, AgentTurn.chat_id == chat_id, AgentTurn.user == user
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_agent_turns(self, user: str, chat_id: str) -> list[AgentTurn]:
        async def _query(session):
            stmt = (
                select(AgentTurn)
                .where(AgentTurn.chat_id == chat_id, AgentTurn.user == user)
                .order_by(AgentTurn.gmt_created.asc())
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_agent_turn_by_idempotency(
        self, user: str, chat_id: str, client_idempotency_key: str
    ) -> Optional[AgentTurn]:
        async def _query(session):
            stmt = select(AgentTurn).where(
                AgentTurn.chat_id == chat_id,
                AgentTurn.user == user,
                AgentTurn.client_idempotency_key == client_idempotency_key,
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_recent_agent_turns(self, user: str, chat_id: str, limit: int = 8) -> list[AgentTurn]:
        async def _query(session):
            stmt = (
                select(AgentTurn)
                .where(AgentTurn.chat_id == chat_id, AgentTurn.user == user)
                .order_by(AgentTurn.gmt_created.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(reversed(result.scalars().all()))

        return await self._execute_query(_query)

    async def create_agent_turn(
        self,
        *,
        chat_id: str,
        user: str,
        bot_id: str,
        request_id: str,
        client_idempotency_key: str,
        input_text: str,
        model_profile: dict,
    ) -> AgentTurn:
        async def _operation(session):
            instance = AgentTurn(
                chat_id=chat_id,
                user=user,
                bot_id=bot_id,
                request_id=request_id,
                client_idempotency_key=client_idempotency_key,
                input_text=input_text,
                model_profile=model_profile,
            )
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def update_agent_turn(
        self,
        turn_id: str,
        **fields,
    ) -> Optional[AgentTurn]:
        async def _operation(session):
            stmt = select(AgentTurn).where(AgentTurn.id == turn_id)
            result = await session.execute(stmt)
            instance = result.scalars().first()
            if not instance:
                return None

            for key, value in fields.items():
                setattr(instance, key, value)
            instance.gmt_updated = utc_now()

            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def create_agent_timeline_event(
        self,
        *,
        turn_id: str,
        sequence: int,
        timestamp,
        event_type: str,
        label: Optional[str],
        status: Optional[str],
        actor,
        data: dict,
    ) -> AgentTimelineEvent:
        async def _operation(session):
            instance = AgentTimelineEvent(
                turn_id=turn_id,
                sequence=sequence,
                timestamp=timestamp,
                type=event_type,
                label=label,
                status=status,
                actor=actor,
                data=data,
            )
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def query_agent_timeline_events(
        self,
        turn_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 500,
    ) -> list[AgentTimelineEvent]:
        async def _query(session):
            stmt = (
                select(AgentTimelineEvent)
                .where(AgentTimelineEvent.turn_id == turn_id, AgentTimelineEvent.sequence > after_sequence)
                .order_by(AgentTimelineEvent.sequence.asc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)
