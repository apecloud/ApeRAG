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


from sqlalchemy import desc, func, select

from aperag.db.models import (
    Invitation,
    Role,
    User,
    UserQuota,
)
from aperag.db.repositories.base import AsyncRepositoryProtocol


class AsyncUserRepositoryMixin(AsyncRepositoryProtocol):
    async def query_user_quota(self, user: str, key: str):
        async def _query(session):
            stmt = select(UserQuota).where(UserQuota.user == user, UserQuota.key == key)
            result = await session.execute(stmt)
            uq = result.scalars().first()
            return uq.value if uq else None

        return await self._execute_query(_query)

    async def query_user_by_username(self, username: str):
        async def _query(session):
            stmt = select(User).where(User.username == username)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_user_by_email(self, email: str):
        async def _query(session):
            stmt = select(User).where(User.email == email)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_user_exists(self, username: str = None, email: str = None):
        async def _query(session):
            stmt = select(User)
            if username:
                stmt = stmt.where(User.username == username)
            if email:
                stmt = stmt.where(User.email == email)
            result = await session.execute(stmt)
            return result.scalars().first() is not None

        return await self._execute_query(_query)

    async def create_user(self, username: str, email: str, password: str, role: Role):
        async def _operation(session):
            user = User(username=username, email=email, password=password, role=role)
            session.add(user)
            await session.flush()
            await session.refresh(user)
            return user

        return await self.execute_with_transaction(_operation)

    async def delete_user(self, user: User):
        async def _operation(session):
            await session.delete(user)
            await session.flush()

        return await self.execute_with_transaction(_operation)

    async def query_invitation_by_token(self, token: str):
        async def _query(session):
            stmt = select(Invitation).where(Invitation.token == token)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def create_invitation(self, email: str, token: str, created_by: str, role: Role):
        async def _operation(session):
            invitation = Invitation(email=email, token=token, created_by=created_by, role=role)
            session.add(invitation)
            await session.flush()
            await session.refresh(invitation)
            return invitation

        return await self.execute_with_transaction(_operation)

    async def mark_invitation_used(self, invitation: Invitation):
        async def _operation(session):
            await invitation.use(session)

        return await self.execute_with_transaction(_operation)

    async def query_invitations(self):
        """Query all valid invitations (not used), ordered by created_at descending."""

        async def _query(session):
            stmt = select(Invitation).where(not Invitation.is_used).order_by(desc(Invitation.created_at))
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_admin_count(self):
        async def _query(session):
            stmt = select(func.count()).select_from(User).where(User.role == Role.ADMIN, User.gmt_deleted.is_(None))
            return await session.scalar(stmt)

        return await self._execute_query(_query)

    async def query_first_user_exists(self):
        async def _query(session):
            stmt = select(User).where(User.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            return result.scalars().first() is not None

        return await self._execute_query(_query)
