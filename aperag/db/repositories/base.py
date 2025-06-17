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

from typing import Optional, Protocol

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, sessionmaker

from aperag.config import async_engine, get_async_session, get_sync_session, sync_engine


class SyncRepositoryProtocol(Protocol):
    def _get_session(self) -> Session: ...

    def _execute_query(self, query_func): ...

    def _execute_transaction(self, operation): ...


class AsyncRepositoryProtocol(Protocol):
    async def _execute_query(self, query_func): ...

    async def execute_with_transaction(self, operation): ...


class SyncBaseRepository(SyncRepositoryProtocol):
    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def _get_session(self):
        if self._session:
            return self._session
        else:
            sync_session = sessionmaker(sync_engine, class_=Session, expire_on_commit=False)
            with sync_session() as session:
                return session

    def _execute_query(self, query_func):
        if self._session:
            return query_func(self._session)
        else:
            sync_session = sessionmaker(sync_engine, class_=Session, expire_on_commit=False)
            with sync_session() as session:
                return query_func(session)

    def _execute_transaction(self, operation):
        if self._session:
            # Use provided session, caller manages transaction
            return operation(self._session)
        else:
            # Create new session and manage transaction lifecycle
            for session in get_sync_session():
                try:
                    result = operation(session)
                    session.commit()
                    return result
                except Exception:
                    session.rollback()
                    raise


class AsyncBaseRepository(AsyncRepositoryProtocol):
    """Database operations manager that handles session management"""

    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def _execute_query(self, query_func):
        """Execute a read-only query with proper session management

        This method is designed for read-only database operations (SELECT queries)
        and provides automatic session lifecycle management. It follows the pattern
        of accepting a query function that encapsulates the database operation.

        Key benefits:
        1. Automatic session creation and cleanup for read operations
        2. Consistent session management across all query methods
        3. Support for both injected sessions and auto-created sessions
        4. Simplified code for read-only operations

        Usage pattern for read operations:
        1. Define an inner async function that takes a session parameter
        2. Write your SELECT query logic inside the inner function
        3. Pass the inner function to this method
        4. Session lifecycle is handled automatically

        Example:
            async def query_user(self, user_id: str):
                async def _query(session):
                    stmt = select(User).where(User.id == user_id)
                    result = await session.execute(stmt)
                    return result.scalars().first()
                return await self._execute_query(_query)
        """
        if self._session:
            return await query_func(self._session)
        else:
            async_session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
            async with async_session() as session:
                return await query_func(session)

    async def execute_with_transaction(self, operation):
        """Execute multiple database operations in a single transaction

        This method is used when you need to perform multiple database operations
        that must all succeed or all fail together. Individual DatabaseOps methods
        will automatically detect that they're running within a managed transaction
        and will only flush (not commit) their changes.

        Design philosophy:
        - Single operations: Use DatabaseOps methods directly, they handle their own transactions
        - Multiple operations: Use this method to wrap them in a single transaction

        Usage pattern:
        1. Define an operation function that takes a session parameter
        2. Create DatabaseOps instance with the session
        3. Perform multiple database operations within the function
        4. All operations will be executed in a single transaction
        5. Automatic commit on success, rollback on error
        """
        if self._session:
            # Use provided session, caller manages transaction
            return await operation(self._session)
        else:
            # Create new session and manage transaction lifecycle
            async for session in get_async_session():
                try:
                    result = await operation(session)
                    await session.commit()
                    return result
                except Exception:
                    await session.rollback()
                    raise
