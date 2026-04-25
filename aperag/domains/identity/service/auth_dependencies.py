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

from fastapi import Depends, HTTPException, Request
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import AuthenticationBackend, CookieTransport, JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy import select

from aperag.config import AsyncSessionDep, settings
from aperag.domains.governance.db.models import ApiKey, ApiKeyStatus
from aperag.domains.identity.db.models import OAuthAccount, Role, User
from aperag.domains.identity.service.user_manager import UserManager

COOKIE_MAX_AGE = 86400


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=settings.jwt_secret, lifetime_seconds=COOKIE_MAX_AGE)


cookie_transport = CookieTransport(
    cookie_name="session",
    cookie_max_age=COOKIE_MAX_AGE,
    cookie_secure=False,
    cookie_httponly=True,
    cookie_samesite="lax",
)

auth_backend = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)


async def get_user_db(session: AsyncSessionDep):
    yield SQLAlchemyUserDatabase(session, User, OAuthAccount)


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


fastapi_users = FastAPIUsers[User, str](
    get_user_manager,
    [auth_backend],
)


async def authenticate_api_key(request: Request, session: AsyncSessionDep) -> Optional[User]:
    authorization: str = request.headers.get("Authorization")
    if not authorization:
        return None
    try:
        scheme, credentials = authorization.split()
        if scheme.lower() != "bearer":
            return None
    except ValueError:
        return None
    result = await session.execute(
        select(ApiKey).where(
            ApiKey.key == credentials,
            ApiKey.status == ApiKeyStatus.ACTIVE,
            ApiKey.gmt_deleted.is_(None),
        )
    )
    api_key = result.scalars().first()
    if not api_key:
        return None
    result = await session.execute(
        select(User).where(
            User.id == api_key.user,
            User.is_active.is_(True),
            User.gmt_deleted.is_(None),
        )
    )
    user = result.scalars().first()
    if user:
        await api_key.update_last_used(session)
        user._auth_method = "api_key"
        user._api_key_id = api_key.id
    return user


async def optional_user(
    request: Request,
    session: AsyncSessionDep,
    user: User = Depends(fastapi_users.current_user(optional=True)),
) -> Optional[User]:
    if user:
        request.state.user_id = user.id
        request.state.username = user.username
        return user
    api_user = await authenticate_api_key(request, session)
    if api_user:
        request.state.user_id = api_user.id
        request.state.username = api_user.username
        return api_user
    return None


async def required_user(user: Optional[User] = Depends(optional_user)) -> User:
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


async def get_current_admin(user: User = Depends(required_user)) -> User:
    if user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Only admin members can perform this action")
    return user
