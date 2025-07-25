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

import logging
import secrets
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket
from fastapi_users import BaseUserManager, FastAPIUsers
from fastapi_users.authentication import AuthenticationBackend, CookieTransport, JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase

from aperag.config import AsyncSessionDep, settings
from aperag.db.models import ApiKey, ApiKeyStatus, Invitation, Role, User
from aperag.db.ops import async_db_ops
from aperag.schema import view_models
from aperag.utils.audit_decorator import audit
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)

# --- fastapi-users Implementation ---

COOKIE_MAX_AGE = 86400


class UserManager(BaseUserManager[User, str]):
    reset_password_token_secret = "SECRET"
    verification_token_secret = "SECRET"

    def parse_id(self, value: any) -> str:
        """Parse ID from any type to str"""
        if isinstance(value, str):
            return value
        return str(value)


# JWT Strategy
SECRET = "SECRET"  # TODO: Use configuration


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=86400)


# Transport methods
cookie_transport = CookieTransport(cookie_name="session", cookie_max_age=COOKIE_MAX_AGE)

# Authentication backends
cookie_backend = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)


# User Database dependency
async def get_user_db(session: AsyncSessionDep):
    yield SQLAlchemyUserDatabase(session, User)


# UserManager dependency
async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


# FastAPI Users instance
fastapi_users = FastAPIUsers[User, str](
    get_user_manager,
    [cookie_backend],
)


async def authenticate_websocket_user(websocket: WebSocket, user_manager: UserManager) -> Optional[str]:
    """Authenticate WebSocket connection using session cookie

    Returns:
        str: User ID if authenticated, None otherwise
    """
    try:
        # Extract cookies from WebSocket headers
        cookies_header = None

        # Try different ways to access headers
        if hasattr(websocket, "headers"):
            # WebSocket headers might be a mapping (dict-like)
            if hasattr(websocket.headers, "get"):
                cookie_value = websocket.headers.get("cookie") or websocket.headers.get(b"cookie")
                if cookie_value:
                    cookies_header = cookie_value.decode() if isinstance(cookie_value, bytes) else cookie_value
            else:
                # WebSocket headers might be an iterable of tuples/pairs
                try:
                    for header_item in websocket.headers:
                        if isinstance(header_item, (list, tuple)) and len(header_item) >= 2:
                            name, value = header_item[0], header_item[1]
                            if name == b"cookie" or name == "cookie":
                                cookies_header = value.decode() if isinstance(value, bytes) else value
                                break
                except (TypeError, ValueError):
                    # If iteration fails, headers format is unexpected
                    logger.debug("WebSocket headers format not supported for authentication")
                    pass

        if not cookies_header:
            logger.debug("No cookies found in WebSocket headers")
            return None

        # Parse cookies to find session cookie
        session_token = None
        for cookie in cookies_header.split(";"):
            cookie = cookie.strip()
            if cookie.startswith("session="):
                session_token = cookie.split("=", 1)[1]
                break

        if not session_token:
            logger.debug("No session cookie found")
            return None

        logger.debug(f"Found session token: {session_token[:20]}...")

        # Verify JWT token using the same strategy as HTTP authentication
        jwt_strategy = get_jwt_strategy()

        # Manually decode and verify the JWT token
        try:
            user_data = await jwt_strategy.read_token(session_token, user_manager)
            if user_data:
                logger.debug(f"Successfully authenticated user from WebSocket: {user_data.id}")
                return str(user_data.id)
            else:
                logger.debug("JWT token validation returned no user data")
                return None
        except Exception as e:
            logger.debug(f"WebSocket JWT verification failed: {e}")
            return None

    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        return None

    return None


# API Key Authentication
async def authenticate_api_key(request: Request, session: AsyncSessionDep) -> Optional[User]:
    """Authenticate using API Key from Authorization header"""
    from sqlalchemy import select

    authorization: str = request.headers.get("Authorization")
    if not authorization:
        return None

    try:
        scheme, credentials = authorization.split()
        if scheme.lower() != "bearer":
            return None
    except ValueError:
        return None

    # Query API key
    result = await session.execute(
        select(ApiKey).where(
            ApiKey.key == credentials, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
        )
    )
    api_key = result.scalars().first()

    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Get user by username
    result = await session.execute(
        select(User).where(User.id == api_key.user, User.is_active.is_(True), User.gmt_deleted.is_(None))
    )
    user = result.scalars().first()

    if user:
        # Update last used timestamp
        await api_key.update_last_used(session)
        # Mark the authentication method for debugging purposes
        user._auth_method = "api_key"
        user._api_key_id = api_key.id

    return user


# Authentication dependency, writes to request.state.user_id
async def current_user(
    request: Request, session: AsyncSessionDep, user: User = Depends(fastapi_users.current_user(optional=True))
) -> Optional[User]:
    """Get current user from JWT/Cookie or API Key and write to request.state.user_id"""
    # First try API Key authentication
    api_user = await authenticate_api_key(request, session)
    if api_user:
        request.state.user_id = api_user.id
        request.state.username = api_user.username
        return api_user

    # Then try JWT/Cookie authentication
    if user:
        request.state.user_id = user.id
        request.state.username = user.username
        return user

    raise HTTPException(status_code=401, detail="Unauthorized")


async def get_current_active_user(
    request: Request, session: AsyncSessionDep, user: Optional[User] = Depends(current_user)
) -> User:
    """Get current active user, raise 401 if not authenticated"""
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


async def get_current_admin(session: AsyncSessionDep, user: User = Depends(get_current_active_user)) -> User:
    """Get current admin user"""
    if user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Only admin members can perform this action")
    return user


router = APIRouter()

# --- API Implementation ---


@router.post("/invite", tags=["invitations"])
@audit(resource_type="invitation", api_name="CreateInvitation")
async def create_invitation_view(
    request: Request,
    data: view_models.InvitationCreate,
    session: AsyncSessionDep,
    user: User = Depends(get_current_admin),
) -> view_models.Invitation:
    # Check if user already exists
    from sqlalchemy import select

    result = await session.execute(select(User).where((User.username == data.username) | (User.email == data.email)))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="User with this email or username already exists")
    token = secrets.token_urlsafe(32)
    expires_at = utc_now() + timedelta(days=7)
    invitation = Invitation(
        email=data.email,
        token=token,
        created_by=str(user.id),
        created_at=utc_now(),
        role=data.role,
        expires_at=expires_at,
        is_used=False,
    )
    session.add(invitation)
    await session.commit()
    return view_models.Invitation(
        email=invitation.email,
        token=token,
        created_by=user.id,
        created_at=invitation.created_at.isoformat(),
        is_valid=invitation.is_valid(),
        role=invitation.role,
        expires_at=invitation.expires_at.isoformat(),
    )


@router.get("/invitations", tags=["invitations"])
async def list_invitations_view(
    session: AsyncSessionDep, user: User = Depends(current_user)
) -> view_models.InvitationList:
    from sqlalchemy import select

    if user.role != Role.ADMIN:
        result = await session.execute(select(Invitation).where(Invitation.created_by == user.id))
    else:
        result = await session.execute(select(Invitation))

    invitations = []
    for invitation in result.scalars():
        invitations.append(
            view_models.Invitation(
                email=invitation.email,
                token=invitation.token,
                created_by=invitation.created_by,
                created_at=invitation.created_at.isoformat(),
                is_valid=invitation.is_valid(),
                used_at=invitation.used_at.isoformat() if invitation.used_at else None,
                role=invitation.role,
                expires_at=invitation.expires_at.isoformat() if invitation.expires_at else None,
            )
        )
    return view_models.InvitationList(items=invitations)


@router.post("/register", tags=["auth"])
@audit(resource_type="user", api_name="RegisterUser")
async def register_view(
    request: Request,
    data: view_models.Register,
    session: AsyncSessionDep,
    user_manager: UserManager = Depends(get_user_manager),
) -> view_models.User:
    from sqlalchemy import select

    is_first_user = not await async_db_ops.query_first_user_exists()
    need_invitation = settings.register_mode == "invitation" and not is_first_user
    invitation = None
    if need_invitation:
        if not data.token:
            raise HTTPException(status_code=400, detail="Invitation token is required")
        if not data.email:
            raise HTTPException(status_code=400, detail="Email is required when using invitation")

        result = await session.execute(select(Invitation).where(Invitation.token == data.token))
        invitation = result.scalars().first()
        if not invitation or not invitation.is_valid():
            raise HTTPException(status_code=400, detail="Invalid or expired invitation")
        if invitation.email != data.email:
            raise HTTPException(status_code=400, detail="Email does not match invitation")

    # Check if user already exists
    result = await session.execute(select(User).where(User.username == data.username))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Username already exists")

    # Only check email uniqueness if email is provided
    if data.email:
        result = await session.execute(select(User).where(User.email == data.email))
        if result.scalars().first():
            raise HTTPException(status_code=400, detail="Email already exists")

    # Create user using fastapi-users
    user_create = {
        "username": data.username,
        "email": data.email,
        "password": data.password,
        "role": invitation.role if invitation else Role.ADMIN if is_first_user else Role.RO,
        "is_active": True,
        "is_verified": True,
        "date_joined": utc_now(),
    }

    user = User(**user_create)
    user.hashed_password = user_manager.password_helper.hash(data.password)
    session.add(user)
    await session.commit()
    await session.refresh(user)

    if invitation:
        invitation.is_used = True
        invitation.used_at = utc_now()
        session.add(invitation)
        await session.commit()

    # Create default API key and bot for the new user
    try:
        from aperag.db.models import BotType
        from aperag.schema.view_models import BotCreate
        from aperag.service.bot_service import bot_service

        # Create a system API key for the user (not visible to user)
        await async_db_ops.create_api_key(user=str(user.id), description="aperag", is_system=True)

        # Create a default bot for the user
        bot_create = BotCreate(
            title="Default Agent Bot",
            type=BotType.AGENT,
            description="Default agent bot created on registration.",
            collection_ids=[],
        )
        await bot_service.create_bot(user=str(user.id), bot_in=bot_create)

        logger.info(f"Created default bot and api key for user {user.username} ({user.id})")
    except Exception as e:
        logger.error(f"Failed to create default bot and api key for user {user.username} ({user.id}): {e}")

    return view_models.User(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        date_joined=user.date_joined.isoformat(),
    )


@router.post("/login", tags=["auth"])
async def login_view(
    request: Request,
    response: Response,
    data: view_models.Login,
    session: AsyncSessionDep,
    user_manager: UserManager = Depends(get_user_manager),
) -> view_models.User:
    from sqlalchemy import select

    result = await session.execute(select(User).where(User.username == data.username))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    # Use fastapi-users correct password verification method
    verified, updated_password_hash = user_manager.password_helper.verify_and_update(
        data.password, user.hashed_password
    )
    if not verified:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    if updated_password_hash:
        user.hashed_password = updated_password_hash
        session.add(user)
        await session.commit()

    # Generate JWT token and set cookie
    strategy = get_jwt_strategy()
    token = await strategy.write_token(user)

    # Set cookie
    response.set_cookie(key="session", value=token, max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax")

    return view_models.User(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        date_joined=user.date_joined.isoformat(),
    )


@router.post("/logout", tags=["auth"])
async def logout_view(response: Response):
    # Clear authentication cookie
    response.delete_cookie(key="session")
    return {"success": True}


@router.get("/user", tags=["users"])
async def get_user_view(request: Request, session: AsyncSessionDep, user: Optional[User] = Depends(current_user)):
    """Get user info, return 401 if not authenticated"""
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return view_models.User(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        date_joined=user.date_joined.isoformat(),
    )


@router.get("/users", tags=["users"])
async def list_users_view(
    session: AsyncSessionDep, user: Optional[User] = Depends(current_user)
) -> view_models.UserList:
    from sqlalchemy import select

    if user.role == Role.ADMIN:
        result = await session.execute(select(User))
    else:
        result = await session.execute(select(User).where(User.id == user.id))

    users = [
        view_models.User(
            id=str(u.id),
            username=u.username,
            email=u.email,
            role=u.role,
            is_active=u.is_active,
            date_joined=u.date_joined.isoformat(),
        )
        for u in result.scalars()
    ]
    return view_models.UserList(items=users)


@router.post("/change-password", tags=["auth"])
@audit(resource_type="user", api_name="ChangePassword")
async def change_password_view(
    request: Request,
    data: view_models.ChangePassword,
    session: AsyncSessionDep,
    user_manager: UserManager = Depends(get_user_manager),
):
    user = await async_db_ops.query_user_by_username(data.username)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    # Verify old password - use correct fastapi-users API
    verified, _ = user_manager.password_helper.verify_and_update(data.old_password, user.hashed_password)
    if not verified:
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    # Set new password
    user.hashed_password = user_manager.password_helper.hash(data.new_password)
    session.add(user)
    await session.commit()

    return view_models.User(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        date_joined=user.date_joined.isoformat(),
    )


@router.delete("/users/{user_id}", tags=["users"])
@audit(resource_type="user", api_name="DeleteUser")
async def delete_user_view(
    request: Request, user_id: str, session: AsyncSessionDep, user: User = Depends(get_current_admin)
):
    from sqlalchemy import select

    result = await session.execute(select(User).where(User.id == user_id))
    target = result.scalars().first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    admin_count = await async_db_ops.query_admin_count()
    if target.role == Role.ADMIN and admin_count <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the last admin user")
    if target.id == user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    await async_db_ops.delete_user(session, target)
    return {"message": "User deleted successfully"}
