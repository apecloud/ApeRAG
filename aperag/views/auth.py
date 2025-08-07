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
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket
from fastapi_users import BaseUserManager, FastAPIUsers
from fastapi_users.authentication import (
    AuthenticationBackend,
    CookieTransport,
    JWTStrategy,
)
from fastapi_users.db import SQLAlchemyUserDatabase
from httpx_oauth.clients.github import GitHubOAuth2
from httpx_oauth.clients.google import GoogleOAuth2
from fastapi_users.router.oauth import get_oauth_router

from aperag.config import AsyncSessionDep, settings
from aperag.db.models import ApiKey, ApiKeyStatus, Invitation, OAuthAccount, Role, User
from aperag.db.ops import async_db_ops
from aperag.schema import view_models
from aperag.utils.audit_decorator import audit
from aperag.utils.utils import utc_now
from aperag.views.utils import is_github_oauth_enabled, is_google_oauth_enabled

logger = logging.getLogger(__name__)

# --- fastapi-users Implementation ---

COOKIE_MAX_AGE = 86400


class UserManager(BaseUserManager[User, str]):
    reset_password_token_secret = settings.jwt_secret
    verification_token_secret = settings.jwt_secret

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        """
        Set the first registered user as an admin and initialize user resources.
        This works for both regular and OAuth registration.
        """
        user_count = await async_db_ops.query_user_count()
        if user_count == 1 and user.role != Role.ADMIN:
            user.role = Role.ADMIN
            self.user_db.session.add(user)
            await self.user_db.session.commit()
            await self.user_db.session.refresh(user)

        # For GitHub OAuth users, fetch username from GitHub API
        await self._fetch_github_username_if_needed(user)

        # Initialize user resources for all new users (including OAuth users)
        try:
            from aperag.db.models import BotType
            from aperag.schema.view_models import BotCreate
            from aperag.service.bot_service import bot_service
            from aperag.service.quota_service import quota_service

            # Initialize user quotas first
            await quota_service.initialize_user_quotas(str(user.id))

            # Create a system API key for the user (not visible to user)
            await async_db_ops.create_api_key(user=str(user.id), description="system", is_system=True)
            # Create a normal API key for the user (visible to user)
            await async_db_ops.create_api_key(user=str(user.id), description="default", is_system=False)

            # Create a default bot for the user (skip quota check for system bot)
            bot_create = BotCreate(
                title="Default Agent Bot",
                type=BotType.AGENT,
                description="Default agent bot created on registration.",
                collection_ids=[],
            )
            await bot_service.create_bot(user=str(user.id), bot_in=bot_create, skip_quota_check=True)

            logger.info(f"Initialized resources for user {user.username or user.email} ({user.id})")
        except Exception as e:
            logger.error(f"Failed to initialize resources for user {user.username or user.email} ({user.id}): {e}")

    async def _fetch_github_username_if_needed(self, user: User):
        """
        For GitHub OAuth users, fetch username from GitHub API using account_id
        """
        try:
            # Check if user has GitHub OAuth account
            github_oauth_account = None
            for oauth_account in user.oauth_accounts:
                if oauth_account.oauth_name == "github":
                    github_oauth_account = oauth_account
                    break
            
            if not github_oauth_account:
                return  # Not a GitHub OAuth user
            
            if user.username:
                return  # Username already set
            
            # Fetch username from GitHub API
            import httpx
            
            github_user_id = github_oauth_account.account_id
            github_api_url = f"https://api.github.com/user/{github_user_id}"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(github_api_url)
                if response.status_code == 200:
                    github_user_data = response.json()
                    github_username = github_user_data.get("login")
                    
                    if github_username:
                        user.username = github_username
                        self.user_db.session.add(user)
                        await self.user_db.session.commit()
                        await self.user_db.session.refresh(user)
                        logger.info(f"Updated GitHub user {user.id} with username: {github_username}")
                else:
                    logger.warning(f"Failed to fetch GitHub user data for user {user.id}: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to fetch GitHub username for user {user.id}: {e}")

    def parse_id(self, value: any) -> str:
        """Parse ID from any type to str"""
        if isinstance(value, str):
            return value
        return str(value)


# JWT Strategy
def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=settings.jwt_secret, lifetime_seconds=COOKIE_MAX_AGE)


# Transport methods
cookie_transport = CookieTransport(
    cookie_name="session", 
    cookie_max_age=COOKIE_MAX_AGE,
    cookie_secure=False,  # Set to False for HTTP development environment
    cookie_httponly=True,
    cookie_samesite="lax"
)

# Authentication backend
auth_backend = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# --- User Database and Manager Dependencies ---
async def get_user_db(session: AsyncSessionDep):
    yield SQLAlchemyUserDatabase(session, User, OAuthAccount)


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


# --- FastAPI Users Instance ---
fastapi_users = FastAPIUsers[User, str](
    get_user_manager,
    [auth_backend],
)


# --- WebSocket Authentication ---
async def authenticate_websocket_user(websocket: WebSocket, user_manager: UserManager) -> Optional[str]:
    """Authenticate WebSocket connection using session cookie"""
    try:
        cookies_header = None
        if hasattr(websocket, "headers"):
            if hasattr(websocket.headers, "get"):
                cookie_value = websocket.headers.get("cookie") or websocket.headers.get(b"cookie")
                if cookie_value:
                    cookies_header = cookie_value.decode() if isinstance(cookie_value, bytes) else cookie_value
            else:
                try:
                    for name, value in websocket.headers:
                        if name == b"cookie" or name == "cookie":
                            cookies_header = value.decode() if isinstance(value, bytes) else value
                            break
                except (TypeError, ValueError):
                    logger.debug("WebSocket headers format not supported for authentication")
        if not cookies_header:
            logger.debug("No cookies found in WebSocket headers")
            return None
        session_token = None
        for cookie in cookies_header.split(";"):
            cookie = cookie.strip()
            if cookie.startswith("session="):
                session_token = cookie.split("=", 1)[1]
                break
        if not session_token:
            logger.debug("No session cookie found")
            return None
        jwt_strategy = get_jwt_strategy()
        user_data = await jwt_strategy.read_token(session_token, user_manager)
        if user_data:
            logger.debug(f"Successfully authenticated user from WebSocket: {user_data.id}")
            return str(user_data.id)
        else:
            logger.debug("JWT token validation returned no user data")
            return None
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        return None


# --- API Key Authentication ---
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
    result = await session.execute(
        select(ApiKey).where(
            ApiKey.key == credentials, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
        )
    )
    api_key = result.scalars().first()
    if not api_key:
        return None  # Don't raise, just return None to allow other auth methods
    result = await session.execute(
        select(User).where(User.id == api_key.user, User.is_active.is_(True), User.gmt_deleted.is_(None))
    )
    user = result.scalars().first()
    if user:
        await api_key.update_last_used(session)
        user._auth_method = "api_key"
        user._api_key_id = api_key.id
    return user


# --- Current User Dependency ---
async def current_user(
    request: Request, session: AsyncSessionDep, user: User = Depends(fastapi_users.current_user(optional=True))
) -> Optional[User]:
    """Get current user from JWT/Cookie, OAuth, or API Key."""
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


async def get_current_active_user(user: Optional[User] = Depends(current_user)) -> User:
    """Get current active user, raise 401 if not authenticated."""
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


async def get_current_admin(user: User = Depends(get_current_active_user)) -> User:
    """Get current admin user."""
    if user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Only admin members can perform this action")
    return user


# --- Router Setup ---
router = APIRouter()


# --- Conditional OAuth Routers ---
if is_google_oauth_enabled():
    google_oauth_client = GoogleOAuth2(
        settings.google_oauth_client_id, settings.google_oauth_client_secret
    )
    google_oauth_router = get_oauth_router(
        google_oauth_client,
        auth_backend,
        get_user_manager,
        settings.jwt_secret,
        redirect_url=settings.oauth_redirect_url,
        associate_by_email=True,
        is_verified_by_default=True,
    )
    router.include_router(google_oauth_router, prefix="/auth/google", tags=["auth"])

if is_github_oauth_enabled():
    github_oauth_client = GitHubOAuth2(
        settings.github_oauth_client_id, settings.github_oauth_client_secret
    )
    github_oauth_router = get_oauth_router(
        github_oauth_client,
        auth_backend,
        get_user_manager,
        settings.jwt_secret,
        redirect_url=settings.oauth_redirect_url,
        associate_by_email=True,
        is_verified_by_default=True,
    )
    router.include_router(github_oauth_router, prefix="/auth/github", tags=["auth"])




@router.post("/invite", tags=["invitations"])
@audit(resource_type="invitation", api_name="CreateInvitation")
async def create_invitation_view(
    request: Request,
    data: view_models.InvitationCreate,
    session: AsyncSessionDep,
    user: User = Depends(get_current_admin),
) -> view_models.Invitation:
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

    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if user.role != Role.ADMIN:
        result = await session.execute(select(Invitation).where(Invitation.created_by == str(user.id)))
    else:
        result = await session.execute(select(Invitation))
    invitations = []
    for invitation in result.unique().scalars():
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

    # Note: User resources (quotas, API keys, default bot) are now initialized 
    # in the on_after_register method which is called automatically by fastapi-users
    await user_manager.on_after_register(user, request)

    # Determine registration source
    registration_source = "local"  # Default to local registration
    if hasattr(user, 'oauth_accounts') and user.oauth_accounts:
        # If user has OAuth accounts, use the first one's provider name
        registration_source = user.oauth_accounts[0].oauth_name

    return view_models.User(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        date_joined=user.date_joined.isoformat(),
        registration_source=registration_source,
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
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Load user with oauth_accounts to determine registration source
    result = await session.execute(
        select(User).options(selectinload(User.oauth_accounts)).where(User.id == user.id)
    )
    user_with_oauth = result.scalars().first()
    
    # Determine registration source
    registration_source = "local"  # Default to local registration
    if user_with_oauth and user_with_oauth.oauth_accounts:
        # If user has OAuth accounts, use the first one's provider name
        registration_source = user_with_oauth.oauth_accounts[0].oauth_name

    return view_models.User(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        date_joined=user.date_joined.isoformat(),
        registration_source=registration_source,
    )


@router.get("/users", tags=["users"])
async def list_users_view(
    session: AsyncSessionDep, user: Optional[User] = Depends(current_user)
) -> view_models.UserList:
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    if user.role == Role.ADMIN:
        result = await session.execute(select(User).options(selectinload(User.oauth_accounts)))
    else:
        result = await session.execute(select(User).options(selectinload(User.oauth_accounts)).where(User.id == user.id))

    users = []
    for u in result.unique().scalars():
        # Determine registration source
        registration_source = "local"  # Default to local registration
        if u.oauth_accounts:
            # If user has OAuth accounts, use the first one's provider name
            registration_source = u.oauth_accounts[0].oauth_name
        
        users.append(
            view_models.User(
                id=str(u.id),
                username=u.username,
                email=u.email,
                role=u.role,
                is_active=u.is_active,
                date_joined=u.date_joined.isoformat(),
                registration_source=registration_source,
            )
        )
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
