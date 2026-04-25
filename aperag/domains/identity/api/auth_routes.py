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

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi_users.router.oauth import get_oauth_router
from httpx_oauth.clients.github import GitHubOAuth2
from httpx_oauth.clients.google import GoogleOAuth2

from aperag.config import AsyncSessionDep, settings
from aperag.db.ops import async_db_ops
from aperag.domains.identity import schemas as identity_schemas
from aperag.domains.identity.db.models import (
    Invitation,
    Role,
    User,
)
from aperag.domains.identity.service.auth_dependencies import (
    COOKIE_MAX_AGE,
    auth_backend,
    get_current_admin,
    get_jwt_strategy,
    get_user_manager,
    required_user,
)
from aperag.domains.identity.service.login_methods import is_github_oauth_enabled, is_google_oauth_enabled
from aperag.domains.identity.service.user_manager import UserManager
from aperag.utils.audit_decorator import audit
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)

# --- Router Setup ---
router = APIRouter()


# --- Conditional OAuth Routers ---
if is_google_oauth_enabled():
    google_oauth_client = GoogleOAuth2(settings.google_oauth_client_id, settings.google_oauth_client_secret)
    google_oauth_router = get_oauth_router(
        google_oauth_client,
        auth_backend,
        get_user_manager,
        settings.jwt_secret,
        redirect_url=settings.oauth_redirect_url + "/google",
        associate_by_email=True,
        is_verified_by_default=True,
    )
    router.include_router(google_oauth_router, prefix="/google", tags=["auth"])

if is_github_oauth_enabled():
    github_oauth_client = GitHubOAuth2(settings.github_oauth_client_id, settings.github_oauth_client_secret)
    github_oauth_router = get_oauth_router(
        github_oauth_client,
        auth_backend,
        get_user_manager,
        settings.jwt_secret,
        redirect_url=settings.oauth_redirect_url + "/github",
        associate_by_email=True,
        is_verified_by_default=True,
    )
    router.include_router(github_oauth_router, prefix="/github", tags=["auth"])


@router.post("/invite", tags=["invitations"])
@audit(resource_type="invitation", api_name="CreateInvitation")
async def create_invitation_view(
    request: Request,
    data: identity_schemas.InvitationCreate,
    session: AsyncSessionDep,
    user: User = Depends(get_current_admin),
) -> identity_schemas.Invitation:
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
    return identity_schemas.Invitation(
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
    session: AsyncSessionDep, user: User = Depends(required_user)
) -> identity_schemas.InvitationList:
    from sqlalchemy import select

    if user.role != Role.ADMIN:
        result = await session.execute(select(Invitation).where(Invitation.created_by == str(user.id)))
    else:
        result = await session.execute(select(Invitation))
    invitations = []
    for invitation in result.unique().scalars():
        invitations.append(
            identity_schemas.Invitation(
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
    return identity_schemas.InvitationList(items=invitations)


@router.post("/register", tags=["auth"])
@audit(resource_type="user", api_name="RegisterUser")
async def register_view(
    request: Request,
    data: identity_schemas.Register,
    session: AsyncSessionDep,
    user_manager: UserManager = Depends(get_user_manager),
) -> identity_schemas.User:
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
    if hasattr(user, "oauth_accounts") and user.oauth_accounts:
        # If user has OAuth accounts, use the first one's provider name
        registration_source = user.oauth_accounts[0].oauth_name

    return identity_schemas.User(
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
    data: identity_schemas.Login,
    session: AsyncSessionDep,
    user_manager: UserManager = Depends(get_user_manager),
) -> identity_schemas.User:
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

    return identity_schemas.User(
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
async def get_user_view(request: Request, session: AsyncSessionDep, user: Optional[User] = Depends(required_user)):
    """Get user info, return 401 if not authenticated"""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Load user with oauth_accounts to determine registration source
    result = await session.execute(select(User).options(selectinload(User.oauth_accounts)).where(User.id == user.id))
    user_with_oauth = result.scalars().first()

    # Determine registration source
    registration_source = "local"  # Default to local registration
    if user_with_oauth and user_with_oauth.oauth_accounts:
        # If user has OAuth accounts, use the first one's provider name
        registration_source = user_with_oauth.oauth_accounts[0].oauth_name

    return identity_schemas.User(
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
    session: AsyncSessionDep, user: Optional[User] = Depends(required_user)
) -> identity_schemas.UserList:
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    if user.role == Role.ADMIN:
        result = await session.execute(select(User).options(selectinload(User.oauth_accounts)))
    else:
        result = await session.execute(
            select(User).options(selectinload(User.oauth_accounts)).where(User.id == user.id)
        )

    users = []
    for u in result.unique().scalars():
        # Determine registration source
        registration_source = "local"  # Default to local registration
        if u.oauth_accounts:
            # If user has OAuth accounts, use the first one's provider name
            registration_source = u.oauth_accounts[0].oauth_name

        users.append(
            identity_schemas.User(
                id=str(u.id),
                username=u.username,
                email=u.email,
                role=u.role,
                is_active=u.is_active,
                date_joined=u.date_joined.isoformat(),
                registration_source=registration_source,
            )
        )
    return identity_schemas.UserList(items=users)


@router.post("/change-password", tags=["auth"])
@audit(resource_type="user", api_name="ChangePassword")
async def change_password_view(
    request: Request,
    data: identity_schemas.ChangePassword,
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

    return identity_schemas.User(
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
