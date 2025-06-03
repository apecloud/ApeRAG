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

import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi_users import BaseUserManager, FastAPIUsers, IntegerIDMixin
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, CookieTransport, JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase

from aperag.config import SessionDep, settings
from aperag.db.models import Invitation, Role, User
from aperag.db.ops import delete_user as db_delete_user
from aperag.db.ops import query_admin_count, query_first_user_exists
from aperag.schema import view_models

# --- fastapi-users Implementation ---


class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    reset_password_token_secret = "SECRET"
    verification_token_secret = "SECRET"

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        pass


# JWT Strategy
SECRET = "SECRET"  # TODO: Use configuration


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)


# Transport methods
cookie_transport = CookieTransport(cookie_name="session", cookie_max_age=3600)
bearer_transport = BearerTransport(tokenUrl="/login")

# Authentication backends
jwt_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)
cookie_backend = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)


# User Database dependency
async def get_user_db(session: SessionDep):
    yield SQLAlchemyUserDatabase(session, User)


# UserManager dependency
async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


# FastAPI Users instance
fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [jwt_backend, cookie_backend],
)


# Authentication dependency, writes to request.state.user_id
async def get_current_user_with_state(
    request: Request, user: User = Depends(fastapi_users.current_user(optional=True))
) -> Optional[User]:
    """Get current user and write to request.state.user_id"""
    if user:
        request.state.user_id = user.id
    return user


async def get_current_active_user(
    request: Request, user: Optional[User] = Depends(get_current_user_with_state)
) -> User:
    """Get current active user, raise 401 if not authenticated"""
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


async def get_current_admin(user: User = Depends(get_current_active_user)) -> User:
    """Get current admin user"""
    if user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Only admin members can perform this action")
    return user


router = APIRouter()

# --- API Implementation ---


@router.post("/invite")
async def create_invitation_view(
    data: view_models.InvitationCreate, session: SessionDep, user: User = Depends(get_current_admin)
) -> view_models.Invitation:
    # Check if user already exists
    from sqlmodel import select

    result = await session.execute(select(User).where((User.username == data.username) | (User.email == data.email)))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="User with this email or username already exists")
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=7)
    invitation = Invitation(
        email=data.email,
        token=token,
        created_by=str(user.id),
        created_at=datetime.utcnow(),
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


@router.get("/invitations")
async def list_invitations_view(
    session: SessionDep, user: User = Depends(get_current_admin)
) -> view_models.InvitationList:
    from sqlmodel import select

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


@router.post("/register")
async def register_view(
    data: view_models.Register, session: SessionDep, user_manager: UserManager = Depends(get_user_manager)
) -> view_models.User:
    from sqlmodel import select

    is_first_user = not await query_first_user_exists(session)
    need_invitation = settings.register_mode == "invitation" and not is_first_user
    invitation = None
    if need_invitation:
        result = await session.execute(select(Invitation).where(Invitation.token == data.token))
        invitation = result.scalars().first()
        if not invitation or not invitation.is_valid() or invitation.email != data.email:
            raise HTTPException(status_code=400, detail="Invalid or expired invitation")

    # Check if user already exists
    result = await session.execute(select(User).where(User.username == data.username))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Username already exists")
    result = await session.execute(select(User).where(User.email == data.email))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Email already exists")

    # Create user using fastapi-users
    user_create = {
        "username": data.username,
        "email": data.email,
        "password": data.password,
        "role": invitation.role if invitation else Role.ADMIN,
        "is_active": True,
        "is_verified": True,
        "date_joined": datetime.utcnow(),
    }

    user = User(**user_create)
    user.hashed_password = user_manager.password_helper.hash(data.password)
    session.add(user)
    await session.commit()
    await session.refresh(user)

    if invitation:
        invitation.is_used = True
        invitation.used_at = datetime.utcnow()
        session.add(invitation)
        await session.commit()

    return view_models.User(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        date_joined=user.date_joined.isoformat(),
    )


@router.post("/login")
async def login_view(
    request: Request,
    response: Response,
    data: view_models.Login,
    session: SessionDep,
    user_manager: UserManager = Depends(get_user_manager),
) -> view_models.User:
    from sqlmodel import select

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

    # If password hash is updated, save to database
    if updated_password_hash is not None:
        user.hashed_password = updated_password_hash
        session.add(user)
        await session.commit()

    # Generate JWT token and set cookie
    strategy = get_jwt_strategy()
    token = await strategy.write_token(user)

    # Set cookie
    response.set_cookie(key="session", value=token, max_age=3600, httponly=True, samesite="lax")

    return view_models.User(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        date_joined=user.date_joined.isoformat(),
    )


@router.post("/logout")
async def logout_view(response: Response):
    # Clear authentication cookie
    response.delete_cookie(key="session")
    return {"success": True}


@router.get("/user")
async def get_user_view(request: Request, user: Optional[User] = Depends(get_current_user_with_state)):
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


@router.get("/users")
async def list_users_view(session: SessionDep, user: User = Depends(get_current_admin)) -> view_models.UserList:
    from sqlmodel import select

    result = await session.execute(select(User))
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


@router.post("/change-password")
async def change_password_view(
    data: view_models.ChangePassword,
    session: SessionDep,
    user_manager: UserManager = Depends(get_user_manager),
    user: User = Depends(get_current_active_user),
):
    if user.username != data.username:
        raise HTTPException(status_code=400, detail="Username mismatch")

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


@router.delete("/users/{user_id}")
async def delete_user_view(user_id: int, session: SessionDep, user: User = Depends(get_current_admin)):
    from sqlmodel import select

    result = await session.execute(select(User).where(User.id == user_id))
    target = result.scalars().first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    admin_count = await query_admin_count(session)
    if target.role == Role.ADMIN and admin_count <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the last admin user")
    if target.id == user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    await db_delete_user(session, target)
    return {"message": "User deleted successfully"}
