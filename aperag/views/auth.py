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
from http import HTTPStatus

from django.conf import settings
from django.contrib.auth import aauthenticate
from ninja import Router

from aperag.db.models import Role, User
from aperag.schema import view_models
from aperag.views.utils import auth_middleware, fail, success

from ..db.ops import (
    create_invitation,
    create_user,
    delete_user,
    login_user,
    logout_user,
    mark_invitation_used,
    query_admin_count,
    query_first_user_exists,
    query_invitation_by_token,
    query_invitations,
    query_user_exists,
    query_users,
    set_user_password,
)

router = Router()


@router.post("/invite", auth=auth_middleware)
async def create_invitation_view(request, data: view_models.InvitationCreate) -> view_models.Invitation:
    """Create a new invitation"""
    user = await request.auser()
    if not user.role == Role.ADMIN:
        return fail(HTTPStatus.FORBIDDEN, "Only admin members can create invitations")

    # Check if user already exists
    if await query_user_exists(username=data.username, email=data.email):
        return fail(HTTPStatus.BAD_REQUEST, "User with this email or username already exists")

    # Generate unique token
    token = secrets.token_urlsafe(32)

    # Create invitation
    invitation = await create_invitation(
        email=data.email,
        token=token,
        created_by=user.id,
        role=data.role,  # Add role to invitation
    )

    # Send invitation email
    # invitation_url = f"{settings.SITE_URL}/register?token={token}"
    # await sync_to_async(send_mail)(
    #     'Invitation to join ApeRAG',
    #     f'You have been invited to join ApeRAG. Please use this link to register: {invitation_url}',
    #     settings.DEFAULT_FROM_EMAIL,
    #     [data.email],
    #     fail_silently=True,
    # )

    return success(
        view_models.Invitation(
            email=invitation.email,
            token=token,
            created_by=user.id,
            created_at=invitation.created_at.isoformat(),
            is_valid=invitation.is_valid(),
            role=invitation.role,
        )
    )


@router.get("/invitations", auth=auth_middleware)
async def list_invitations_view(request) -> view_models.InvitationList:
    """List all invitations (admin only)"""
    user = await request.auser()
    if not user.role == Role.ADMIN:
        return fail(HTTPStatus.FORBIDDEN, "Only admin members can view invitations")

    invitations = []
    async for invitation in await query_invitations():
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
    return success(view_models.InvitationList(items=invitations))


@router.post("/register")
async def register_view(request, data: view_models.Register) -> view_models.User:
    """Register a new user with invitation token"""
    # Check if this is the first user (will be admin)
    is_first_user = not await query_first_user_exists()
    need_invitation = settings.REGISTER_MODE == "invitation" and not is_first_user

    if need_invitation:
        # For non-first users, validate invitation
        invitation = await query_invitation_by_token(data.token)
        if not invitation:
            return fail(HTTPStatus.BAD_REQUEST, "Invalid invitation token")

        if not invitation.is_valid():
            return fail(HTTPStatus.BAD_REQUEST, "Invitation has expired or has been used")

        if invitation.email != data.email:
            return fail(HTTPStatus.BAD_REQUEST, "Email does not match invitation")

    if await query_user_exists(username=data.username):
        return fail(HTTPStatus.BAD_REQUEST, "Username already exists")

    if await query_user_exists(email=data.email):
        return fail(HTTPStatus.BAD_REQUEST, "Email already exists")

    # Create user
    user = await create_user(
        username=data.username,
        email=data.email,
        password=data.password,
        role=invitation.role if need_invitation else Role.ADMIN,
    )

    # If not first user, mark invitation as used
    if need_invitation:
        await mark_invitation_used(invitation)

    return success(
        view_models.User(
            id=str(user.id),
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            date_joined=user.date_joined.isoformat(),
        )
    )


@router.post("/login")
async def login_view(request, data: view_models.Login) -> view_models.User:
    """Login a user"""
    user = await aauthenticate(username=data.username, password=data.password)
    if not user:
        return fail(HTTPStatus.BAD_REQUEST, "Invalid credentials")

    await login_user(request, user)

    return success(
        view_models.User(
            id=str(user.id),
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            date_joined=user.date_joined.isoformat(),
        )
    )


@router.post("/logout", auth=auth_middleware)
async def logout_view(request):
    """Logout a user"""
    await logout_user(request)
    return success({})


@router.get("/user", auth=auth_middleware)
async def get_user_view(request) -> view_models.User:
    """Get user info"""
    user = await request.auser()
    return success(
        view_models.User(
            id=str(user.id),
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            date_joined=user.date_joined.isoformat(),
        )
    )


@router.get("/users", auth=auth_middleware)
async def list_users_view(request) -> view_models.UserList:
    """List all users (admin only)"""
    user = await request.auser()
    if not user.role == Role.ADMIN:
        return success([])

    result = query_users()

    users = []
    async for user in result:
        users.append(
            view_models.User(
                id=str(user.id),
                username=user.username,
                email=user.email,
                role=user.role,
                is_active=user.is_active,
                date_joined=user.date_joined.isoformat(),
            )
        )
    return success(view_models.UserList(items=users))


@router.post("/change-password")
async def change_password_view(request, data: view_models.ChangePassword) -> view_models.User:
    """Change user password"""
    user = await aauthenticate(username=data.username, password=data.old_password)

    if user is None:
        return fail(HTTPStatus.BAD_REQUEST, "Current password is incorrect")

    await set_user_password(user, data.new_password)

    return success(
        view_models.User(
            id=str(user.id),
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            date_joined=user.date_joined.isoformat(),
        )
    )


@router.delete("/users/{user_id}", auth=auth_middleware)
async def delete_user_view(request, user_id: int) -> view_models.User:
    """Delete a user (admin only)"""
    user = await request.auser()
    if not user.role == Role.ADMIN:
        return fail(HTTPStatus.FORBIDDEN, "Only admin members can delete users")

    try:
        user = await User.objects.aget(id=user_id)
    except User.DoesNotExist:
        return fail(HTTPStatus.NOT_FOUND, "User not found")

    # Prevent deleting the last admin
    admin_count = await query_admin_count()
    if user.role == Role.ADMIN and admin_count <= 1:
        return fail(HTTPStatus.BAD_REQUEST, "Cannot delete the last admin user")

    # Prevent deleting yourself
    if user.username == user.username:
        return fail(HTTPStatus.BAD_REQUEST, "Cannot delete your own account")

    await delete_user(user)

    return success(view_models.User(message="User deleted successfully"))
