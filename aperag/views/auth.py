from django.contrib.auth import authenticate, login, logout
from ninja import Router
from ninja.security import django_auth
from aperag.db.utils import User
import secrets
from django.core.mail import send_mail
from django.conf import settings
from django.urls import reverse
from django.utils import timezone
from ninja.pagination import paginate
from ninja import Query
from typing import Optional, List
from asgiref.sync import sync_to_async
from django.db import models
from ..db.ops import (
    query_user_exists, query_first_user_exists, create_user,
    query_invitation_by_token, create_invitation, mark_invitation_used,
    authenticate_user, login_user, logout_user, set_user_password,
    delete_user, query_users, query_staff_count
)
import aperag.views.models as view_models
from aperag.views.utils import success,fail, auth_middleware
from http import HTTPStatus

router = Router()

@router.post("/invite", auth=auth_middleware)
async def create_invitation_view(request, data: view_models.InvitationCreate) -> view_models.Invitation:
    """Create a new invitation"""
    user = await request.auser()
    if not user.is_staff:
        return fail(HTTPStatus.FORBIDDEN, "Only staff members can create invitations")
        
    # Check if user already exists
    if await query_user_exists(email=data.email):
        return fail(HTTPStatus.BAD_REQUEST, "User with this email already exists")
        
    # Generate unique token
    token = secrets.token_urlsafe(32)
    
    # Create invitation
    invitation = await create_invitation(
        email=data.email,
        token=token,
        created_by=user
    )
    
    # Send invitation email
    invitation_url = f"{settings.SITE_URL}/register?token={token}"
    await sync_to_async(send_mail)(
        'Invitation to join ApeRAG',
        f'You have been invited to join ApeRAG. Please use this link to register: {invitation_url}',
        settings.DEFAULT_FROM_EMAIL,
        [data.email],
        fail_silently=True,
    )
    
    return success(view_models.Invitation(
        email=invitation.email,
        url=invitation_url,
        created_by=user.id,
        created_at=invitation.created_at.isoformat(),
        is_valid=invitation.is_valid(),
        used_at=invitation.used_at
    ))


@router.post("/register")
async def register(request, data: view_models.Register) -> view_models.User:
    """Register a new user with invitation token"""
    # Check if this is the first user (will be admin)
    is_first_user = not await query_first_user_exists()
    
    if not is_first_user:
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
        is_staff=is_first_user,  # First user will be staff/admin
        is_superuser=is_first_user  # First user will be superuser
    )
    
    # If not first user, mark invitation as used
    if not is_first_user:
        await mark_invitation_used(invitation)
    
    return success(view_models.User(
        username=user.username,
        email=user.email,
        is_staff=user.is_staff,
        is_active=user.is_active,
        date_joined=user.date_joined.isoformat(),
    ))

@router.post("/login")
async def login_view(request, data: view_models.Login) -> view_models.User:
    """Login a user"""
    user = await authenticate_user(request, username=data.username, password=data.password)
    
    if user is None:
        return fail(HTTPStatus.BAD_REQUEST, "Invalid credentials")
        
    await login_user(request, user)
    
    return success(view_models.User(
        username=user.username,
        email=user.email,
        is_staff=user.is_staff,
        is_active=user.is_active,
        date_joined=user.date_joined.isoformat(),
    ))

@router.post("/logout", auth=auth_middleware)
async def logout_view(request):
    """Logout a user"""
    await logout_user(request)
    return success({})

@router.get("/users", auth=auth_middleware)
async def list_users(request) -> List[view_models.User]:
    """List all users (admin only)"""
    user = await request.auser()
    if not user.is_staff:
        return success([])
        
    result = query_users()
    
    return success([
        view_models.User(
            username=user.username,
            email=user.email,
            is_staff=user.is_staff,
            is_active=user.is_active,
            date_joined=user.date_joined.isoformat()
        )
        for user in result
    ])

@router.post("/change-password")
def change_password(request, data: view_models.ChangePassword) -> view_models.User:
    """Change user password"""
    user = authenticate(request, username=data.username, password=data.old_password)
    
    if user is None:
        return fail(HTTPStatus.BAD_REQUEST, "Current password is incorrect")
        
    set_user_password(user, data.new_password)
    
    return success(view_models.User(
        username=user.username,
        email=user.email,
        is_staff=user.is_staff,
        is_active=user.is_active,
        date_joined=user.date_joined.isoformat()
    ))

@router.delete("/users/{user_id}", auth=auth_middleware)
async def delete_user_view(request, user_id: int) -> view_models.User:
    """Delete a user (admin only)"""
    user = await request.auser()
    if not user.is_staff:
        return fail(HTTPStatus.FORBIDDEN, "Only staff members can delete users")
        
    try:
        user = await User.objects.aget(id=user_id)
    except User.DoesNotExist:
        return fail(HTTPStatus.NOT_FOUND, "User not found")
        
    # Prevent deleting the last admin
    staff_count = await query_staff_count()
    if user.is_staff and staff_count <= 1:
        return fail(HTTPStatus.BAD_REQUEST, "Cannot delete the last admin user")
        
    # Prevent deleting yourself
    if user.username == user.username:
        return fail(HTTPStatus.BAD_REQUEST, "Cannot delete your own account")
        
    await delete_user(user)
    
    return success(view_models.User(message="User deleted successfully")) 

