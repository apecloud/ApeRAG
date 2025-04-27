import base64
import logging
import hashlib
from typing import Optional, Any
from channels.middleware import BaseMiddleware
from django.http import HttpRequest
from ninja.security import HttpBearer, APIKeyCookie
from django.core.cache import cache
from ninja.security.http import HttpAuthBase

from aperag.auth.validator import get_user_from_token, logger
from aperag.db.ops import get_api_key_by_key
from aperag.utils.constant import KEY_USER_ID, KEY_WEBSOCKET_PROTOCOL
import config.settings as settings
from aperag.auth import tv
from django.utils import timezone

logger = logging.getLogger(__name__)

DEFAULT_USER = "aperag"

def get_user_from_token(token: str) -> str:
    """Extract user ID from authentication token
    
    Parse token based on different auth types (auth0/authing/logto)
    to extract user identifier
    """
    match settings.AUTH_TYPE:
        case "auth0" | "authing" | "logto":
            payload = tv.verify(token)
            user = payload["sub"]
        case "none":
            user = base64.b64decode(token).decode("ascii")
        case _:
            user = DEFAULT_USER
    return user

async def get_user_from_api_key(key: str) -> Optional[str]:
    """Get user ID from API key
    
    Args:
        key: API key string
        
    Returns:
        str: User ID if key is valid
        None: If key is invalid or deleted
        
    Uses Redis cache to optimize performance and avoid frequent database queries
    """
    cache_key = f"api_key:{key}"
    user = cache.get(cache_key)
    if user is not None:
        return user

    api_key = await get_api_key_by_key(key)
    if not api_key or api_key.status == api_key.Status.DELETED:
        return None
    await api_key.update_last_used()

    cache.set(cache_key, api_key.user)
    return api_key.user

class BaseAuthBackend:
    """Base authentication class
    
    Provides common authentication functionality:
    1. Unified user information setting
    2. Authentication type identification
    3. Error handling
    """
    def set_user(self, request: HttpRequest, user_id: str) -> None:
        """Set user information in request
        
        Args:
            request: HTTP request object
            user_id: User ID
            
        Stores user ID in request.META for later use
        Also sets authentication type identifier
        """
        request.META[KEY_USER_ID] = user_id
        request.auth_type = self.__class__.__name__.lower().replace('auth', '')

class ApiKeyAuth(HttpBearer, BaseAuthBackend):
    """API Key authentication
    
    Validates API key and extracts user information
    """
    async def authenticate(self, request: HttpRequest, key: str) -> Optional[Any]:
        # Check if it's API key format
        if not key.startswith('sk-'):
            return None
            
        user = await get_user_from_api_key(key)
        if user:
            self.set_user(request, user)
            return key

        return None

class JWTAuth(HttpBearer, BaseAuthBackend):
    """JWT authentication
    
    Supports standard Bearer token format
    Used for auth0/authing/logto and other third-party auth services
    """
    async def authenticate(self, request: HttpRequest, token: str) -> Optional[Any]:
        # Skip API key format
        if token.startswith('sk-'):
            return None
            
        try:
            user = get_user_from_token(token)
            self.set_user(request, user)
            return token
        except Exception as e:
            logger.error(f"JWT authentication failed: {e}")
            return None

class SessionAuth(APIKeyCookie, BaseAuthBackend):
    """Cookie authentication
    
    Supports Django session authentication
    Used for web client user authentication after login
    """
    param_name = "sessionid"
    
    def __init__(self):
        super().__init__()
        self.openapi_name = "CookieAuth"
    
    async def authenticate(self, request: HttpRequest) -> Optional[Any]:
        user = await request.auser()
        if not user.is_authenticated:
            return None
        self.set_user(request, user.id)
        return user

class AdminAuth(HttpBearer, BaseAuthBackend):
    """Admin authentication
    
    Uses specific admin token for authentication
    Has highest authentication priority
    """
    async def authenticate(self, request: HttpRequest, token: str) -> Optional[Any]:
        if not settings.ADMIN_TOKEN or token != settings.ADMIN_TOKEN:
            return None

        self.set_user(request, settings.ADMIN_USER)
        return token

class GlobalAuth:
    """Global authentication class
    
    Combines multiple authentication methods with priority:
    1. Admin authentication (highest priority)
    2. API Key authentication
    3. JWT authentication
    4. Session authentication (lowest priority)
    
    Usage:
    1. Router level: router = Router(auth=GlobalAuth())
    2. View level: @router.get("/xxx", auth=GlobalAuth())
    """
    def __init__(self):
        self.admin_auth = AdminAuth()
        self.jwt_auth = JWTAuth()
        self.api_key_auth = ApiKeyAuth()
        self.session_auth = SessionAuth()

    async def __call__(self, request: HttpRequest) -> Optional[Any]:
        # 1. Try Bearer token authentication
        auth_header = request.headers.get('Authorization')
        if auth_header:
            parts = auth_header.split(' ')
            if len(parts) > 1 and parts[0].lower() == 'bearer':
                token = parts[1]
                # Try different auth methods by priority
                auth_methods = [
                    self.admin_auth,  # Admin auth has highest priority
                    self.api_key_auth,  # API key auth second
                    self.jwt_auth,  # JWT auth third
                ]
                for auth in auth_methods:
                    result = await auth.authenticate(request, token)
                    if result:
                        return result
                
        # 2. Try Session authentication
        result = await self.session_auth.authenticate(request)
        if result:
            return result
            
        return None

    @staticmethod
    def get_user_id(request: HttpRequest) -> Optional[str]:
        """Get current request's user ID
        
        Unified method to get user information, independent of auth method
        """
        return request.META.get(KEY_USER_ID)

    @staticmethod
    def get_auth_type(request: HttpRequest) -> Optional[str]:
        """Get current request's authentication type
        
        Possible values:
        - admin: Admin authentication
        - apikey: API Key authentication
        - jwt: JWT authentication
        - session: Session authentication
        """
        return getattr(request, 'auth_type', None) 


class FeishuEventVerification(HttpAuthBase):

    def __init__(self, encrypt_key=None):
        super().__init__()
        self.openapi_scheme = "feishu"
        self.header = "X-Lark-Signature"
        self.encrypt_key = encrypt_key

    def __call__(self, request: HttpRequest) -> Optional[Any]:
        if not self.encrypt_key:
            return True

        timestamp = request.headers.get("X-Lark-Request-Timestamp", None)
        if not timestamp:
            logger.error("Invalid timestamp in header: %s", request.headers)
            return False

        nonce = request.headers.get("X-Lark-Request-Nonce", None)
        if not nonce:
            logger.error("Invalid nonce in header: %s", request.headers)
            return False

        signature = request.headers.get("X-Lark-Signature", None)
        if not signature:
            logger.error("Invalid signature in header: %s", request.headers)
            return False

        bytes_b1 = (timestamp + nonce + self.encrypt_key).encode('utf-8')
        bytes_b = bytes_b1 + request.body
        h = hashlib.sha256(bytes_b)
        if h.hexdigest() != signature:
            logger.error("Invalid signature: %s", request)
            return False

        return True


class WebSocketAuthMiddleware(BaseMiddleware):
    def __init__(self, app):
        self.app = app

    def __call__(self, scope, receive, send):
        headers = dict(scope["headers"])
        path = scope['path']

        if "/web-chats" in path:
            return self.app(scope, receive, send)

        token = headers.get(KEY_WEBSOCKET_PROTOCOL.lower().encode("ascii"), None)
        if token is not None:
            token = token.decode("ascii")
        scope[KEY_WEBSOCKET_PROTOCOL] = token
        scope[KEY_USER_ID] = get_user_from_token(token)
        return self.app(scope, receive, send)