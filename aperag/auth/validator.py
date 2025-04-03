import base64
import hashlib
import logging
from typing import Any, Optional

from channels.middleware import BaseMiddleware
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest
from ninja.security import HttpBearer
from ninja.security.http import HttpAuthBase

import config.settings as settings
from deeprag.auth import tv
from deeprag.utils.constant import KEY_USER_ID, KEY_WEBSOCKET_PROTOCOL
from deeprag.db.models import ApiKeyToken, ApiKeyStatus
from ninja.compatibility.request import get_headers
from django.core.cache import cache
from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)

DEFAULT_USER = "deeprag"


def get_user_from_token(token):
    match settings.AUTH_TYPE:
        case "auth0" | "authing" | "logto":
            payload = tv.verify(token)
            user = payload["sub"]
        case "none":
            user = base64.b64decode(token).decode("ascii")
        case _:
            user = DEFAULT_USER
    return user

async def get_user_from_api_key(key):
    cache_key = f"api_key:{key}"
    user = cache.get(cache_key)
    if user is not None:
        return user
    try:
        # api_key = ApiKeyToken.objects.get(key=key)
        api_key = await sync_to_async(ApiKeyToken.objects.get)(key=key)
    except ApiKeyToken.DoesNotExist:
        return None
    if api_key.status == ApiKeyStatus.DELETED:
        return None
    
    cache.set(cache_key, api_key.user)
    return api_key.user

class AdminAuth(HttpBearer):
    def authenticate(self, request, token):
        if not settings.ADMIN_TOKEN or token != settings.ADMIN_TOKEN:
            return None

        request.META[KEY_USER_ID] = settings.ADMIN_USER
        return token


class GlobalHTTPAuth(HttpAuthBase):
    api_key_scheme: str = "api-key"
    openapi_scheme: str = "bearer"
    header: str = "Authorization"
    async def __call__(self, request: HttpRequest) -> Optional[Any]:
        headers = get_headers(request)
        auth_value = headers.get(self.header)
        if not auth_value:
            return None
        parts = auth_value.split(" ")

        if parts[0].lower() not in (self.openapi_scheme, self.api_key_scheme):
            if settings.DEBUG:
                logger.error(f"Unexpected auth - '{auth_value}'")
            return None
        token = " ".join(parts[1:])
        return await self.authenticate(request, token, parts[0].lower())
    
    async def authenticate(self, request, token, scheme):
        if scheme == self.openapi_scheme:
            request.META[KEY_USER_ID] = get_user_from_token(token)
        elif scheme == self.api_key_scheme:
            user = await get_user_from_api_key(token)
            if user == None:
                return None
            request.META[KEY_USER_ID] = user
        return token


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


class HTTPAuthMiddleware(BaseMiddleware):
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        path = scope['path']
        if path == "/api/v1/config" or path.startswith("/api/v1/feishu"):
            return await self.inner(scope, receive, send)

        headers = dict(scope["headers"])
        token = headers.get(b"authorization", None)
        if token is None:
            raise PermissionDenied

        token = token.decode("ascii").lstrip("Bearer ")
        user = get_user_from_token(token)
        scope[KEY_USER_ID] = user
        return await self.inner(scope, receive, send)


