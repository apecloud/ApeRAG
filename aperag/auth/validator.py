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

import base64
import logging
from typing import Any, Optional

from channels.middleware import BaseMiddleware
from django.core.cache import cache
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest
from ninja.security.http import HttpAuthBase

import config.settings as settings
from aperag.auth import tv
from aperag.db.models import ApiKey
from aperag.utils.constant import KEY_USER_ID

logger = logging.getLogger(__name__)

DEFAULT_USER = "aperag"


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
        api_key = await ApiKey.objects.aget(key=key)
    except ApiKey.DoesNotExist:
        return None
    if api_key.status == ApiKey.Status.DELETED:
        return None

    cache.set(cache_key, api_key.user)
    return api_key.user


class GlobalHTTPAuth(HttpAuthBase):
    api_key_scheme: str = "api-key"
    openapi_scheme: str = "bearer"
    header: str = "Authorization"

    async def __call__(self, request: HttpRequest) -> Optional[Any]:
        headers = request.headers
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
            if user is None:
                return None
            request.META[KEY_USER_ID] = user
        return token


class HTTPAuthMiddleware(BaseMiddleware):
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        path = scope["path"]
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
