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

"""Latency logging middleware.

Measures wall-clock time for every HTTP request and emits a structured
``INFO`` log line.  This gives operators a lightweight, always-on view of
API performance without requiring an external tracing back-end.

Log format (one line per request)::

    INFO aperag.middleware.latency  GET /api/v1/bots 200 42ms

The ``X-Response-Time`` response header is also set so browser DevTools and
upstream proxies can surface the latency directly.
"""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths that generate a lot of noise but carry no useful perf signal.
_SKIP_PATHS = frozenset(["/health", "/docs", "/openapi.json", "/redoc"])


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that measures and logs request latency.

    For every request it:

    1. Records the wall-clock start time (``time.perf_counter``).
    2. Passes control to the next handler.
    3. Records the end time and computes ``duration_ms``.
    4. Emits an ``INFO`` log with ``method``, ``path``, ``status_code``, and
       ``duration_ms``.
    5. Attaches ``X-Response-Time: <duration_ms>ms`` to the response so that
       upstream load balancers and browser DevTools surface latency directly.

    Paths listed in ``_SKIP_PATHS`` (e.g. ``/health``) are processed but
    logged at ``DEBUG`` level to avoid flooding production logs.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        status_code = 500  # default in case call_next raises

        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            path = request.url.path

            msg = "%s %s %d %dms", request.method, path, status_code, duration_ms
            if path in _SKIP_PATHS:
                logger.debug(*msg)
            else:
                logger.info(*msg)

            # Expose latency to the caller via a response header.
            # We must read the response before adding headers; Starlette's
            # BaseHTTPMiddleware gives us the response object so this is safe.
            response.headers["X-Response-Time"] = f"{duration_ms}ms"

        return response
