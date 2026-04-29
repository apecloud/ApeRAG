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

import asyncio
import hmac
import os
import time
from typing import Any

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text

from aperag.config import get_sync_database_url, settings

DIAGNOSTICS_TOKEN_ENV = "HEALTH_DIAGNOSTICS_TOKEN"
DIAGNOSTICS_TIMEOUT_SECONDS = 2.0
DIAGNOSTICS_DB_POOL_TIMEOUT_SECONDS = 1
DIAGNOSTICS_DB_CONNECT_TIMEOUT_SECONDS = 1

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Legacy compatibility endpoint for existing probes."""
    return {"status": "healthy", "service": "aperag-api"}


@router.get("/health/live")
async def live_check() -> dict[str, str]:
    return {"status": "live", "service": "aperag-api"}


@router.get("/health/ready")
async def ready_check() -> dict[str, str]:
    return {"status": "ready", "service": "aperag-api"}


@router.get("/health/diagnostics", include_in_schema=False)
async def diagnostics_check(x_internal_token: str | None = Header(default=None)) -> JSONResponse:
    token = os.getenv(DIAGNOSTICS_TOKEN_ENV)
    if not token:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "service": "aperag-api",
                "detail": f"{DIAGNOSTICS_TOKEN_ENV} is not configured",
            },
        )
    if not x_internal_token or not hmac.compare_digest(x_internal_token, token):
        return JSONResponse(
            status_code=401,
            content={"status": "unauthorized", "service": "aperag-api"},
        )

    started = time.perf_counter()
    database = await _run_database_probe()
    status = "healthy" if database["ok"] else "unhealthy"
    response_status = 200 if database["ok"] else 503
    return JSONResponse(
        status_code=response_status,
        content={
            "status": status,
            "service": "aperag-api",
            "checks": {"database": database},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )


async def _run_database_probe() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_probe_database),
            timeout=DIAGNOSTICS_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        return {
            "ok": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": f"database probe timed out after {DIAGNOSTICS_TIMEOUT_SECONDS}s",
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "ok": True,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _probe_database() -> None:
    sync_database_url = get_sync_database_url(settings.database_url)
    engine_kwargs: dict[str, Any] = {
        "echo": settings.debug,
        "pool_pre_ping": True,
    }
    if not sync_database_url.startswith("sqlite"):
        engine_kwargs.update(
            {
                "pool_size": 1,
                "max_overflow": 0,
                "pool_timeout": DIAGNOSTICS_DB_POOL_TIMEOUT_SECONDS,
                "connect_args": {"connect_timeout": DIAGNOSTICS_DB_CONNECT_TIMEOUT_SECONDS},
            }
        )

    engine = create_engine(sync_database_url, **engine_kwargs)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    finally:
        engine.dispose()

