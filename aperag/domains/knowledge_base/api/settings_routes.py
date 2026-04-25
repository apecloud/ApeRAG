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

"""System settings routes for the knowledge_base domain.

Hosts ``/settings`` endpoints related to document parser configuration
(MinerU token, parser health). Carved here from ``aperag/views/settings.py``
in Phase 8 task #48 (G2) per canonical D7-2: ``/api/v1/settings*`` is
hard-cut to ``/api/v2/settings*`` (see cleanup-inventory.md §3.2.2).

Backing service is ``aperag.domains.governance.service.setting_service``
because the underlying ``Setting`` ORM lives in the governance domain;
this module is a thin api layer that the knowledge_base domain owns
because the settings exposed here are doc-parser configuration that
the KB ingest pipeline consumes.
"""

from typing import Optional

from fastapi import APIRouter, Body, Depends, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from aperag.docparser.health import ParserHealthReport, get_parser_health_report
from aperag.domains.governance.service.setting_service import setting_service
from aperag.domains.identity.service.auth_dependencies import required_user


class Settings(BaseModel):
    """Knowledge-base parser settings request/response schema.

    Carved here from ``aperag.schema.view_models.Settings`` in #48 (G2)
    so the knowledge_base domain owns the shape directly and does not
    depend on the legacy aggregate ``aperag.schema.view_models``.
    """

    use_mineru: Optional[bool] = Field(None, description="Whether to use MinerU")
    mineru_api_token: Optional[str] = Field(None, description="API token for MinerU")
    use_markitdown: Optional[bool] = Field(None, description="Whether to use MarkItDown")


router = APIRouter()


@router.get("/settings", tags=["Settings"])
async def get_settings(user: dict = Depends(required_user)):
    settings = await setting_service.get_all_settings()
    return settings


@router.put("/settings", tags=["Settings"])
async def update_settings(
    settings: Settings,
    user: dict = Depends(required_user),
):
    await setting_service.update_settings(settings.model_dump())
    return Response(status_code=204)


@router.get("/settings/parser_health", tags=["Settings"], response_model=ParserHealthReport)
async def get_parser_health(user: dict = Depends(required_user)):
    current_settings = await setting_service.get_all_settings()
    return await get_parser_health_report(current_settings)


@router.post("/settings/test_mineru_token", tags=["Settings"])
async def test_mineru_token(
    token_data: Optional[dict] = Body(None),
    user: dict = Depends(required_user),
):
    token_to_test = None
    if token_data and "token" in token_data:
        token_to_test = token_data["token"]
    else:
        token_to_test = await setting_service.get_setting("mineru_api_token")

    if not token_to_test:
        return JSONResponse(
            status_code=404,
            content={"code": -1, "msg": "MinerU API token not set"},
        )

    result = await setting_service.test_mineru_token(token_to_test)
    return JSONResponse(status_code=200, content=result)
