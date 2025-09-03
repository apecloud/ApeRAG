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

from typing import Optional

from fastapi import APIRouter, Body, Depends, Response
from fastapi.responses import JSONResponse

from aperag.schema.view_models import Settings
from aperag.service.setting_service import setting_service
from aperag.views.auth import required_user

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
