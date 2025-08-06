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

import json
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.ops import AsyncDatabaseOps, async_db_ops, db_ops


class SettingService:
    """Service for handling global settings"""

    def __init__(self, session: AsyncSession = None):
        if session is None:
            self.db_ops = async_db_ops
        else:
            self.db_ops = AsyncDatabaseOps(session)

    async def get_setting(self, key: str) -> Any | None:
        setting = await self.db_ops.query_setting(key)
        if not setting or setting.value is None:
            return None
        return json.loads(setting.value)

    async def update_setting(self, key: str, value: Any):
        await self.db_ops.update_setting(key, json.dumps(value))

    async def get_mineru_api_token(self) -> str | None:
        return await self.get_setting("mineru_api_token")

    async def update_mineru_api_token(self, token: str):
        await self.update_setting("mineru_api_token", token)

    async def get_use_mineru(self) -> bool:
        return await self.get_setting("use_mineru") or False

    async def update_use_mineru(self, use_mineru: bool):
        await self.update_setting("use_mineru", use_mineru)

    async def get_all_settings(self) -> dict:
        settings = await self.db_ops.query_all_settings()
        return {s.key: json.loads(s.value) for s in settings}

    def get_all_settings_sync(self) -> dict:
        settings = db_ops.query_all_settings()
        return {s.key: json.loads(s.value) for s in settings}

    async def update_settings(self, settings: dict):
        for key, value in settings.items():
            if value is not None:
                await self.update_setting(key, value)

    async def test_mineru_token(self, token: str) -> dict:
        """Test the MinerU API token."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "https://mineru.net/api/v4/extract-results/batch/test-token",
                    headers={"Authorization": f"Bearer {token}"},
                )
                return {"status_code": response.status_code, "data": response.json()}
            except httpx.RequestError as e:
                return {"status_code": 500, "data": {"msg": f"Request failed: {e}"}}


setting_service = SettingService()
