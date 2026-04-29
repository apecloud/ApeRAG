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

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from aperag.domains.governance.service.api_key_service import ApiKeyService
from aperag.views.utils import mask_api_key


def build_api_key(key: str = "sk-1234567890abcdef", description: str = "test key"):
    now = datetime.now(UTC)
    return SimpleNamespace(
        id="api-key-id",
        key=key,
        description=description,
        gmt_created=now,
        gmt_updated=now,
        last_used_at=None,
    )


def test_to_api_key_model_masks_by_default():
    service = ApiKeyService()
    token = build_api_key()

    result = service.to_api_key_model(token)

    assert result.key == mask_api_key(token.key)


async def test_list_api_keys_returns_masked_keys():
    service = ApiKeyService()
    token = build_api_key()
    service.db_ops = MagicMock()
    service.db_ops.query_api_keys = AsyncMock(return_value=[token])

    result = await service.list_api_keys("user-id")

    assert len(result.items) == 1
    assert result.items[0].key == mask_api_key(token.key)


async def test_create_api_key_returns_plaintext_key_once():
    service = ApiKeyService()
    token = build_api_key()
    service.db_ops = MagicMock()
    service.db_ops.create_api_key = AsyncMock(return_value=token)

    result = await service.create_api_key("user-id", SimpleNamespace(description="created key"))

    assert result.key == token.key


async def test_update_api_key_returns_masked_key():
    service = ApiKeyService()
    token = build_api_key(description="updated key")
    service.db_ops = MagicMock()
    service.db_ops.update_api_key_by_id = AsyncMock(return_value=token)

    result = await service.update_api_key("user-id", "api-key-id", SimpleNamespace(description="updated key"))

    assert result.key == mask_api_key(token.key)
