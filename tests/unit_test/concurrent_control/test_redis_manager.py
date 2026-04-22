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

import pytest

from aperag.db.redis_manager import RedisConnectionManager


class _DummyPool:
    def __init__(
        self,
        *,
        max_connections=20,
        created_connections=4,
        available_connections=None,
        in_use_connections=None,
    ):
        self.max_connections = max_connections
        self.created_connections = created_connections
        if available_connections is not None:
            self._available_connections = available_connections
        if in_use_connections is not None:
            self._in_use_connections = in_use_connections


@pytest.fixture(autouse=True)
def reset_redis_pools():
    old_async_pool = RedisConnectionManager._async_pool
    old_sync_pool = RedisConnectionManager._sync_pool
    RedisConnectionManager._async_pool = None
    RedisConnectionManager._sync_pool = None
    try:
        yield
    finally:
        RedisConnectionManager._async_pool = old_async_pool
        RedisConnectionManager._sync_pool = old_sync_pool


def test_get_pool_info_returns_not_initialized_when_no_pool():
    assert RedisConnectionManager.get_pool_info() == {"status": "not_initialized"}


def test_get_pool_info_reads_public_fields_without_private_internals():
    RedisConnectionManager._sync_pool = _DummyPool()

    assert RedisConnectionManager.get_pool_info() == {
        "sync_pool": {
            "max_connections": 20,
            "created_connections": 4,
        }
    }


def test_get_pool_info_counts_private_connection_lists_when_present():
    RedisConnectionManager._async_pool = _DummyPool(
        max_connections=10,
        created_connections=6,
        available_connections=["a", "b"],
        in_use_connections=["c"],
    )

    assert RedisConnectionManager.get_pool_info() == {
        "async_pool": {
            "max_connections": 10,
            "created_connections": 6,
            "available_connections": 2,
            "in_use_connections": 1,
        }
    }


def test_get_pool_info_tolerates_non_sized_private_internals():
    RedisConnectionManager._sync_pool = _DummyPool(
        available_connections=object(),
        in_use_connections=object(),
    )

    assert RedisConnectionManager.get_pool_info() == {
        "sync_pool": {
            "max_connections": 20,
            "created_connections": 4,
        }
    }
